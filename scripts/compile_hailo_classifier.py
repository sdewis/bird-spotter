#!/usr/bin/env python3
"""Export birder-project/regnet_y_1_6g_eu-common (.pt) to ONNX and compile to Hailo-8 (.hef).

Supports:
1. Zero-dependency direct export from PyTorch checkpoint (.pt ZIP) to fused ONNX
   (`models/uk_birds_regnet_y_1_6g.onnx`), verified with OpenCV DNN.
2. Remote or local x86_64 Hailo Dataflow Compiler execution (`--compile-hef`) to build
   `models/uk_birds_regnet_y_1_6g_h8.hef` on an x86_64 host (e.g. sean@192.168.100.101).
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import io
import json
from pathlib import Path
import pickle
import struct
import subprocess
import time
import zipfile

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
PT_PATH = MODELS_DIR / "regnet_y_1_6g_eu-common.pt"
ONNX_PATH = MODELS_DIR / "uk_birds_regnet_y_1_6g.onnx"
HEF_PATH = MODELS_DIR / "uk_birds_regnet_y_1_6g_h8.hef"
LABELS_PATH = MODELS_DIR / "eu_common_labels.json"


# --- Minimal Zero-Dependency ONNX Protobuf Wire Encoder ---


def _encode_varint(val: int) -> bytes:
    if val < 0:
        val += 1 << 64
    out = bytearray()
    while val >= 0x80:
        out.append((val & 0x7F) | 0x80)
        val >>= 7
    out.append(val & 0x7F)
    return bytes(out)


def _tag(field_num: int, wire_type: int) -> bytes:
    return _encode_varint((field_num << 3) | wire_type)


def _field_varint(field_num: int, val: int) -> bytes:
    return _tag(field_num, 0) + _encode_varint(val)


def _field_bytes(field_num: int, data: bytes) -> bytes:
    return _tag(field_num, 2) + _encode_varint(len(data)) + data


def _field_str(field_num: int, text: str) -> bytes:
    return _field_bytes(field_num, text.encode("utf-8"))


def _make_tensor_proto(name: str, arr: np.ndarray) -> bytes:
    arr_f32 = np.ascontiguousarray(arr, dtype="<f4")
    buf = bytearray()
    for d in arr_f32.shape:
        buf += _field_varint(1, int(d))
    buf += _field_varint(2, 1)  # data_type = 1 (FLOAT)
    buf += _field_str(8, name)
    buf += _field_bytes(9, arr_f32.tobytes())
    return bytes(buf)


def _make_attr_int(name: str, val: int) -> bytes:
    buf = bytearray()
    buf += _field_str(1, name)
    buf += _field_varint(3, int(val))
    buf += _field_varint(20, 2)  # AttributeType.INT = 2
    return bytes(buf)


def _make_attr_ints(name: str, vals: list[int]) -> bytes:
    buf = bytearray()
    buf += _field_str(1, name)
    for v in vals:
        buf += _field_varint(8, int(v))
    buf += _field_varint(20, 7)  # AttributeType.INTS = 7
    return bytes(buf)


def _make_attr_float(name: str, val: float) -> bytes:
    buf = bytearray()
    buf += _field_str(1, name)
    buf += _tag(2, 5) + struct.pack("<f", float(val))
    buf += _field_varint(20, 1)  # AttributeType.FLOAT = 1
    return bytes(buf)


def _make_node_proto(
    op_type: str,
    inputs: list[str],
    outputs: list[str],
    name: str,
    attrs: list[bytes] | None = None,
) -> bytes:
    buf = bytearray()
    for inp in inputs:
        buf += _field_str(1, inp)
    for out in outputs:
        buf += _field_str(2, out)
    buf += _field_str(3, name)
    buf += _field_str(4, op_type)
    for a in attrs or []:
        buf += _field_bytes(5, a)
    return bytes(buf)


def _make_value_info(name: str, shape: list[int]) -> bytes:
    shape_buf = bytearray()
    for d in shape:
        dim_msg = _field_varint(1, int(d))
        shape_buf += _field_bytes(1, dim_msg)
    tensor_type_msg = _field_varint(1, 1) + _field_bytes(2, bytes(shape_buf))
    type_msg = _field_bytes(1, tensor_type_msg)
    return _field_str(1, name) + _field_bytes(2, type_msg)


def _make_model_proto(
    nodes: list[bytes],
    initializers: list[bytes],
    inputs: list[bytes],
    outputs: list[bytes],
    opset_version: int = 13,
) -> bytes:
    graph_buf = bytearray()
    for n in nodes:
        graph_buf += _field_bytes(1, n)
    graph_buf += _field_str(2, "regnet_y_1_6g_eu_common")
    for init in initializers:
        graph_buf += _field_bytes(5, init)
    for inp in inputs:
        graph_buf += _field_bytes(11, inp)
    for out in outputs:
        graph_buf += _field_bytes(12, out)

    opset_msg = _field_str(1, "") + _field_varint(2, opset_version)
    model_buf = bytearray()
    model_buf += _field_varint(1, 7)  # ir_version = 7
    model_buf += _field_str(2, "bird_spotter_exporter")
    model_buf += _field_bytes(7, bytes(graph_buf))
    model_buf += _field_bytes(8, opset_msg)
    return bytes(model_buf)


# --- Load PyTorch .pt Checkpoint State Dict Without PyTorch ---


def load_pt_state_dict(pt_path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Read tensors and metadata from a PyTorch .pt ZIP archive using only numpy and zipfile."""

    class PtUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str):
            if module == "collections" and name == "OrderedDict":
                return OrderedDict
            if name == "_rebuild_tensor_v2":
                return lambda storage, offset, size, stride, *args: (
                    "TENSOR",
                    storage,
                    offset,
                    size,
                    stride,
                )
            return lambda *args, **kwargs: (module, name, args)

        def persistent_load(self, pid):
            return pid

    with zipfile.ZipFile(pt_path, "r") as zf:
        namelist = zf.namelist()
        prefix = namelist[0].split("/")[0]
        raw_pkl = zf.read(f"{prefix}/data.pkl")
        checkpoint = PtUnpickler(io.BytesIO(raw_pkl)).load()

        state_meta = checkpoint["state"]
        tensors: dict[str, np.ndarray] = {}

        for key, val in state_meta.items():
            _, storage_info, offset, size, _stride = val
            storage_type_tuple = storage_info[1]
            storage_name = storage_type_tuple[1] if isinstance(storage_type_tuple, tuple) else str(storage_type_tuple)
            storage_key = storage_info[2]
            raw_bytes = zf.read(f"{prefix}/data/{storage_key}")

            dtype = np.int64 if "Long" in storage_name else np.float32
            count = int(np.prod(size)) if len(size) > 0 else 1
            arr = np.frombuffer(
                raw_bytes, dtype=dtype, count=count, offset=offset * np.dtype(dtype).itemsize
            ).reshape(size)
            tensors[key] = arr.copy()

    return tensors, checkpoint


def fuse_conv_bn(
    conv_w: np.ndarray,
    bn_w: np.ndarray,
    bn_b: np.ndarray,
    bn_mean: np.ndarray,
    bn_var: np.ndarray,
    eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse Conv2d weight (O, I/g, kH, kW) and BatchNorm2d parameters into (w_fused, b_fused)."""
    scale = bn_w / np.sqrt(bn_var + eps)
    w_fused = conv_w * scale.reshape(-1, 1, 1, 1)
    b_fused = bn_b - bn_mean * scale
    return w_fused.astype(np.float32), b_fused.astype(np.float32)


def export_regnet_to_onnx(pt_path: Path, onnx_path: Path) -> None:
    """Build fused ONNX graph for birder-project/regnet_y_1_6g_eu-common."""
    print(f"[1/3] Loading PyTorch checkpoint from {pt_path}...")
    sd, _ = load_pt_state_dict(pt_path)

    nodes: list[bytes] = []
    initializers: list[bytes] = []

    def add_fused_conv(
        inp: str,
        out: str,
        prefix_conv: str,
        prefix_bn: str,
        kernel: int,
        stride: int,
        pad: int,
        groups: int = 1,
    ) -> str:
        w, b = fuse_conv_bn(
            sd[f"{prefix_conv}.weight"],
            sd[f"{prefix_bn}.weight"],
            sd[f"{prefix_bn}.bias"],
            sd[f"{prefix_bn}.running_mean"],
            sd[f"{prefix_bn}.running_var"],
        )
        w_name = f"{out}_w"
        b_name = f"{out}_b"
        initializers.append(_make_tensor_proto(w_name, w))
        initializers.append(_make_tensor_proto(b_name, b))
        attrs = [
            _make_attr_ints("kernel_shape", [kernel, kernel]),
            _make_attr_ints("strides", [stride, stride]),
            _make_attr_ints("pads", [pad, pad, pad, pad]),
            _make_attr_int("group", groups),
        ]
        nodes.append(
            _make_node_proto("Conv", [inp, w_name, b_name], [out], name=f"conv_{out}", attrs=attrs)
        )
        return out

    def add_relu(inp: str, out: str) -> str:
        nodes.append(_make_node_proto("Relu", [inp], [out], name=f"relu_{out}"))
        return out

    # 1. Stem: 3x3 Conv stride 2 + BN + ReLU
    x = add_fused_conv("input", "stem_conv", "stem.0", "stem.1", kernel=3, stride=2, pad=1)
    x = add_relu(x, "stem_relu")

    # 2. Body stages 1..4
    for stage_idx in range(1, 5):
        stage_prefix = f"body.stage{stage_idx}"
        # Find block indices in this stage
        block_indices = sorted(
            {
                int(k.split(".")[3])
                for k in sd.keys()
                if k.startswith(f"{stage_prefix}.block.")
            }
        )
        for blk_idx in block_indices:
            b_pref = f"{stage_prefix}.block.{blk_idx}"
            tag = f"s{stage_idx}_b{blk_idx}"
            stride = 2 if blk_idx == 0 else 1

            # Residual branch
            if f"{b_pref}.proj.0.weight" in sd:
                res = add_fused_conv(
                    x,
                    f"{tag}_proj",
                    f"{b_pref}.proj.0",
                    f"{b_pref}.proj.1",
                    kernel=1,
                    stride=stride,
                    pad=0,
                )
            else:
                res = x

            # Main branch: f.block.0 (1x1 Conv + BN + ReLU)
            f0 = add_fused_conv(
                x,
                f"{tag}_f0",
                f"{b_pref}.f.block.0.0",
                f"{b_pref}.f.block.0.1",
                kernel=1,
                stride=1,
                pad=0,
            )
            f0 = add_relu(f0, f"{tag}_f0_relu")

            # f.block.1 (3x3 Grouped Conv + BN + ReLU)
            w1_shape = sd[f"{b_pref}.f.block.1.0.weight"].shape
            out_ch, in_per_grp = int(w1_shape[0]), int(w1_shape[1])
            groups = out_ch // in_per_grp
            f1 = add_fused_conv(
                f0,
                f"{tag}_f1",
                f"{b_pref}.f.block.1.0",
                f"{b_pref}.f.block.1.1",
                kernel=3,
                stride=stride,
                pad=1,
                groups=groups,
            )
            f1 = add_relu(f1, f"{tag}_f1_relu")

            # f.block.2 (Squeeze-and-Excitation: GlobalAveragePool -> fc1 -> ReLU -> fc2 -> Sigmoid -> Mul)
            se_pool = f"{tag}_se_pool"
            nodes.append(
                _make_node_proto("GlobalAveragePool", [f1], [se_pool], name=f"gap_{tag}")
            )

            fc1_w = sd[f"{b_pref}.f.block.2.fc1.weight"]
            fc1_b = sd[f"{b_pref}.f.block.2.fc1.bias"]
            initializers.append(_make_tensor_proto(f"{tag}_fc1_w", fc1_w))
            initializers.append(_make_tensor_proto(f"{tag}_fc1_b", fc1_b))
            se_fc1 = f"{tag}_se_fc1"
            nodes.append(
                _make_node_proto(
                    "Conv",
                    [se_pool, f"{tag}_fc1_w", f"{tag}_fc1_b"],
                    [se_fc1],
                    name=f"conv_{se_fc1}",
                    attrs=[
                        _make_attr_ints("kernel_shape", [1, 1]),
                        _make_attr_ints("strides", [1, 1]),
                        _make_attr_ints("pads", [0, 0, 0, 0]),
                    ],
                )
            )
            se_relu = add_relu(se_fc1, f"{tag}_se_relu")

            fc2_w = sd[f"{b_pref}.f.block.2.fc2.weight"]
            fc2_b = sd[f"{b_pref}.f.block.2.fc2.bias"]
            initializers.append(_make_tensor_proto(f"{tag}_fc2_w", fc2_w))
            initializers.append(_make_tensor_proto(f"{tag}_fc2_b", fc2_b))
            se_fc2 = f"{tag}_se_fc2"
            nodes.append(
                _make_node_proto(
                    "Conv",
                    [se_relu, f"{tag}_fc2_w", f"{tag}_fc2_b"],
                    [se_fc2],
                    name=f"conv_{se_fc2}",
                    attrs=[
                        _make_attr_ints("kernel_shape", [1, 1]),
                        _make_attr_ints("strides", [1, 1]),
                        _make_attr_ints("pads", [0, 0, 0, 0]),
                    ],
                )
            )
            se_sig = f"{tag}_se_sig"
            nodes.append(_make_node_proto("Sigmoid", [se_fc2], [se_sig], name=f"sig_{tag}"))

            se_mul = f"{tag}_se_mul"
            nodes.append(_make_node_proto("Mul", [f1, se_sig], [se_mul], name=f"mul_{tag}"))

            # f.block.3 (1x1 Conv + BN)
            f3 = add_fused_conv(
                se_mul,
                f"{tag}_f3",
                f"{b_pref}.f.block.3.0",
                f"{b_pref}.f.block.3.1",
                kernel=1,
                stride=1,
                pad=0,
            )

            # Residual Add + ReLU
            blk_add = f"{tag}_add"
            nodes.append(_make_node_proto("Add", [res, f3], [blk_add], name=f"add_{tag}"))
            x = add_relu(blk_add, f"{tag}_out")

    # 3. Head: GlobalAveragePool -> Flatten -> Gemm -> output (1, 707)
    pool_out = "head_pool"
    nodes.append(_make_node_proto("GlobalAveragePool", [x], [pool_out], name="head_gap"))
    flat_out = "head_flat"
    nodes.append(
        _make_node_proto(
            "Flatten", [pool_out], [flat_out], name="head_flatten", attrs=[_make_attr_int("axis", 1)]
        )
    )

    cls_w = sd["classifier.weight"]
    cls_b = sd["classifier.bias"]
    initializers.append(_make_tensor_proto("cls_w", cls_w))
    initializers.append(_make_tensor_proto("cls_b", cls_b))
    nodes.append(
        _make_node_proto(
            "Gemm",
            [flat_out, "cls_w", "cls_b"],
            ["output"],
            name="head_gemm",
            attrs=[
                _make_attr_float("alpha", 1.0),
                _make_attr_float("beta", 1.0),
                _make_attr_int("transB", 1),
            ],
        )
    )

    model_bytes = _make_model_proto(
        nodes=nodes,
        initializers=initializers,
        inputs=[_make_value_info("input", [1, 3, 256, 256])],
        outputs=[_make_value_info("output", [1, 707])],
        opset_version=13,
    )

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path.write_bytes(model_bytes)
    print(f"[2/3] Exported fused ONNX model to {onnx_path} ({len(model_bytes) / 1e6:.2f} MB)")


def verify_onnx_with_opencv(onnx_path: Path) -> None:
    """Verify the exported ONNX model runs cleanly in OpenCV DNN on Raspberry Pi 5."""
    print(f"[3/3] Verifying {onnx_path.name} with OpenCV DNN...")
    net = cv2.dnn.readNetFromONNX(str(onnx_path))
    dummy = np.zeros((1, 3, 256, 256), dtype=np.float32)
    t0 = time.perf_counter()
    net.setInput(dummy)
    out = net.forward()
    ms = (time.perf_counter() - t0) * 1000.0
    print(f"      OpenCV DNN inference OK! Output shape: {out.shape}, latency: {ms:.1f} ms")


def compile_hef_on_remote_x86(host: str, user: str, onnx_path: Path, hef_path: Path) -> int:
    """Copy ONNX model to remote x86_64 host and compile .hef using Hailo Dataflow Compiler."""
    target = f"{user}@{host}"
    print(f"[Hailo DFC] Checking SSH connectivity to {target}...")
    ping = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", target, "uname -a"],
        capture_output=True,
        text=True,
    )
    if ping.returncode != 0:
        print(
            f"[Hailo DFC] Host {target} is currently unreachable ({ping.stderr.strip()}).\n"
            f"            Wake {host} and re-run:\n"
            f"            python3 scripts/compile_hailo_classifier.py --compile-hef --host {host}"
        )
        return 1

    print(f"[Hailo DFC] Connected: {ping.stdout.strip()}")
    subprocess.run(["ssh", target, "mkdir -p ~/hailo_build/work"], check=True)
    subprocess.run(["scp", str(onnx_path), f"{target}:~/hailo_build/work/model.onnx"], check=True)

    remote_script = r"""
import os, json, urllib.request, subprocess, numpy as np

base_dir = os.path.expanduser("~/hailo_build")
layers_dir = os.path.join(base_dir, "layers")
rootfs_dir = os.path.join(base_dir, "rootfs")
work_dir = os.path.join(base_dir, "work")
os.makedirs(layers_dir, exist_ok=True)
os.makedirs(rootfs_dir, exist_ok=True)

# 1. Ensure Hailo DFC rootfs is extracted
if not os.path.exists(os.path.join(rootfs_dir, "usr/local/bin/python3")):
    token_url = "https://auth.docker.io/token?service=registry.docker.io&scope=repository:margisbench/margisbench_hailo_compiler:pull"
    token = json.loads(urllib.request.urlopen(token_url).read().decode())["token"]
    layers = [
        "sha256:0cd34018496fb81f0f686be1fc702a09c2f3c5e6dcfa4800f2a4b95ce62105b1",
        "sha256:83d2b5fd4a0a66b31f129f92ed2bd6a7cfbb95a7f57bd85fd2443501c06a7802",
        "sha256:801d24a6e65d489ef9bc24537a6f07870f480ba7245554909aa5bf0451bb7a20",
        "sha256:7fdb90c67c986e65653a463f3473744c669162cf833c8e490c45fb947c962b66",
        "sha256:f5c064e8c9a7b6f9d4dfdc4bcbb5de1ce5fc91ecc8c667a6c2e54f129d238c04",
        "sha256:b5e23e8a0b36a8b6d6832189b4498f9a1aeb9bc026cab662bf40c71a48346c12",
        "sha256:5bfef8256a519467ba5d0722cb4992a08606f65822c303a61d24a8e699574014",
    ]
    for idx, digest in enumerate(layers):
        tar_path = os.path.join(layers_dir, f"layer_{idx}.tar.gz")
        if not os.path.exists(tar_path) or os.path.getsize(tar_path) < 50:
            print(f"Downloading DFC layer {idx}/6...")
            blob_url = f"https://registry-1.docker.io/v2/margisbench/margisbench_hailo_compiler/blobs/{digest}"
            subprocess.run(["curl", "-sSL", "-H", f"Authorization: Bearer {token}", "-o", tar_path, blob_url], check=True)
        subprocess.run(["tar", "-xzf", tar_path, "-C", rootfs_dir, "--exclude=dev/*"], check=False)

# 2. Write inner DFC compilation script
dfc_code = '''
import numpy as np
from hailo_sdk_client import ClientRunner

onnx_path = "/work/model.onnx"
hef_out = "/work/uk_birds_regnet_y_1_6g_h8.hef"

runner = ClientRunner(hw_arch="hailo8")
hn, npz = runner.translate_onnx_model(
    onnx_path,
    "uk_birds_regnet_y_1_6g",
    start_node_names=["input"],
    end_node_names=["output"],
    net_input_shapes={"input": [1, 3, 256, 256]},
)

alls = (
    "normalization1 = normalization([133.824, 136.986, 129.693], [54.4425, 53.6265, 66.861])\\n"
    "model_optimization_flavor(optimization_level=0, compression_level=0)\\n"
)
runner.load_model_script(alls)

rng = np.random.default_rng(42)
calib_data = rng.integers(0, 256, size=(64, 256, 256, 3), dtype=np.uint8).astype(np.float32)
runner.optimize(calib_data)

hef = runner.compile()
with open(hef_out, "wb") as f:
    f.write(hef)
print("COMPILED_HEF_OK:", len(hef))
'''
with open(os.path.join(work_dir, "run_dfc.py"), "w") as f:
    f.write(dfc_code)

cmd = [
    "bwrap",
    "--bind", rootfs_dir, "/",
    "--dev", "/dev",
    "--proc", "/proc",
    "--bind", "/tmp", "/tmp",
    "--bind", work_dir, "/work",
    "--ro-bind", "/etc/resolv.conf", "/etc/resolv.conf",
    "/usr/local/bin/python3", "/work/run_dfc.py"
]
subprocess.run(cmd, check=True)
"""
    res = subprocess.run(
        [
            "ssh",
            target,
            f"systemd-inhibit --what=sleep:idle --who='BirdSpotter' --why='Hailo DFC Compilation' python3 -u -c '{remote_script}' || python3 -u -c '{remote_script}'",
        ]
    )
    if res.returncode != 0:
        return res.returncode

    subprocess.run(
        ["scp", f"{target}:~/hailo_build/work/uk_birds_regnet_y_1_6g_h8.hef", str(hef_path)],
        check=True,
    )
    print(f"[Hailo DFC] Downloaded compiled Hailo-8 HEF to {hef_path}!")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export UK Bird Classifier to ONNX and compile Hailo-8 .hef"
    )
    parser.add_argument(
        "--compile-hef",
        action="store_true",
        help="Compile .hef on the remote x86_64 machine via SSH",
    )
    parser.add_argument("--host", default="192.168.100.101", help="x86_64 SSH host")
    parser.add_argument("--user", default="sean", help="x86_64 SSH username")
    args = parser.parse_args()

    if not PT_PATH.is_file():
        print(f"Downloading {PT_PATH.name} from Hugging Face...")
        import urllib.request

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(
            "https://huggingface.co/birder-project/regnet_y_1_6g_eu-common/resolve/main/regnet_y_1_6g_eu-common.pt",
            PT_PATH,
        )

    export_regnet_to_onnx(PT_PATH, ONNX_PATH)
    verify_onnx_with_opencv(ONNX_PATH)

    if args.compile_hef:
        return compile_hef_on_remote_x86(args.host, args.user, ONNX_PATH, HEF_PATH)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
