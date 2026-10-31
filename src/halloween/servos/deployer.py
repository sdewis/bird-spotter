"""Remote Raspberry Pi Servo Node Provisioner & Deployer.

Packages the Picon Zero node software (`pi2_daemon.py`, `piconzero/`, `piconz.sh`,
and `halloween-pi2.service`) and deploys it to any target Raspberry Pi (e.g., RPi 4)
given its hostname/IP and SSH credentials.
"""

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import pty
import re
import select
import shutil
import tarfile
import time

PROJECT_ROOT = Path(__file__).resolve().parents[3]
NODE_BUNDLE_DIR = Path(__file__).resolve().parent / "node_bundle"
DAEMON_SRC = Path(__file__).resolve().parent / "pi2_daemon.py"
ARCHIVE_PATH = PROJECT_ROOT / "dist" / "pi_servo_node_bundle.tar.gz"
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.toml"


@dataclass
class DeployResult:
    """Result summary from deploying the servo node bundle to a Raspberry Pi."""

    success: bool
    host: str
    user: str
    model: str = ""
    i2c_detected: bool = False
    daemon_listening: bool = False
    log: str = ""


def build_node_bundle_archive() -> Path:
    """Package `pi2_daemon.py`, `piconzero/`, `piconz.sh`, and service unit into a .tar.gz bundle."""
    NODE_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Always keep node_bundle/pi2_daemon.py synced with latest src/halloween/servos/pi2_daemon.py
    if DAEMON_SRC.is_file():
        shutil.copy2(DAEMON_SRC, NODE_BUNDLE_DIR / "pi2_daemon.py")

    # Also provide a clean Python 3 piconzero module inside node_bundle/piconzero/piconzero_py3.py
    picon_dir = NODE_BUNDLE_DIR / "piconzero"
    picon_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(ARCHIVE_PATH, "w:gz") as tar:
        for item in sorted(NODE_BUNDLE_DIR.iterdir()):
            tar.add(item, arcname=item.name)

    return ARCHIVE_PATH


def run_ssh_interactive(cmd: list[str], password: str, timeout: float = 30.0) -> tuple[int, str]:
    """Execute an ssh or scp command in a pseudo-terminal, supplying password when prompted."""
    pid, fd = pty.fork()
    if pid == 0:
        os.execvp(cmd[0], cmd)

    output = b""
    prompts_answered = 0
    end = time.time() + timeout

    while time.time() < end:
        r, _, _ = select.select([fd], [], [], 0.2)
        if r:
            try:
                data = os.read(fd, 4096)
                if not data:
                    break
                output += data
                # Check recent tail for password prompt (SSH or sudo)
                tail = output[-240:].lower()
                if (b"password:" in tail or b"[sudo]" in tail) and prompts_answered < 3:
                    os.write(fd, (password + "\n").encode("utf-8"))
                    prompts_answered += 1
                    # Clear prompt marker from buffer tail by appending newline marker
                    output += b"\n"
            except OSError:
                break

    _, status = os.waitpid(pid, 0)
    exit_code = os.waitstatus_to_exitcode(status) if hasattr(os, "waitstatus_to_exitcode") else status
    return exit_code, output.decode("utf-8", errors="replace")


def render_systemd_unit(user: str, daemon_port: int = 5555) -> str:
    home_dir = "/root" if user == "root" else f"/home/{user}"
    return f"""[Unit]
Description=Halloween Picon Zero Servo & Audio Daemon (Starts in Safe STANDBY)
After=network-online.target sound.target
Wants=network-online.target

[Service]
Type=simple
User={user}
WorkingDirectory={home_dir}
ExecStart=/usr/bin/python3 -u {home_dir}/pi2_daemon.py --port {daemon_port}
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
"""


def update_settings_host(new_host: str, new_user: str) -> None:
    """Update [pi2_servos] host and ssh_user in config/settings.toml."""
    if not SETTINGS_PATH.is_file():
        return
    text = SETTINGS_PATH.read_text(encoding="utf-8")
    text = re.sub(r'(?m)^host\s*=\s*"[^"]*"', f'host = "{new_host}"', text, count=1)
    text = re.sub(r'(?m)^ssh_user\s*=\s*"[^"]*"', f'ssh_user = "{new_user}"', text, count=1)
    SETTINGS_PATH.write_text(text, encoding="utf-8")


def deploy_to_pi(
    host: str,
    user: str = "sean",
    password: str = " ",
    ssh_port: int = 22,
    daemon_port: int = 5555,
    update_config: bool = True,
) -> DeployResult:
    """Package and deploy the Picon Zero servo & audio node software to a target Raspberry Pi."""
    logs: list[str] = []
    archive = build_node_bundle_archive()
    logs.append(f"[1/4] Packaged node bundle: {archive} ({archive.stat().st_size} bytes)")

    target = f"{user}@{host}"
    ssh_opts = [
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=6",
    ]

    # Upload bundle archive and rendered systemd unit
    unit_tmp = Path("/tmp/halloween-pi2.service")
    unit_tmp.write_text(render_systemd_unit(user=user, daemon_port=daemon_port), encoding="utf-8")

    logs.append(f"[2/4] Uploading bundle to {target}:~/pi_servo_node_bundle.tar.gz ...")
    code, scp_out = run_ssh_interactive(
        [
            "scp",
            "-P", str(ssh_port),
            *ssh_opts,
            str(archive),
            str(unit_tmp),
            f"{target}:~/",
        ],
        password=password,
        timeout=25.0,
    )
    logs.append(scp_out.strip())
    if code != 0:
        return DeployResult(
            success=False,
            host=host,
            user=user,
            log="\n".join(logs),
        )

    # Extract bundle, install bundled .so libraries, enable I2C & audio kernel modules, and start service
    logs.append(f"[3/4] Provisioning hardware kernel modules, .so libraries, and systemd service on {target} ...")
    remote_script = (
        "echo '=== MODEL ==='; tr -d '\\0' < /proc/device-tree/model 2>/dev/null || uname -a; echo ''; "
        "tar -xzf ~/pi_servo_node_bundle.tar.gz -C ~/ && "
        "ARCH=$(uname -m); "
        "if [ \"$ARCH\" = \"aarch64\" ] && [ -d ~/libs/aarch64 ]; then "
        "  sudo cp -n ~/libs/aarch64/libi2c.so.0 /usr/lib/aarch64-linux-gnu/ 2>/dev/null || true; "
        "  sudo cp -n ~/libs/aarch64/smbus*.so /usr/lib/python3/dist-packages/ 2>/dev/null || true; "
        "elif [ -d ~/libs/armhf ]; then "
        "  sudo cp -n ~/libs/armhf/libi2c.so.0 /usr/lib/arm-linux-gnueabihf/ 2>/dev/null || true; "
        "  sudo cp -n ~/libs/armhf/smbus*.so /usr/lib/python3/dist-packages/ 2>/dev/null || true; "
        "fi; "
        "sudo ldconfig 2>/dev/null || true; "
        "sudo -S raspi-config nonint do_i2c 0 2>/dev/null || true; "
        "sudo dtparam i2c_arm=on 2>/dev/null || true; "
        "sudo dtparam audio=on 2>/dev/null || true; "
        "sudo modprobe i2c_bcm2835 2>/dev/null || true; "
        "sudo modprobe i2c-dev 2>/dev/null || true; "
        "sudo modprobe snd_bcm2835 2>/dev/null || true; "
        "(dpkg -s python3-smbus i2c-tools alsa-utils >/dev/null 2>&1 || "
        "(sudo apt-get update -qq && sudo apt-get install -y -qq python3-smbus i2c-tools alsa-utils) || true) && "
        f"fuser -k {daemon_port}/tcp 2>/dev/null || true; "
        "sudo mv ~/halloween-pi2.service /etc/systemd/system/halloween-pi2.service && "
        "sudo systemctl daemon-reload && "
        "sudo systemctl enable halloween-pi2.service && "
        "sudo systemctl restart halloween-pi2.service && "
        "sleep 1.0 && "
        "echo '=== I2C BUS 1 ==='; /usr/sbin/i2cdetect -y 1 2>/dev/null || ls -l /dev/i2c* 2>/dev/null || true; "
        "echo '=== SERVICE STATUS ==='; sudo systemctl is-active halloween-pi2.service || true"
    )
    code, ssh_out = run_ssh_interactive(
        ["ssh", "-p", str(ssh_port), *ssh_opts, target, remote_script],
        password=password,
        timeout=60.0,
    )
    logs.append(ssh_out.strip())

    model_str = ""
    if "=== MODEL ===" in ssh_out:
        after = ssh_out.split("=== MODEL ===", 1)[1].strip().splitlines()
        if after:
            model_str = after[0].strip()

    i2c_ok = " 22 " in ssh_out
    svc_ok = "active" in ssh_out.splitlines()[-1] if ssh_out.strip() else False

    if update_config and code == 0:
        update_settings_host(new_host=host, new_user=user)
        logs.append(f"[4/4] Updated config/settings.toml -> host = '{host}', ssh_user = '{user}'")

    return DeployResult(
        success=(code == 0 and svc_ok),
        host=host,
        user=user,
        model=model_str,
        i2c_detected=i2c_ok,
        daemon_listening=svc_ok,
        log="\n".join(logs),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Package or deploy the Picon Zero Servo & Audio Node to a Raspberry Pi."
    )
    parser.add_argument(
        "--package-only",
        action="store_true",
        help="Only build dist/pi_servo_node_bundle.tar.gz without deploying",
    )
    parser.add_argument("--host", help="Target Raspberry Pi hostname or IP address")
    parser.add_argument("--user", default="sean", help="SSH username (default: sean)")
    parser.add_argument("--password", default=" ", help="SSH/sudo password (default: space)")
    parser.add_argument("--ssh-port", type=int, default=22, help="SSH port (default: 22)")
    parser.add_argument("--daemon-port", type=int, default=5555, help="Daemon TCP port (default: 5555)")
    parser.add_argument(
        "--no-config-update",
        action="store_true",
        help="Do not overwrite config/settings.toml with the new host",
    )
    args = parser.parse_args()

    if args.package_only or not args.host:
        archive = build_node_bundle_archive()
        print(f"Packaged servo node bundle: {archive} ({archive.stat().st_size} bytes)")
        if not args.host and not args.package_only:
            print("Tip: pass --host <IP_OR_HOSTNAME> --user <USER> --password <PASS> to deploy to a new Pi.")
        return 0

    res = deploy_to_pi(
        host=args.host,
        user=args.user,
        password=args.password,
        ssh_port=args.ssh_port,
        daemon_port=args.daemon_port,
        update_config=not args.no_config_update,
    )
    print(res.log)
    print("\n=== Deployment Summary ===")
    print(f"  Target: {res.user}@{res.host} ({res.model or 'Unknown Model'})")
    print(f"  Daemon Active: {'YES' if res.daemon_listening else 'NO'}")
    print(f"  Picon Zero HAT (0x22) Detected: {'YES' if res.i2c_detected else 'NOT DETECTED YET'}")
    return 0 if res.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
