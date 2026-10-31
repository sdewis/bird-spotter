#!/usr/bin/env python3
"""Scan the local /24 subnet for the Raspberry Pi 2 SSH server."""

import concurrent.futures
import socket
import subprocess


def get_local_subnet() -> tuple[str, str]:
    out = subprocess.check_output(["ip", "-4", "-o", "addr", "show", "wlan0"], text=True)
    cidr = out.split()[3]  # e.g. 192.168.100.122/24
    ip = cidr.split("/")[0]
    prefix = ".".join(ip.split(".")[:3])
    return ip, prefix


def probe_ssh(ip: str) -> tuple[str, str] | None:
    try:
        with socket.create_connection((ip, 22), timeout=0.6) as s:
            banner = s.recv(128).decode("utf-8", errors="ignore").strip()
            return ip, banner
    except OSError:
        return None


def main() -> None:
    local_ip, prefix = get_local_subnet()
    print(f"Scanning {prefix}.0/24 (excluding {local_ip}) for SSH on port 22...")
    candidates = [f"{prefix}.{i}" for i in range(1, 255) if f"{prefix}.{i}" != local_ip]

    found = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=48) as ex:
        for res in ex.map(probe_ssh, candidates):
            if res:
                found.append(res)
                print(f"  [FOUND] {res[0]} -> {res[1]}")

    if not found:
        print("  No SSH hosts responded. Ensure the Raspberry Pi 2 is powered on and connected to LAN.")


if __name__ == "__main__":
    main()
