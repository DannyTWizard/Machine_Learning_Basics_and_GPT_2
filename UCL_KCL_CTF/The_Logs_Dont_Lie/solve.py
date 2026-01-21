#!/usr/bin/env python3
"""
Solution for "The Logs Don't Lie" CTF challenge.

The challenge involves analyzing sysmon.json logs to find an XOR key,
then using it to decrypt cache.bin to reveal the flag.
"""

import json
import os

def main():
    # Paths to the input files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    utils_dir = os.path.join(script_dir, '..', 'utils')

    cache_path = os.path.join(utils_dir, 'cache.bin')
    sysmon_path = os.path.join(utils_dir, 'sysmon.json')

    # Step 1: Read and analyze sysmon.json to find the XOR key
    print("[*] Analyzing sysmon.json for clues...")

    with open(sysmon_path, 'r') as f:
        data = json.load(f)

    # Look for command lines that might contain a key
    key = None
    for event in data:
        cmd = event.get('CommandLine', '')
        # Look for SYNC_TOKEN being set
        if 'SYNC_TOKEN' in cmd and 'setx' in cmd:
            # Extract the value from: cmd.exe /c setx SYNC_TOKEN "bluewhale"
            start = cmd.find('"') + 1
            end = cmd.rfind('"')
            if start > 0 and end > start:
                key = cmd[start:end]
                print(f"[+] Found SYNC_TOKEN in command line: {key}")
                break

    if not key:
        print("[-] Could not find key in sysmon logs")
        return

    # Step 2: Read cache.bin and decrypt using XOR
    print(f"\n[*] Reading cache.bin...")

    with open(cache_path, 'rb') as f:
        cache_bytes = f.read()

    print(f"[+] Cache size: {len(cache_bytes)} bytes")
    print(f"[+] Cache hex: {cache_bytes.hex()}")

    # Step 3: XOR decrypt
    print(f"\n[*] Decrypting with key: '{key}'")

    decrypted = []
    for i, byte in enumerate(cache_bytes):
        key_byte = ord(key[i % len(key)])
        decrypted.append(chr(byte ^ key_byte))

    message = ''.join(decrypted)

    print(f"\n[+] Decrypted message: {message}")
    print(f"\n{'='*50}")
    print(f"FLAG: {message}")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()
