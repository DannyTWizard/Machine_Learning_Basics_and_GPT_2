#!/usr/bin/env python3
"""
SystemSync Investigation CTF Challenge Solver

Analyzes artifacts from:
- registry.txt
- sysmon_final.json
- tasks.xml

To reconstruct the validation token and obtain the flag.
"""

import codecs
import re
import json
import os

def rot13(text):
    """Apply ROT13 cipher to text"""
    return codecs.decode(text, 'rot_13')

def parse_registry(filepath):
    """Extract SyncLabel from registry dump"""
    sync_label = None
    with open(filepath, 'r') as f:
        content = f.read()

    # Find SyncLabel REG_BINARY value
    match = re.search(r'SyncLabel\s+REG_BINARY\s+([\da-fA-F\s]+)', content)
    if match:
        hex_values = match.group(1).strip().split()
        sync_label = ''.join(chr(int(h, 16)) for h in hex_values)

    return sync_label

def parse_tasks_xml(filepath):
    """Extract modeTag from tasks.xml"""
    with open(filepath, 'rb') as f:
        content = f.read().decode('utf-16', errors='ignore')

    # Find modeTag variable
    match = re.search(r"\$modeTag\s*=\s*['\"]([^'\"]+)['\"]", content)
    if match:
        return match.group(1)
    return None

def parse_sysmon(filepath):
    """Extract syncId from sysmon logs"""
    with open(filepath, 'r') as f:
        content = f.read()

    # Find syncId in PowerShell command
    match = re.search(r'\$syncId\s*=\s*(\d+)', content)
    if match:
        return int(match.group(1))
    return None

print("="*60)
print("SystemSync Investigation - CTF Challenge Solver")
print("="*60)

# Path to artifacts
utils_path = "/mnt/c/users/armando/documents/Machine_Learning_Basics_and_GPT_2/UCL_KCL_CTF/utils"

# Parse all artifacts
print("\n[*] Parsing artifacts...")

sync_label = parse_registry(os.path.join(utils_path, "registry.txt"))
print(f"    SyncLabel (from registry): {sync_label}")

mode_tag = parse_tasks_xml(os.path.join(utils_path, "tasks.xml"))
print(f"    modeTag (from tasks.xml): {mode_tag}")

sync_id = parse_sysmon(os.path.join(utils_path, "sysmon_final.json"))
print(f"    syncId (from sysmon): {sync_id}")

# Decode using ROT13
print("\n[*] Applying ROT13 decoding...")
sync_label_decoded = rot13(sync_label)
mode_tag_decoded = rot13(mode_tag)

print(f"    SyncLabel decoded: '{sync_label}' -> '{sync_label_decoded}'")
print(f"    modeTag decoded: '{mode_tag}' -> '{mode_tag_decoded}'")
print(f"    syncId: {sync_id} (not encoded)")

print("\n" + "="*60)
print("SOLUTION")
print("="*60)

print(f"\nOriginal Message (Validation Token):")
print(f"  {sync_label_decoded} {mode_tag_decoded} {sync_id}")

print(f"\nFlag:")
print(f"  flag{{{sync_label_decoded}_{mode_tag_decoded}_{sync_id}}}")

print("\n" + "="*60)
print("Alternative flag formats (if the above doesn't work):")
print("="*60)
alternatives = [
    f"ctf{{{sync_label_decoded}_{mode_tag_decoded}_{sync_id}}}",
    f"FLAG{{{sync_label_decoded}_{mode_tag_decoded}_{sync_id}}}",
    f"CTF{{{sync_label_decoded}_{mode_tag_decoded}_{sync_id}}}",
    f"flag{{{mode_tag_decoded}_{sync_label_decoded}_{sync_id}}}",
]
for alt in alternatives:
    print(f"  {alt}")
