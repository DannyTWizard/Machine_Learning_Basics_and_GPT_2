#!/usr/bin/env python3
"""
Double Wrapped CTF Challenge Solver
Challenge: The data has been transformed more than once. Undo the layers to recover the flag.
"""

import base64

def solve():
    # Read the encoded data
    with open('../utils/chef.txt.txt', 'r') as f:
        encoded_data = f.read().strip()

    print(f"[*] Original encoded data:")
    print(f"    {encoded_data}")
    print()

    # Layer 1: Base64 decode
    layer1_decoded = base64.b64decode(encoded_data).decode('utf-8')
    print(f"[*] After Base64 decode (Layer 1):")
    print(f"    {layer1_decoded}")
    print()

    # Layer 2: Hex to ASCII
    # The hex values are space-separated
    hex_values = layer1_decoded.split()
    flag = ''.join(chr(int(h, 16)) for h in hex_values)
    print(f"[*] After Hex decode (Layer 2):")
    print(f"    {flag}")
    print()

    return flag

if __name__ == "__main__":
    print("=" * 50)
    print("Double Wrapped CTF Challenge Solver")
    print("=" * 50)
    print()

    flag = solve()

    print("=" * 50)
    print(f"FLAG: {flag}")
    print("=" * 50)
