#!/usr/bin/env python3
"""
XOR Cipher Decryption with Known Plaintext Attack

This script decrypts a message encrypted with a repeating XOR key.
We exploit the known flag format "flag{" to recover the key.
"""

def hex_to_bytes(hex_string):
    """Convert hex string to bytes."""
    return bytes.fromhex(hex_string)

def xor_bytes(data, key):
    """XOR data with a repeating key."""
    return bytes([data[i] ^ key[i % len(key)] for i in range(len(data))])

def derive_key(ciphertext, known_plaintext):
    """Derive key by XORing ciphertext with known plaintext."""
    return bytes([ciphertext[i] ^ known_plaintext[i] for i in range(len(known_plaintext))])

def main():
    # The intercepted ciphertext (hex encoded)
    ciphertext_hex = "15091702150b0a043a07003a01000f183a04001b00001218"

    print("=" * 60)
    print("XOR Cipher Decryption - Known Plaintext Attack")
    print("=" * 60)

    # Convert hex to bytes
    ciphertext = hex_to_bytes(ciphertext_hex)
    print(f"\nCiphertext (hex): {ciphertext_hex}")
    print(f"Ciphertext length: {len(ciphertext)} bytes")

    # Known plaintext - CTF flags typically start with "flag{"
    known_plaintext = b"flag{"
    print(f"\nKnown plaintext: {known_plaintext.decode()}")
    print(f"Known plaintext (hex): {known_plaintext.hex()}")

    # Derive the key using known plaintext attack
    # XOR property: plaintext XOR key = ciphertext
    #              plaintext XOR ciphertext = key
    partial_key = derive_key(ciphertext, known_plaintext)
    print(f"\nDerived key fragment: {partial_key.decode()}")
    print(f"Key fragment (hex): {partial_key.hex()}")

    # The key appears to be "seven" - let's verify
    key = partial_key
    print(f"\nAssuming full key: '{key.decode()}'")

    # Decrypt the full message
    plaintext = xor_bytes(ciphertext, key)
    print(f"\n{'=' * 60}")
    print("DECRYPTION RESULT")
    print("=" * 60)
    print(f"Decrypted message: {plaintext.decode()}")
    print(f"Key used: {key.decode()}")
    print("=" * 60)

    # Verification: show the XOR operation step by step
    print("\nVerification (byte-by-byte XOR):")
    print("-" * 60)
    print(f"{'Index':<6}{'Cipher':<8}{'Key':<8}{'Plain':<8}{'Char':<6}")
    print("-" * 60)
    for i in range(len(ciphertext)):
        c = ciphertext[i]
        k = key[i % len(key)]
        p = c ^ k
        char = chr(p) if 32 <= p < 127 else '?'
        print(f"{i:<6}{hex(c):<8}{hex(k):<8}{hex(p):<8}{char:<6}")

if __name__ == "__main__":
    main()
