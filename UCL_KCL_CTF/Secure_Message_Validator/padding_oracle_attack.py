#!/usr/bin/env python3
"""
Padding Oracle Attack - CTF Challenge Solution
Decrypts AES-CBC encrypted message using padding oracle vulnerability.
Optimized with concurrent requests for faster execution.
"""

import requests
import sys
import concurrent.futures
from typing import List, Tuple, Optional

# Configuration
BASE_URL = "http://13.53.139.173/crypto3/check"
BLOCK_SIZE = 16  # AES block size in bytes
MAX_WORKERS = 32  # Number of concurrent threads

# Given values
IV_HEX = "18aeb3d267f3f9a9803dc9f16bcac18c"
CIPHERTEXT_HEX = "9130f576cd1318c91ffe610fb44e6feeefb2e6cb2fd3d3c52d43cd0496464cc7ee0454534c76b6c6d7f27ffb4d0a3052d72aea6ab20af96ff82016e4aa4237da"

# Global session for connection pooling
session = requests.Session()


def check_padding(data_hex: str) -> bool:
    """
    Send ciphertext to oracle and check if padding is valid.
    Returns True if padding is valid, False otherwise.
    """
    try:
        response = session.post(BASE_URL, data={"data": data_hex}, timeout=10)
        if "class='ok'" in response.text or 'class="ok"' in response.text:
            return True
        if "Valid padding" in response.text and "Invalid padding" not in response.text:
            return True
        return False
    except requests.RequestException as e:
        return False


def hex_to_bytes(hex_str: str) -> bytes:
    return bytes.fromhex(hex_str)


def bytes_to_hex(data: bytes) -> str:
    return data.hex()


def split_into_blocks(data: bytes) -> List[bytes]:
    """Split data into BLOCK_SIZE byte blocks."""
    return [data[i:i+BLOCK_SIZE] for i in range(0, len(data), BLOCK_SIZE)]


def xor_bytes(a: bytes, b: bytes) -> bytes:
    """XOR two byte sequences."""
    return bytes(x ^ y for x, y in zip(a, b))


def check_guess(args: Tuple[int, bytearray, bytes, int]) -> Tuple[int, bool]:
    """Check a single guess for parallel execution."""
    guess, mal_prev, target_block, byte_pos = args
    mal_prev[byte_pos] = guess
    mal_ciphertext = bytes(mal_prev) + target_block
    data_hex = bytes_to_hex(mal_ciphertext)
    return (guess, check_padding(data_hex))


def find_valid_byte(byte_pos: int, padding_value: int, intermediate: bytearray,
                    target_block: bytes) -> Optional[int]:
    """
    Find the correct byte value using parallel requests.
    Returns the guess value that produces valid padding.
    """
    # Prepare all guesses
    tasks = []
    for guess in range(256):
        mal_prev = bytearray(BLOCK_SIZE)
        # Set already-known bytes to produce correct padding
        for i in range(byte_pos + 1, BLOCK_SIZE):
            mal_prev[i] = intermediate[i] ^ padding_value
        mal_prev[byte_pos] = guess
        tasks.append((guess, mal_prev.copy(), target_block, byte_pos))

    # Execute in parallel
    valid_guesses = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        futures = {executor.submit(check_guess_simple, task): task[0] for task in tasks}

        for future in concurrent.futures.as_completed(futures):
            guess = futures[future]
            try:
                result = future.result()
                if result:
                    valid_guesses.append(guess)
            except Exception as e:
                pass

    # If we found valid guesses, verify which one is correct
    if len(valid_guesses) == 1:
        return valid_guesses[0]
    elif len(valid_guesses) > 1:
        # Multiple valid guesses (edge case at last byte)
        # Verify by modifying another byte
        for guess in valid_guesses:
            mal_prev = bytearray(BLOCK_SIZE)
            for i in range(byte_pos + 1, BLOCK_SIZE):
                mal_prev[i] = intermediate[i] ^ padding_value
            mal_prev[byte_pos] = guess
            if byte_pos > 0:
                mal_prev[byte_pos - 1] ^= 0xFF
            mal_ciphertext = bytes(mal_prev) + target_block
            if not check_padding(bytes_to_hex(mal_ciphertext)):
                continue
            return guess
        return valid_guesses[0]
    return None


def check_guess_simple(task: Tuple[int, bytearray, bytes, int]) -> bool:
    """Simple check for a guess."""
    guess, mal_prev, target_block, byte_pos = task
    mal_ciphertext = bytes(mal_prev) + target_block
    data_hex = bytes_to_hex(mal_ciphertext)
    return check_padding(data_hex)


def decrypt_block(prev_block: bytes, target_block: bytes, block_num: int, total_blocks: int) -> bytes:
    """
    Decrypt a single block using the padding oracle attack.
    """
    intermediate = bytearray(BLOCK_SIZE)

    print(f"\n[*] Decrypting block {block_num}/{total_blocks}")

    # Work backwards from the last byte
    for byte_pos in range(BLOCK_SIZE - 1, -1, -1):
        padding_value = BLOCK_SIZE - byte_pos

        guess = find_valid_byte(byte_pos, padding_value, intermediate, target_block)

        if guess is not None:
            intermediate[byte_pos] = guess ^ padding_value
            plaintext_byte = intermediate[byte_pos] ^ prev_block[byte_pos]
            progress = BLOCK_SIZE - byte_pos
            char_repr = chr(plaintext_byte) if 32 <= plaintext_byte < 127 else '?'
            print(f"    Byte {byte_pos:2d}: 0x{plaintext_byte:02x} ('{char_repr}') [{progress}/{BLOCK_SIZE}]")
        else:
            print(f"[!] Failed to find valid padding for byte {byte_pos}")
            intermediate[byte_pos] = 0

    plaintext = xor_bytes(bytes(intermediate), prev_block)
    return plaintext


def padding_oracle_attack(iv: bytes, ciphertext: bytes) -> bytes:
    """
    Perform the complete padding oracle attack.
    """
    blocks = split_into_blocks(ciphertext)
    total_blocks = len(blocks)

    print(f"[*] IV: {bytes_to_hex(iv)}")
    print(f"[*] Ciphertext length: {len(ciphertext)} bytes ({total_blocks} blocks)")

    plaintext = b""
    prev_block = iv

    for i, block in enumerate(blocks):
        decrypted = decrypt_block(prev_block, block, i + 1, total_blocks)
        plaintext += decrypted
        prev_block = block
        print(f"[+] Block {i+1} plaintext: {decrypted}")

    # Remove PKCS#7 padding
    padding_len = plaintext[-1]
    if padding_len <= BLOCK_SIZE and all(b == padding_len for b in plaintext[-padding_len:]):
        plaintext = plaintext[:-padding_len]
        print(f"\n[+] Removed {padding_len} bytes of PKCS#7 padding")

    return plaintext


def test_oracle():
    """Test the oracle with known data."""
    print("[*] Testing oracle...")
    original_data = IV_HEX + CIPHERTEXT_HEX
    result = check_padding(original_data)
    print(f"[*] Original ciphertext: {'Valid' if result else 'Invalid'}")

    corrupted = "00" * 32
    result = check_padding(corrupted)
    print(f"[*] Corrupted ciphertext: {'Valid' if result else 'Invalid'}")


def main():
    print("=" * 60)
    print("Padding Oracle Attack - AES-CBC Decryption")
    print("=" * 60)

    test_oracle()

    iv = hex_to_bytes(IV_HEX)
    ciphertext = hex_to_bytes(CIPHERTEXT_HEX)

    plaintext = padding_oracle_attack(iv, ciphertext)

    print("\n" + "=" * 60)
    print("DECRYPTION COMPLETE")
    print("=" * 60)
    print(f"Plaintext (hex): {bytes_to_hex(plaintext)}")
    print(f"Plaintext (raw): {plaintext}")
    try:
        decoded = plaintext.decode('utf-8')
        print(f"Plaintext (UTF-8): {decoded}")
    except UnicodeDecodeError:
        print("[!] Could not decode as UTF-8")

    return plaintext


if __name__ == "__main__":
    main()
