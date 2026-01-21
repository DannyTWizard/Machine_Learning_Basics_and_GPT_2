# XOR Cipher Decryption Report

## Challenge Overview

An intercepted message was provided that was encrypted using a "repeating operation" with a "predictable format."

**Ciphertext (hex):** `15091702150b0a043a07003a01000f183a04001b00001218`

## Analysis

### Identifying the Cipher

Key observations:
1. The ciphertext is hex-encoded (48 hex characters = 24 bytes)
2. "Repeating operation" suggests a cipher that cycles through a key
3. "Predictable format" hints at a known plaintext (likely `flag{`)

These clues point to **XOR encryption with a repeating key**.

### XOR Cipher Properties

XOR (exclusive or) has a crucial property that makes it vulnerable to known-plaintext attacks:

```
plaintext XOR key = ciphertext
plaintext XOR ciphertext = key
ciphertext XOR key = plaintext
```

If we know any portion of the plaintext, we can recover the corresponding portion of the key.

### Known Plaintext Attack

Since CTF flags follow the format `flag{...}`, we know the first 5 bytes of plaintext.

**Step 1: Convert hex ciphertext to bytes**
```
Ciphertext bytes: 15 09 17 02 15 0b 0a 04 3a 07 00 3a 01 00 0f 18 3a 04 00 1b 00 00 12 18
```

**Step 2: XOR known plaintext with ciphertext to get key**
```
Known plaintext: f    l    a    g    {
Hex values:      66   6c   61   67   7b
Ciphertext:      15   09   17   02   15
XOR result:      73   65   76   65   6e  =  "seven"
```

**Step 3: Apply the derived key to decrypt**

Using the key `seven` (which repeats), XOR each byte:

| Position | Ciphertext | Key Byte | XOR Result | Character |
|----------|------------|----------|------------|-----------|
| 0        | 0x15       | s (0x73) | 0x66       | f         |
| 1        | 0x09       | e (0x65) | 0x6c       | l         |
| 2        | 0x17       | v (0x76) | 0x61       | a         |
| 3        | 0x02       | e (0x65) | 0x67       | g         |
| 4        | 0x15       | n (0x6e) | 0x7b       | {         |
| 5        | 0x0b       | s (0x73) | 0x78       | x         |
| 6        | 0x0a       | e (0x65) | 0x6f       | o         |
| 7        | 0x04       | v (0x76) | 0x72       | r         |
| 8        | 0x3a       | e (0x65) | 0x5f       | _         |
| 9        | 0x07       | n (0x6e) | 0x69       | i         |
| 10       | 0x00       | s (0x73) | 0x73       | s         |
| 11       | 0x3a       | e (0x65) | 0x5f       | _         |
| 12       | 0x01       | v (0x76) | 0x77       | w         |
| 13       | 0x00       | e (0x65) | 0x65       | e         |
| 14       | 0x0f       | n (0x6e) | 0x61       | a         |
| 15       | 0x18       | s (0x73) | 0x6b       | k         |
| 16       | 0x3a       | e (0x65) | 0x5f       | _         |
| 17       | 0x04       | v (0x76) | 0x72       | r         |
| 18       | 0x00       | e (0x65) | 0x65       | e         |
| 19       | 0x1b       | n (0x6e) | 0x75       | u         |
| 20       | 0x00       | s (0x73) | 0x73       | s         |
| 21       | 0x00       | e (0x65) | 0x65       | e         |
| 22       | 0x12       | v (0x76) | 0x64       | d         |
| 23       | 0x18       | e (0x65) | 0x7d       | }         |

## Solution

| Property | Value |
|----------|-------|
| **Cipher Type** | XOR with repeating key |
| **Encryption Key** | `seven` |
| **Original Message** | `flag{xor_is_weak_reused}` |
| **Flag** | `flag{xor_is_weak_reused}` |

## Why XOR is Weak

The flag itself reveals the vulnerability: "xor_is_weak_reused"

XOR encryption is insecure when:
1. **Key is reused** - Same key encrypts multiple messages
2. **Key is shorter than message** - Key must repeat, creating patterns
3. **Plaintext is predictable** - Known formats (like `flag{`) enable key recovery

A single-use key as long as the message (one-time pad) would be unbreakable, but practical implementations rarely achieve this.

## Files

- `xor_decrypt.py` - Python script implementing the decryption
