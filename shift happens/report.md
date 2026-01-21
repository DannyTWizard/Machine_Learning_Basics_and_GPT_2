# Cipher Analysis Report

## Challenge Overview

An intercepted message was provided that appeared to be encoded with a classical cipher. The message was not readable but retained a recognizable structure suggesting a CTF flag format.

**Ciphertext:** `uapv{yjhi_pc_xcigd_rxewtg}`

## Analysis

### Initial Observations

1. The message structure `uapv{...}` resembles the standard CTF flag format `flag{...}`
2. The description indicated "nothing modern" was used for encryption
3. Special characters (braces and underscores) were preserved, suggesting a simple alphabetic substitution

### Cipher Identification

The preserved structure and hint about classical encryption pointed to a **Caesar cipher** (also known as a ROT cipher). This is one of the oldest known encryption methods, dating back to Julius Caesar.

In a Caesar cipher:
- Each letter is shifted by a fixed number of positions in the alphabet
- Non-alphabetic characters remain unchanged
- The shift amount is the "key"

### Determining the Shift

By comparing the ciphertext prefix `uapv` to the expected plaintext `flag`:

| Ciphertext | Plaintext | Shift Calculation |
|------------|-----------|-------------------|
| u (20)     | f (5)     | (5 - 20) mod 26 = 11 |
| a (0)      | l (11)    | (11 - 0) mod 26 = 11 |
| p (15)     | a (0)     | (0 - 15) mod 26 = 11 |
| v (21)     | g (6)     | (6 - 21) mod 26 = 11 |

The consistent shift of **11** confirms this is a Caesar cipher.

### Decryption Method

The original message was encrypted using **ROT15** (shifting each letter forward by 15 positions). To decrypt, we apply **ROT11** (shifting forward by 11, which is equivalent to shifting backward by 15).

```python
def caesar_decrypt(ciphertext, shift):
    result = ''
    for char in ciphertext:
        if char.isalpha():
            base = ord('a') if char.islower() else ord('A')
            result += chr((ord(char) - base + shift) % 26 + base)
        else:
            result += char
    return result

ciphertext = 'uapv{yjhi_pc_xcigd_rxewtg}'
plaintext = caesar_decrypt(ciphertext, 11)
```

## Solution

**Decrypted Flag:** `flag{just_an_intro_cipher}`

## Summary

| Property | Value |
|----------|-------|
| Cipher Type | Caesar / ROT Cipher |
| Encryption Shift | ROT15 |
| Decryption Shift | ROT11 |
| Original Message | `flag{just_an_intro_cipher}` |

The flag humorously acknowledges that this was an introductory cipher challenge, fitting for the classic and straightforward nature of the Caesar cipher.
