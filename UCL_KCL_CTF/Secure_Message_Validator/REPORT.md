# Secure Message Validator - CTF Challenge Report

## Challenge Description

A secure message validation service is deployed that allows submission of modified ciphertexts and only indicates whether the padding is valid. The goal is to recover the original encrypted message using only this padding validity information.

## Given Information

- **Service URL**: http://13.53.139.173/crypto3/
- **IV (hex)**: `18aeb3d267f3f9a9803dc9f16bcac18c`
- **Ciphertext (hex)**: `9130f576cd1318c91ffe610fb44e6feeefb2e6cb2fd3d3c52d43cd0496464cc7ee0454534c76b6c6d7f27ffb4d0a3052d72aea6ab20af96ff82016e4aa4237da`

## Vulnerability Analysis

The challenge describes a classic **Padding Oracle Attack** scenario. Key observations:

1. The service uses AES-CBC encryption (evident from the 16-byte IV and block-aligned ciphertext)
2. The ciphertext is 64 bytes = 4 blocks of 16 bytes each
3. The oracle reveals only whether the padding is valid or not
4. This is enough information to completely decrypt the ciphertext

## Attack Methodology

### Padding Oracle Attack Background

In AES-CBC mode, decryption works as follows:
```
Plaintext[i] = Decrypt(Ciphertext[i]) XOR Ciphertext[i-1]
```

Where `Ciphertext[0]` is the IV for the first block.

PKCS#7 padding is used, where the padding bytes contain the value equal to the number of padding bytes. For example:
- 1 byte of padding: `0x01`
- 2 bytes of padding: `0x02 0x02`
- 3 bytes of padding: `0x03 0x03 0x03`

### The Attack

For each ciphertext block (working from last to first), and for each byte position (working from last to first):

1. Create a malicious "previous block" that we control
2. Try all 256 possible values for the target byte position
3. When the oracle returns "valid padding", we know:
   - `Decrypt(Block)[position] XOR our_guess = padding_value`
   - Therefore: `Intermediate = our_guess XOR padding_value`
4. The actual plaintext byte is: `Intermediate XOR Original_Previous_Block[position]`

This allows us to recover each plaintext byte using at most 256 oracle queries per byte.

### Implementation Details

The attack was implemented in Python with the following optimizations:
- **Concurrent requests**: Using ThreadPoolExecutor with 32 workers to parallelize the 256 guesses per byte
- **Connection pooling**: Using requests.Session() for faster HTTP connections
- **Edge case handling**: Verification step for the last byte to avoid false positives when padding value happens to equal 0x01

## Results

### Decrypted Message

```
internal_note: do not share -> flag{padding_oracle_gg} <- end
```

### Flag

```
flag{padding_oracle_gg}
```

### Block-by-Block Decryption

| Block | Plaintext |
|-------|-----------|
| 1 | `internal_note: d` |
| 2 | `o not share -> f` |
| 3 | `lag{padding_orac` |
| 4 | `le_gg} <- end` + 3 bytes PKCS#7 padding |

## Files

- `padding_oracle_attack.py` - The exploit script

## Conclusion

The challenge demonstrated the danger of revealing padding validity in CBC mode encryption. Even though no key or plaintext is directly exposed, the simple binary feedback (valid/invalid padding) is sufficient to completely decrypt any ciphertext.

This vulnerability was famously exploited in real-world attacks against:
- ASP.NET (CVE-2010-3332)
- SSL/TLS (POODLE attack)
- Various web applications

The mitigation is to use authenticated encryption modes (like AES-GCM) that verify message integrity before attempting decryption, making padding oracle attacks impossible.
