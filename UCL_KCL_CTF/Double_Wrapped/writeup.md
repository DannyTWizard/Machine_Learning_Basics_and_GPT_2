# Double Wrapped - CTF Challenge Writeup

## Challenge Description
**Challenge Name:** Double Wrapped
**Category:** Cryptography / Encoding
**Challenge:** The data has been transformed more than once. Undo the layers to recover the flag.

## Analysis

### Initial Data
The challenge provided a file `chef.txt.txt` containing the following encoded string:

```
NjYgNmMgNjEgNjcgN2IgNmMgNjEgNzkgNjUgNzIgNzMgNWYgNjEgNmUgNjQgNWYgNmMgNjEgNzkgNjUgNzIgNzMgN2Q=
```

The challenge name "Double Wrapped" hints that there are two layers of encoding to unwrap.

### Layer 1: Base64 Decoding

The initial string ends with `=` (padding character) and contains only alphanumeric characters, which is characteristic of Base64 encoding.

Decoding the Base64 string reveals:
```
66 6c 61 67 7b 6c 61 79 65 72 73 5f 61 6e 64 5f 6c 61 79 65 72 73 7d
```

### Layer 2: Hexadecimal to ASCII

The decoded output consists of space-separated two-character hex values. Each hex pair represents an ASCII character:

| Hex | ASCII |
|-----|-------|
| 66  | f     |
| 6c  | l     |
| 61  | a     |
| 67  | g     |
| 7b  | {     |
| 6c  | l     |
| 61  | a     |
| 79  | y     |
| 65  | e     |
| 72  | r     |
| 73  | s     |
| 5f  | _     |
| 61  | a     |
| 6e  | n     |
| 64  | d     |
| 5f  | _     |
| 6c  | l     |
| 61  | a     |
| 79  | y     |
| 65  | e     |
| 72  | r     |
| 73  | s     |
| 7d  | }     |

Converting all hex values to their ASCII equivalents yields the flag.

## Solution

The encoding layers in order (from outer to inner):
1. **Base64** - Standard Base64 encoding
2. **Hexadecimal** - Space-separated hex representation of ASCII characters

To decode:
1. Base64 decode the original string
2. Convert each hex pair to its ASCII character

## Tools Used
- Python 3 (base64 library)
- Command line tools: `base64 -d`, `xxd -r -p`

---

## Summary

**Original Encoded Message:**
```
NjYgNmMgNjEgNjcgN2IgNmMgNjEgNzkgNjUgNzIgNzMgNWYgNjEgNmUgNjQgNWYgNmMgNjEgNzkgNjUgNzIgNzMgN2Q=
```

**Decoded Message / Flag:**
```
flag{layers_and_layers}
```

---

## Files
- `solve.py` - Python script that automates the decoding process
