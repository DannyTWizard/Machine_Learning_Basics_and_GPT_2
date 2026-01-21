# The Logs Don't Lie - CTF Challenge Writeup

## Summary

| Item | Value |
|------|-------|
| **Original Message / Flag** | `flag{them_logs_dont_lie}` |
| **XOR Key** | `bluewhale` |
| **Key Source** | Command line in sysmon.json: `cmd.exe /c setx SYNC_TOKEN "bluewhale"` |

---

## Challenge Description

An analyst recovered endpoint telemetry and a small binary artifact from a user profile. One of these contains the clue you need to decode the other.

**Files provided:**
- `cache.bin` - A 24-byte binary file (encrypted data)
- `sysmon.json` - Sysmon event log export

---

## Solution Process

### Step 1: Initial Analysis

First, I examined both files to understand their structure:

**cache.bin** (24 bytes):
```
04 00 14 02 0c 1c 09 09 08 3d 00 1a 02 04 37 05 03 0b 16 33 19 0c 12 15
```

**sysmon.json**: A JSON array containing Windows Sysmon event logs with:
- EventID 1: Process creation events
- EventID 3: Network connection events
- EventID 11: File creation events

### Step 2: Finding the Key in Sysmon Logs

Analyzing the sysmon.json file, I searched through command lines for suspicious activity. I found a critical command:

```
cmd.exe /c setx SYNC_TOKEN "bluewhale"
```

This sets an environment variable `SYNC_TOKEN` to the value `bluewhale`.

Additionally, I found PowerShell commands referencing the cache.bin file:
```
powershell -NoP -W Hidden -Command "$p='C:\Users\User\AppData\Roaming\cache.bin'; ..."
```

This confirmed that `bluewhale` is likely the XOR key used to encrypt cache.bin.

### Step 3: Decryption

Using XOR decryption with the key `bluewhale`:

```python
key = 'bluewhale'
cache_bytes = [4, 0, 20, 2, 12, 28, 9, 9, 8, 61, 0, 26, 2, 4, 55, 5, 3, 11, 22, 51, 25, 12, 18, 21]

decrypted = []
for i, byte in enumerate(cache_bytes):
    key_byte = ord(key[i % len(key)])
    decrypted.append(chr(byte ^ key_byte))

message = ''.join(decrypted)
# Result: flag{them_logs_dont_lie}
```

**XOR operation breakdown:**

| Index | Cache Byte | Key Char | Key Byte | XOR Result | Char |
|-------|------------|----------|----------|------------|------|
| 0     | 0x04       | b        | 0x62     | 0x66       | f    |
| 1     | 0x00       | l        | 0x6c     | 0x6c       | l    |
| 2     | 0x14       | u        | 0x75     | 0x61       | a    |
| 3     | 0x02       | e        | 0x65     | 0x67       | g    |
| 4     | 0x0c       | w        | 0x77     | 0x7b       | {    |
| ...   | ...        | ...      | ...      | ...        | ...  |

---

## Files

- `solve.py` - Python solution script that automatically finds the key and decrypts the message

## How to Run

```bash
python solve.py
```

---

## Flag

```
flag{them_logs_dont_lie}
```
