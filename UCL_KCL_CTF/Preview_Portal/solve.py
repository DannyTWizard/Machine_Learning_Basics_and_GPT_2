#!/usr/bin/env python3
"""
CTF Challenge: Preview_Portal
Solution script to extract the flag from client-side JavaScript
"""

import requests
import re

URL = "http://13.53.139.173/web3/"

def solve():
    # Fetch the page
    response = requests.get(URL)

    # Search for the flag in the JavaScript code
    flag_pattern = r'flag\{[^}]+\}'
    match = re.search(flag_pattern, response.text)

    if match:
        flag = match.group(0)
        print(f"[+] Found flag: {flag}")
        return flag
    else:
        print("[-] Flag not found")
        return None

if __name__ == "__main__":
    solve()
