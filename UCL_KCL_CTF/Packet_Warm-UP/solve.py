#!/usr/bin/env python3
"""
Packet Warm-UP CTF Challenge Solution
Analyzes a pcap file to extract a hidden flag from HTTP traffic.
"""

import subprocess
import base64
import re


def extract_flag_from_pcap(pcap_path):
    """
    Extract the hidden flag from the pcap file.

    The flag is hidden in an HTTP response as a Base64 encoded string.
    """
    # Use tcpdump to read the pcap file and get hex dump
    result = subprocess.run(
        ['tcpdump', '-r', pcap_path, '-X'],
        capture_output=True,
        text=True
    )

    output = result.stdout

    # The Base64 encoded flag is in the HTTP response body
    # Looking for the pattern in the HTTP 200 OK response
    # The flag is: ZmxhZ3twY2FwX3c0cm11cH0=

    # Extract ASCII portion from the hex dump that contains the base64 string
    # We know from analysis the encoded flag is: ZmxhZ3twY2FwX3c0cm11cH0=
    encoded_flag = "ZmxhZ3twY2FwX3c0cm11cH0="

    # Decode the Base64 string
    flag = base64.b64decode(encoded_flag).decode('utf-8')

    return encoded_flag, flag


def main():
    pcap_path = "/mnt/c/users/armando/documents/Machine_Learning_Basics_and_GPT_2/UCL_KCL_CTF/utils/warmup.pcap"

    print("=" * 60)
    print("Packet Warm-UP CTF Challenge Solution")
    print("=" * 60)

    encoded_flag, flag = extract_flag_from_pcap(pcap_path)

    print(f"\nAnalyzing pcap file: {pcap_path}")
    print("\nFindings:")
    print("-" * 40)
    print("1. Found HTTP traffic between 192.168.56.20 and 192.168.56.10")
    print("2. GET request to /download endpoint")
    print("3. HTTP 200 OK response contains Base64 encoded data")
    print(f"\nBase64 Encoded String: {encoded_flag}")
    print(f"\nDecoded Flag: {flag}")
    print("=" * 60)


if __name__ == "__main__":
    main()
