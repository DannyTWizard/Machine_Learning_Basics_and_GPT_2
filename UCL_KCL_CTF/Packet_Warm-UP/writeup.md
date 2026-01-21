# Packet Warm-UP - CTF Challenge Writeup

## Summary

| Item | Value |
|------|-------|
| **Challenge Name** | Packet Warm-UP |
| **Category** | Network Forensics |
| **Original Message (Base64)** | `ZmxhZ3twY2FwX3c0cm11cH0=` |
| **Flag** | `flag{pcap_w4rmup}` |

---

## Challenge Description

We captured some network traffic from an internal host. Find the hidden flag from the capture.

## Analysis Process

### Step 1: Initial Packet Analysis

First, I examined the pcap file using `tcpdump` to understand the network traffic:

```bash
tcpdump -r warmup.pcap
```

This revealed several types of traffic:
- ICMP echo request/reply (ping) between 192.168.56.20 and 192.168.56.10
- DNS query to 8.8.8.8 for example.com
- HTTP traffic on port 80 between 192.168.56.20 and 192.168.56.10
- SSH connection attempts to 192.168.56.30

### Step 2: Examining HTTP Traffic

The HTTP traffic was the most interesting. I saw:
1. A TCP 3-way handshake establishing a connection
2. An HTTP GET request to `/download` endpoint
3. An HTTP 200 OK response from the server

### Step 3: Extracting Packet Contents

Using the hex dump option (`-X`), I examined the actual packet contents:

```bash
tcpdump -r warmup.pcap -X
```

The HTTP GET request:
```
GET /download HTTP/1.1
Host: files.internal
User-Agent: curl/7.68.0
```

The HTTP response contained:
```
HTTP/1.1 200 OK
Server: Apache/2.4.41
Content-Type: application/octet-stream
Content-Length: 24

ZmxhZ3twY2FwX3c0cm11cH0=
```

### Step 4: Decoding the Flag

The response body `ZmxhZ3twY2FwX3c0cm11cH0=` is clearly Base64 encoded (indicated by the `=` padding and character set). Decoding it:

```bash
echo "ZmxhZ3twY2FwX3c0cm11cH0=" | base64 -d
```

Result: `flag{pcap_w4rmup}`

## Tools Used

- `tcpdump` - for reading and analyzing the pcap file
- `base64` - for decoding the Base64 encoded flag

## Key Takeaways

1. Always examine HTTP traffic in packet captures - it often contains valuable data
2. Look for Base64 encoded strings (recognizable by their character set and `=` padding)
3. The `Content-Type: application/octet-stream` header indicated the response contained binary/encoded data

---

## Final Answer

**Flag: `flag{pcap_w4rmup}`**
