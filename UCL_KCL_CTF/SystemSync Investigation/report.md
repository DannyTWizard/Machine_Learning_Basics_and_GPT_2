# SystemSync Investigation - CTF Challenge Writeup

## Challenge Description
A host is suspected of running a persistence mechanism named SystemSync. We are given several artefacts from the investigation and need to reconstruct the validation token to obtain the flag.

## Artefacts Analyzed
1. **registry.txt** - Windows Registry dump
2. **sysmon_final.json** - Sysmon event logs
3. **tasks.xml** - Windows Scheduled Task XML

---

## Analysis

### 1. Registry Analysis (registry.txt)

The registry dump contained a suspicious `SystemSync` key:

```
HKEY_LOCAL_MACHINE\SOFTWARE\SystemSync
  InstallPath    REG_SZ    C:\ProgramData\SystemSync
  CacheMode      REG_DWORD 0x1
  SyncLabel      REG_BINARY 72 70 75 62
```

**Key Finding:** The `SyncLabel` value contains hex bytes `72 70 75 62`, which decode to ASCII characters:
- 0x72 = 'r'
- 0x70 = 'p'
- 0x75 = 'u'
- 0x62 = 'b'

**SyncLabel = "rpub"**

### 2. Scheduled Task Analysis (tasks.xml)

The XML file defines a scheduled task that runs PowerShell with elevated privileges (SYSTEM):

```xml
<Command>C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe</Command>
<Arguments>-NoP -W Hidden -Command "$modeTag = 'qrygn'; # legacy compatibility
$null=$modeTag; exit 0"</Arguments>
```

**Key Finding:** The `modeTag` variable is set to **"qrygn"**

### 3. Sysmon Log Analysis (sysmon_final.json)

The Sysmon logs revealed several SystemSync-related activities:

1. Creating the scheduled task:
   ```
   schtasks /create /tn "SystemSync" /xml "C:\ProgramData\SystemSync\SystemSync.xml" /f
   ```

2. Setting the registry value:
   ```
   reg add "HKLM\SOFTWARE\SystemSync" /v SyncLabel /t REG_BINARY /d 72707562 /f
   ```

3. PowerShell session initialization:
   ```
   powershell -NoProfile -Command "$syncId = 42; Write-Output 'SystemSync session 42 initialized' | Out-Null; exit 0"
   ```

**Key Finding:** The `syncId` value is **42**

---

## Decoding

All three encoded values appear to use **ROT13** cipher:

| Component | Encoded Value | ROT13 Decoded |
|-----------|---------------|---------------|
| SyncLabel | rpub | **echo** |
| modeTag | qrygn | **delta** |
| syncId | 42 | **42** (not encoded) |

### ROT13 Verification:
- 'r' -> 'e' (18 - 13 = 5)
- 'p' -> 'c' (16 - 13 = 3)
- 'u' -> 'h' (21 - 13 = 8)
- 'b' -> 'o' (2 + 13 = 15)

- 'q' -> 'd' (17 - 13 = 4)
- 'r' -> 'e' (18 - 13 = 5)
- 'y' -> 'l' (25 - 13 = 12)
- 'g' -> 't' (7 + 13 = 20)
- 'n' -> 'a' (14 - 13 = 1)

---

## Summary

### Original Message (Validation Token)
The decoded components combine to form:

**Original Message: `echo delta 42`**

### Flag
Based on standard CTF flag formats, the flag is:

**`flag{echo_delta_42}`**

Alternative formats that may be valid:
- `ctf{echo_delta_42}`
- `FLAG{echo_delta_42}`
- `CTF{echo_delta_42}`

---

## Persistence Mechanism Summary

The SystemSync persistence mechanism works as follows:
1. Creates a scheduled task that runs at elevated privileges
2. Stores configuration data in the registry (encoded with ROT13)
3. Uses PowerShell with hidden window to execute commands
4. The task is masquerading as a legitimate "Microsoft Corporation" authored task

This is a classic Windows persistence technique using scheduled tasks combined with registry storage for configuration data.
