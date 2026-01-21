# IntraDesk Directory - CTF Challenge Investigation Report

## Challenge Overview

**Challenge Name:** IntraDesk Directory
**URL:** http://13.53.139.173/web1/
**Category:** Web Security (Likely LDAP Injection)

**Challenge Description:**
> An internal directory search tool used by employees to look up user accounts. Search results appear to be limited, and some inputs may be filtered. See what information you can extract.

## Application Analysis

### Functionality
The IntraDesk Directory is a web application that allows users to search for employee accounts by username. The interface includes:
- Username search field
- Department dropdown (IT, Finance, HR, Operations, All)
- Office dropdown (London, New York, Singapore, Any)

### Known Users Discovered
Through enumeration, three valid users were identified:
- `admin` - Active
- `guest` - Active
- `alice` - Active

### Key Observations
1. The search requires an **exact match** - partial usernames do not return results
2. Wildcard character `*` does NOT work - returns "0 results"
3. Department and office fields do not appear to filter results (likely decorative)
4. The response shows: Username, Department (always "—"), and Status (Active)

## Vulnerability Assessment

### LDAP Injection Testing

The challenge hints at LDAP directory search. Extensive LDAP injection testing was performed:

#### Wildcard Bypass Attempts
- Standard wildcard: `*` - Blocked (0 results)
- Partial wildcards: `admin*`, `*admin`, `a*` - Blocked
- URL encoded: `%2a`, `%252a` - Blocked
- Unicode asterisks: `＊`, `✱`, `⋆` - Blocked
- LDAP escape: `\2a`, `\x2a` - Blocked
- HTML entities: `&#42;` - Blocked

#### Filter Injection Attempts
- Closing parenthesis: `admin)` - 0 results
- OR injection: `admin)(|(uid=guest)` - 0 results
- AND injection: `admin)(&)(uid=guest` - 0 results
- ObjectClass injection: `admin)(objectClass=*` - 0 results
- Attribute injection: `admin)(description=*` - 0 results
- Negation: `admin)(!(uid=admin)` - 0 results

#### Other Injection Types Tested
- SQL injection: `' OR '1'='1`, `admin'--` - Not vulnerable
- NoSQL injection: `$ne`, `$regex` - Not vulnerable
- SSTI: `{{7*7}}`, `${7*7}` - Not vulnerable
- Path traversal: `../../../etc/passwd` - Not vulnerable
- XXE: XML entity injection - Not vulnerable

### Parameter Manipulation
- Pagination parameters: `limit`, `offset`, `page`, `size` - No effect
- Attribute selection: `attrs`, `fields`, `columns` - No effect
- Format parameters: `format=json`, `output=ldif` - No effect
- Debug parameters: `debug=1`, `verbose=1` - No effect

### Response Analysis
- Valid user response: ~6507 bytes, contains result table
- Invalid/no match response: ~6231 bytes, shows "0 results"
- Empty input: ~6094 bytes, no result section
- No timing differences detected between different query types

## Techniques Not Successful

1. **LDAP Wildcard Bypass** - All encoding and bypass techniques failed
2. **Blind LDAP Injection** - Could not detect boolean differences
3. **Time-based Injection** - No timing variations observed
4. **Parameter Pollution** - Multiple user parameters ignored
5. **Header Injection** - Custom headers had no effect
6. **Username Enumeration** - Only found admin, guest, alice despite extensive wordlists

## Conclusion

**STATUS: FLAG NOT FOUND**

Despite extensive testing with multiple LDAP injection techniques, encoding bypasses, and enumeration strategies, the vulnerability was not successfully exploited. Possible reasons:

1. The application may be using parameterized LDAP queries or properly escaping all special characters
2. There may be a specific bypass technique not covered in this investigation
3. The wildcard filter may be implemented at multiple layers (application + LDAP server)
4. The challenge may require a different approach entirely

## Files Created
- `blind_ldap.py` - Blind LDAP injection testing script
- `enumerate_users.py` - User enumeration script
- `comprehensive_enum.py` - Comprehensive username enumeration

## Recommendations for Further Investigation
1. Check for LDAP-specific bypass techniques not covered (extensible matching rules)
2. Investigate if there's a different API endpoint
3. Look for server-side template injection with different syntax
4. Consider if the flag might be in a configuration file or database accessible via different vulnerability

---

## Summary

**Original Message/Flag:** NOT FOUND

The challenge appears to be an LDAP injection challenge where wildcards are filtered to prevent unauthorized enumeration of the user directory. Despite extensive testing, the bypass technique was not discovered. The three known users (admin, guest, alice) all display the same limited information (username, empty department, Active status).

Further investigation or hints may be needed to solve this challenge.
