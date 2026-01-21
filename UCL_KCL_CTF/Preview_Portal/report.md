# CTF Challenge: Preview_Portal

## Challenge Description
An internal preview portal that claims to restrict access to authorised staff only. You are not authorised.

**URL:** http://13.53.139.173/web3/

## Solution

### Step 1: Initial Reconnaissance

I started by fetching the webpage using `curl -v` to examine both the HTTP headers and the page content.

```bash
curl -v http://13.53.139.173/web3/
```

### Step 2: Source Code Analysis

Upon examining the HTML source code, I discovered that the page contained a "Request Access" button that, when clicked, simply displays "Access denied. Your request has been logged."

However, looking at the JavaScript code embedded in the page, I found the following:

```javascript
<script>
const flag = "flag{cli3nt_side_is_not_s3curity}";

function checkAccess() {
  document.getElementById("msg").innerText =
    "Access denied. Your request has been logged.";
}
</script>
```

### Step 3: Vulnerability Identified

The vulnerability is **client-side security through obscurity**. The developers made a critical mistake by:

1. Storing the flag directly in the client-side JavaScript code
2. Assuming users would only interact with the page through the UI (clicking the button)
3. Not realizing that anyone can view the page source code

This is a fundamental web security flaw - **client-side code is never secure** because users have complete access to anything sent to their browser.

## Flag

```
flag{cli3nt_side_is_not_s3curity}
```

## Original Message

The flag itself contains the lesson: **"client side is not security"** (with leetspeak: `cli3nt_side_is_not_s3curity`)

This message emphasizes that:
- Never trust client-side validation alone
- Sensitive data should never be stored in JavaScript
- Authentication and authorization must be performed server-side
- Any security checks done in JavaScript can be bypassed by viewing the source

## Tools Used

- `curl` - for fetching the raw HTTP response and viewing source code
- Python `requests` library - for automated flag extraction (see `solve.py`)

## Lessons Learned

1. Always view the page source when analyzing web challenges
2. Check JavaScript files for hardcoded secrets
3. Client-side "security" is just theater - real security must be server-side
