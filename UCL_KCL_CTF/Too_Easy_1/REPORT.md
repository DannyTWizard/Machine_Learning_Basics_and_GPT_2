# Too Easy 1 - CTF Challenge Report

## Challenge Information
- **Name:** Too Easy 1
- **URL:** http://13.53.139.173:9000/
- **Category:** Web Security / SQL Injection

## Challenge Description
Reggie was interested in building a small application to track inventory at the Student Union Shop. For now, you can simply search for things, but there might be a secret lurking...

## Solution Summary

### Vulnerability: SQL Injection

The Inventory App search functionality is vulnerable to SQL injection. The application takes user input from the `search` parameter and directly inserts it into an SQL query without proper sanitization or parameterization.

### Steps to Solve

1. **Initial Reconnaissance**
   - Accessed the web application at http://13.53.139.173:9000/
   - Found a simple search form that queries an inventory database
   - Server identified as Werkzeug/Python (Flask application)

2. **Vulnerability Discovery**
   - Tested the search parameter with a single quote: `'`
   - Received an SQL error: `unrecognized token: "'''"`
   - This confirmed SQL injection vulnerability and identified SQLite as the database

3. **Exploitation**
   - Used a classic SQL injection payload to bypass the search filter:
   ```
   ' OR '1'='1
   ```
   - URL encoded: `%27%20OR%20%271%27%3D%271`

4. **Result**
   - The payload forced the SQL query to return all rows in the database
   - Among the regular inventory items (Laptop, Mouse, Keyboard, etc.), a hidden entry was revealed containing the flag

### Technical Analysis

The vulnerable code likely looks something like:
```python
query = f"SELECT * FROM items WHERE name LIKE '%{search}%'"
```

When we inject `' OR '1'='1`, the query becomes:
```sql
SELECT * FROM items WHERE name LIKE '%' OR '1'='1%'
```

The condition `'1'='1'` is always TRUE, causing the query to return ALL rows, including the hidden flag entry.

## Flag

**Flag:** `flag{c4n_i_g3t_4n_1nj3ct1on?}`

The flag is a play on words: "can I get an injection?" - referencing both the SQL injection vulnerability and possibly medical injections.

### Hidden Entry Details
- **Item Name:** `kJ9mXp2qnL4vRt6yW8bN3cF1hZ5sD7eA` (random string used to hide the entry)
- **Description:** `flag{c4n_i_g3t_4n_1nj3ct1on?}`

## Remediation Recommendations

To fix this vulnerability, the application should:

1. **Use Parameterized Queries (Prepared Statements)**
   ```python
   cursor.execute("SELECT * FROM items WHERE name LIKE ?", ('%' + search + '%',))
   ```

2. **Use an ORM like SQLAlchemy**
   ```python
   items = Item.query.filter(Item.name.like(f'%{search}%')).all()
   ```

3. **Input Validation**
   - Sanitize and validate all user inputs
   - Use allowlist approaches where possible

4. **Principle of Least Privilege**
   - Ensure database accounts have minimal necessary permissions
   - Separate sensitive data into different tables with access controls

## Files

- `exploit.py` - Python script to automatically exploit the vulnerability
- `REPORT.md` - This report

## Tools Used

- curl - HTTP requests
- Python requests library - Automated exploitation
