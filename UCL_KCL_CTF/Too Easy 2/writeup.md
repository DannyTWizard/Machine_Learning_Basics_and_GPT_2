# Too Easy 2 - CTF Writeup

## Challenge Description

> While developing the app, Reggie created some dummy data that he might have left when he deployed the app to production. Are you able to find what he left behind?
>
> NOTE: This challenge uses the same endpoint as Too Easy 1. You don't need to have solved the first challenge to solve this one.
>
> http://13.53.139.173:9000/

## Solution

### Step 1: Initial Reconnaissance

The target is an "Inventory App" that allows users to search for items in a database. The search functionality accepts user input through a GET parameter called `search`.

### Step 2: Identifying the Vulnerability

Testing the search field with a single quote (`'`) produced an SQL error:

```
Error: unrecognized token: "'''"
```

This error message is characteristic of SQLite and confirms that the application is vulnerable to SQL injection. The user input is being directly concatenated into the SQL query without proper sanitization or parameterization.

### Step 3: Exploiting the SQL Injection

Since the challenge mentions "dummy data" left by the developer, the goal was to retrieve all records from the database to find any hidden entries.

Using the classic SQL injection payload:

```
' OR 1=1--
```

This payload:
1. Closes the string literal with `'`
2. Adds `OR 1=1` which always evaluates to true, returning all rows
3. Comments out the rest of the query with `--`

The likely original query:
```sql
SELECT * FROM items WHERE name LIKE '%[user_input]%'
```

Becomes:
```sql
SELECT * FROM items WHERE name LIKE '%' OR 1=1--%'
```

### Step 4: Finding the Flag

The injection returned all 16 items in the inventory database. Among the regular items (Laptop, Mouse, Keyboard, etc.), there was one unusual entry:

| Name | Description |
|------|-------------|
| kJ9mXp2qnL4vRt6yW8bN3cF1hZ5sD7eA | flag{c4n_i_g3t_4n_1nj3ct1on?} |

This was the dummy data that Reggie left behind during development.

---

## Summary

| Field | Value |
|-------|-------|
| **Original Message/Dummy Data Name** | `kJ9mXp2qnL4vRt6yW8bN3cF1hZ5sD7eA` |
| **Flag** | `flag{c4n_i_g3t_4n_1nj3ct1on?}` |

---

## Key Takeaways

1. **SQL Injection vulnerability** - The application does not sanitize or parameterize user input before including it in SQL queries
2. **Development artifacts** - The developer left test/dummy data in the production database
3. **Information disclosure** - Error messages revealed the database type (SQLite), which helped craft the exploit

## Remediation

1. Use parameterized queries or prepared statements
2. Remove all test/development data before deploying to production
3. Implement proper error handling that doesn't expose internal details
4. Validate and sanitize all user inputs
