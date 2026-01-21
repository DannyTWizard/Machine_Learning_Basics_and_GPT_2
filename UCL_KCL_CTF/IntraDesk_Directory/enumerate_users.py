import requests
import string
import itertools
from concurrent.futures import ThreadPoolExecutor

URL = "http://13.53.139.173/web1/"

def search(user):
    try:
        r = requests.get(URL, params={"user": user, "dept": "All", "office": "Any"}, timeout=10)
        return "1 result" in r.text
    except:
        return False

# Test basic users first
basic_users = ["admin", "guest", "alice", "bob", "test", "user", "root", "flag", "secret"]
print("Basic users:")
for u in basic_users:
    if search(u):
        print(f"  Found: {u}")

# Try 2-3 letter combinations that might be usernames
print("\nTrying short usernames (2-4 chars)...")
found = []

# Common short usernames
short_names = ["aa", "ab", "ac", "ad", "ae", "al", "am", "an", "be", "bo", "ca", "ch", "da", "de", "ed", "em", "ev", "fl", "fr", "ge", "gu", "ha", "he", "ho", "ia", "ja", "je", "jo", "ka", "ke", "ki", "la", "le", "li", "lu", "ma", "me", "mi", "mo", "na", "ne", "ni", "no", "ol", "pa", "pe", "pi", "ra", "re", "ri", "ro", "sa", "se", "si", "so", "st", "su", "ta", "te", "ti", "to", "vi", "wa", "we", "wi", "xa", "ya", "ze"]

for name in short_names:
    if search(name):
        found.append(name)
        print(f"  Found: {name}")

# Try common names
common_names = [
    "aaron", "adam", "alex", "amanda", "amy", "andrew", "anna", "anthony", "ashley",
    "benjamin", "brian", "brittany", "carlos", "charles", "chris", "christina", 
    "daniel", "david", "emily", "emma", "eric", "frank", "george", "hannah",
    "heather", "jack", "jacob", "james", "jason", "jennifer", "jessica", "john",
    "joseph", "joshua", "karen", "kevin", "kim", "kyle", "laura", "linda", "lisa",
    "mark", "mary", "matthew", "melissa", "michael", "michelle", "mike", "nancy",
    "natalie", "nicholas", "nicole", "oliver", "patricia", "patrick", "paul", "peter",
    "rachel", "rebecca", "richard", "robert", "ryan", "samantha", "samuel", "sarah",
    "sean", "stephanie", "stephen", "steven", "susan", "thomas", "timothy", "tyler",
    "victoria", "william", "zach"
]

print("\nTrying common names...")
for name in common_names:
    if search(name):
        if name not in ["admin", "guest", "alice"]:
            found.append(name)
            print(f"  Found: {name}")

print(f"\nTotal found: {found}")
