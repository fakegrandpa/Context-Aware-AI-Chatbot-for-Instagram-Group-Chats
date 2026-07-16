import os

from instagrapi import Client
from instagrapi.exceptions import LoginRequired

# Paste your own sessionid here, or set IG_SESSIONID in your environment / .env
SESSION_ID = os.getenv("IG_SESSIONID", "your_instagram_sessionid")

cl = Client()

try:
    print("[1] Logging in using browser sessionid...")
    cl.login_by_sessionid(SESSION_ID)

    print("[2] Checking account...")
    account = cl.account_info()
    print(f"SUCCESS: Logged in as @{account.username}")
    print(f"User ID: {account.pk}")

    print("[3] Testing Instagram Direct API...")
    threads = cl.direct_threads(amount=1)

    print("SUCCESS: Direct API works!")
    if threads:
        print(f"First thread ID: {threads[0].id}")

    # Save the resulting instagrapi settings if everything works
    cl.dump_settings("working_session.json")
    print("[4] Saved working session to working_session.json")

except LoginRequired as e:
    print("FAILED: Instagram returned login_required")
    print(e)

except Exception as e:
    print(f"FAILED: {type(e).__name__}")
    print(e)