import logging
import sys

import instagram_client

# Reconfigure stdout for UTF-8 to handle special characters
sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    cl = instagram_client.build_client()
    threads = instagram_client.list_threads(cl, amount=20)

    print("\nYour DM threads:\n")
    for t in threads:
        kind = "group" if t["is_group"] else "dm"
        title = t["title"] or "(untitled)"
        print(f"[{kind}] {title!r:40} id={t['id']}  users={t['user_count']}")

    print("\nCopy the id of your target group chat into TARGET_THREAD_ID in .env")


if __name__ == "__main__":
    main()
