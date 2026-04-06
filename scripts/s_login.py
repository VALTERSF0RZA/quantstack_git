from pathlib import Path
import schwab

CLIENT_ID = ""
CLIENT_SECRET = ""

# MUST match Schwab dev portal exactly
REDIRECT_URI = "https://127.0.0.1"

TOKEN_PATH = Path("/quantstack/secrets/schwab/token.json")

def main():
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)

    client = schwab.auth.client_from_login_flow(
        CLIENT_ID,
        CLIENT_SECRET,
        REDIRECT_URI,
        str(TOKEN_PATH),
        asyncio=False,
        interactive=True,
    )

    r = client.get_account_numbers()
    r.raise_for_status()
    print("✅ Token OK. Saved:", TOKEN_PATH)

if __name__ == "__main__":
    main()
