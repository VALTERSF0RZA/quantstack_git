from pathlib import Path
import schwab

CLIENT_ID = ""
CLIENT_SECRET = ""
REDIRECT_URI = "https://127.0.0.1:8182"
TOKEN_PATH = Path("/quantstack/secrets/schwab/token.json")

def main():
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    client = schwab.auth.client_from_manual_flow(
        CLIENT_ID, CLIENT_SECRET, REDIRECT_URI, str(TOKEN_PATH),
        asyncio=False,
    )
    r = client.get_account_numbers()
    r.raise_for_status()
    print("✅ Token OK. Saved:", TOKEN_PATH)

if __name__ == "__main__":
    main()

