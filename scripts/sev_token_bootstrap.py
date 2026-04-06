import os
from pathlib import Path

# schwabdev
import schwabdev

CLIENT_ID = os.environ["SCHWAB_CLIENT_ID"]
CLIENT_SECRET = os.environ["SCHWAB_CLIENT_SECRET"]
REDIRECT_URI = os.environ.get("SCHWAB_REDIRECT_URI", "https://127.0.0.1")

TOKEN_PATH = Path(os.environ.get("SCHWAB_TOKEN_PATH", "/quantstack/secrets/schwab/token.json"))

def main():
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)

    # schwabdev typically handles token storage/refresh for you; we keep token path explicit for QuantStack hygiene.
    # The important part: use a manual-style auth where you paste the final redirected URL (no local callback server needed).
    client = schwabdev.Client(
        app_key=CLIENT_ID,
        app_secret=CLIENT_SECRET,
        callback_url=REDIRECT_URI,
        token_path=str(TOKEN_PATH),
    )

    # Trigger auth if token missing/expired
    # schwabdev may open a URL or print one depending on version; follow prompts and paste the final redirected URL if asked.
    r = client.account_numbers()
    print("✅ Schwabdev auth OK. Token at:", TOKEN_PATH)
    print(r)

if __name__ == "__main__":
    main()
