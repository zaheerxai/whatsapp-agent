from msal import PublicClientApplication

# Paste your Application (client) ID from Azure App registration Overview
CLIENT_ID = "c215cb27-ffba-47c2-8f32-7c3dec27edf5"

SCOPES = [
    "Files.ReadWrite",
    "User.Read",
]

app = PublicClientApplication(
    CLIENT_ID,
    authority="https://login.microsoftonline.com/consumers",  # personal accounts
)

print("A browser window will open. Sign in with xaheeru23@gmail.com ...")
result = app.acquire_token_interactive(scopes=SCOPES)

if "access_token" in result:
    print("\n=== SUCCESS ===")
    print("ONEDRIVE_CLIENT_ID=" + CLIENT_ID)
    if result.get("refresh_token"):
        print("ONEDRIVE_REFRESH_TOKEN=" + result["refresh_token"])
    else:
        print("No refresh_token. Full result keys:", list(result.keys()))
        print(result)
else:
    print("\n=== FAILED ===")
    print(result)