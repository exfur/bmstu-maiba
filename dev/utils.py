import os

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# Загружаем переменные окружения
load_dotenv()

# ==========================================
# КОНФИГУРАЦИЯ ПУТЕЙ И GOOGLE API
# ==========================================
GOOGLE_API_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar",
]

GOOGLE_API_SECRET_PATH = os.path.expanduser("~/google_api_client_secret.json")
GOOGLE_API_TOKEN_PATH = os.path.expanduser("~/google_api_token.json")


# ==========================================
# АВТОРИЗАЦИЯ И ПОЛУЧЕНИЕ ДАННЫХ
# ==========================================
def get_creds(scopes=None):
    """
    Returns User Credentials (3-Legged OAuth).
    Opens a browser window for the first login, then uses a saved JSON token.
    """
    if scopes is None:
        scopes = GOOGLE_API_SCOPES

    creds = None
    token_path = GOOGLE_API_TOKEN_PATH

    # 1. Load existing token if available
    if os.path.exists(token_path):
        try:
            creds = Credentials.from_authorized_user_file(token_path, scopes)
        except Exception:
            # If the file is corrupted or incompatible, ignore it
            creds = None

    # 2. If no valid credentials, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                # Refresh the token silently
                creds.refresh(Request())
            except Exception:
                # If refresh fails (e.g. revoked), force re-login
                creds = None

        if not creds:
            if not os.path.exists(GOOGLE_API_CLIENT_SECRET_PATH):
                raise FileNotFoundError(
                    f"OAuth Client Secret not found at: {GOOGLE_API_CLIENT_SECRET_PATH}. "
                    "Please download it from Google Cloud Console (Credentials -> Create -> OAuth Client ID -> Desktop App) "
                    "and rename it to client_secret.json"
                )

            # Open browser for login
            flow = InstalledAppFlow.from_client_secrets_file(
                GOOGLE_API_CLIENT_SECRET_PATH, scopes
            )

            # Фикс: принудительно запрашиваем offline доступ для генерации долговечного refresh_token
            creds = flow.run_local_server(
                port=0, prompt="consent", access_type="offline"
            )

        # 3. Save the credentials as JSON
        with open(token_path, "w", encoding="utf-8") as token:
            token.write(creds.to_json())

    return creds
