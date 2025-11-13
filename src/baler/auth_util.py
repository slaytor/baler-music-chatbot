import os
import google.auth
import google.auth.transport.requests
from dotenv import load_dotenv

# Import config from our new file
from . import config


# Load .env file and set absolute path for credentials
load_dotenv(config.ENV_PATH)
if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
    absolute_cred_path = config.PROJECT_ROOT / os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(absolute_cred_path)
else:
    print("Warning: GOOGLE_APPLICATION_CREDENTIALS not set in .env file.")


def get_gcp_auth_token():
    """Generates a GCP auth token from the service account credentials."""
    try:
        credentials, project = google.auth.default(
            scopes=['https://www.googleapis.com/auth/generative-language']
        )
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)
        return credentials.token
    except Exception as e:
        print(f"Could not get GCP auth token. Is GOOGLE_APPLICATION_CREDENTIALS set? Error: {e}")
        return None
