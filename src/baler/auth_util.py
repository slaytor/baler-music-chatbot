import google.auth
import google.auth.transport.requests


def get_gcp_auth_token():
    """Generates a GCP auth token from the service account credentials."""
    try:
        # Using the specific scope required for the Generative Language API
        credentials, project = google.auth.default(
            scopes=['https://www.googleapis.com/auth/generative-language']
        )
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)
        return credentials.token
    except Exception as e:
        print(f"Could not get GCP auth token. Is GOOGLE_APPLICATION_CREDENTIALS set? Error: {e}")
        return None
