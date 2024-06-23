import base64
import requests
import os

class Authenticator:
    def __init__(self, api_key: str, secret_key: str, host: str = "api.hume.ai"):
        self.api_key = api_key
        self.secret_key = secret_key
        self.host = host

    def fetch_access_token(self) -> str:
        auth_string = f"{self.api_key}:{self.secret_key}"
        encoded = base64.b64encode(auth_string.encode()).decode()

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {encoded}",
        }

        data = {
            "grant_type": "client_credentials",
        }

        response = requests.post(
            f"https://{self.host}/oauth2-cc/token", headers=headers, data=data
        )

        data = response.json()

        if "access_token" not in data:
            raise ValueError("Access token not found in response")

        return data["access_token"]

# ENV
HUME_API_KEY = "3LpGtjH4qrFPqAq6ySRYuAA6IV8zkSpxtiYHs1nqpwvK7FWY"
HUME_SECRET_KEY = "CkpfSUeJtApvpbmrFnHTqpGgSKf9bl3jndfihMYVakHkBShv1wMlLmcwsd7yalGG"

# Initialize Authenticator
authenticator = Authenticator(HUME_API_KEY, HUME_SECRET_KEY)

# Fetch the access token
def get_access_token():
    return authenticator.fetch_access_token()
