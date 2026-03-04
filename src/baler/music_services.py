import base64
import time

import httpx

from . import config


class SpotifyRateLimitError(Exception):
    """Raised when Spotify returns 429. Carries the Retry-After value in seconds."""

    def __init__(self, retry_after: int):
        self.retry_after = retry_after
        super().__init__(f"Spotify rate limit hit. Retry-After: {retry_after}s")


class SpotifyClient:
    """
    Handles all interactions with the Spotify Web API.
    """

    def __init__(self):
        self.client_id = config.SPOTIFY_CLIENT_ID
        self.client_secret = config.SPOTIFY_CLIENT_SECRET
        self.token_url = "https://accounts.spotify.com/api/token"
        self.api_base_url = "https://api.spotify.com/v1/"
        self.access_token = None
        self.token_expiry_time = 0

    async def _get_access_token(self):
        """
        Fetches or refreshes the Spotify API access token using the Client Credentials Flow.
        """
        if self.access_token and time.time() < self.token_expiry_time:
            return

        auth_string = f"{self.client_id}:{self.client_secret}"
        auth_b64 = base64.b64encode(auth_string.encode()).decode()

        headers = {
            "Authorization": f"Basic {auth_b64}",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {"grant_type": "client_credentials"}

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(self.token_url, headers=headers, data=data)
                response.raise_for_status()
                token_data = response.json()
                self.access_token = token_data["access_token"]
                self.token_expiry_time = (
                    time.time() + token_data.get("expires_in", 3600) - 60
                )
            except httpx.RequestError as e:
                print(f"Error requesting access token from Spotify: {e}")
                self.access_token = None
            except Exception as e:
                print(f"An unexpected error occurred during token refresh: {e}")
                self.access_token = None

    async def get_album_spotify_url(self, album_title: str, artist: str) -> str | None:
        """
        Searches for an album by a specific artist and returns its public Spotify URL.
        """
        await self._get_access_token()
        if not self.access_token:
            return None

        headers = {"Authorization": f"Bearer {self.access_token}"}

        search_query = f'album:"{album_title}" artist:"{artist}"'
        search_params = {"q": search_query, "type": "album", "limit": 1, "market": "US"}

        print(f"Searching Spotify for album with query: '{search_query}'")

        search_url = f"{self.api_base_url}search"

        async with httpx.AsyncClient() as client:
            try:
                search_response = await client.get(
                    search_url, headers=headers, params=search_params
                )
                search_response.raise_for_status()
                search_results = search_response.json()

                if not search_results.get("albums") or not search_results["albums"].get(
                    "items"
                ):
                    print(
                        f"Spotify search returned no albums for query: {search_query}"
                    )
                    return None

                album = search_results["albums"]["items"][0]
                album_url = album.get("external_urls", {}).get("spotify")

                if album_url:
                    print(f"Found album '{album['name']}' with URL: {album_url}")
                    return album_url
                else:
                    print(
                        f"Found album '{album['name']}', but it has no external Spotify URL."
                    )
                    return None

            except httpx.RequestError as e:
                print(f"Error communicating with Spotify: {e}")
                return None
            except Exception as e:
                print(f"An unexpected error occurred during Spotify search: {e}")
                return None

    async def get_album_metadata(self, album_title: str, artist: str) -> dict | None:
        """
        Fetches enrichment metadata for an album from Spotify:
        - artist_genres: list of genre strings from the artist profile
        - label: record label string from the album
        - related_artists: list of related artist name strings

        Returns None if the album cannot be found.
        """
        await self._get_access_token()
        if not self.access_token:
            return None

        headers = {"Authorization": f"Bearer {self.access_token}"}
        search_query = f'album:"{album_title}" artist:"{artist}"'
        search_params = {"q": search_query, "type": "album", "limit": 1, "market": "US"}

        def _check_rate_limit(response: httpx.Response):
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 30))
                raise SpotifyRateLimitError(retry_after)

        async with httpx.AsyncClient() as client:
            try:
                # 1. Search for the album to get album_id and artist_id
                search_response = await client.get(
                    f"{self.api_base_url}search", headers=headers, params=search_params
                )
                _check_rate_limit(search_response)
                search_response.raise_for_status()
                search_results = search_response.json()

                albums = search_results.get("albums", {}).get("items", [])
                if not albums:
                    return None

                album = albums[0]
                album_id = album.get("id")
                artist_items = album.get("artists", [])
                artist_id = artist_items[0].get("id") if artist_items else None

                if not album_id or not artist_id:
                    return None

                # 2. Fetch album details (label)
                album_response = await client.get(
                    f"{self.api_base_url}albums/{album_id}", headers=headers
                )
                _check_rate_limit(album_response)
                album_response.raise_for_status()
                album_data = album_response.json()
                label = album_data.get("label") or "N/A"

                # 3. Fetch artist details (genres)
                artist_response = await client.get(
                    f"{self.api_base_url}artists/{artist_id}", headers=headers
                )
                _check_rate_limit(artist_response)
                artist_response.raise_for_status()
                artist_data = artist_response.json()
                genres = artist_data.get("genres", [])

                # 4. Fetch related artists (404 is normal for smaller artists)
                related_response = await client.get(
                    f"{self.api_base_url}artists/{artist_id}/related-artists",
                    headers=headers,
                )
                if related_response.status_code == 404:
                    related_artists = []
                else:
                    _check_rate_limit(related_response)
                    related_response.raise_for_status()
                    related_data = related_response.json()
                    related_artists = [
                        a["name"] for a in related_data.get("artists", [])[:10]
                    ]

                return {
                    "artist_genres": genres,
                    "label": label,
                    "related_artists": related_artists,
                }

            except SpotifyRateLimitError:
                raise  # Let the caller handle rate limits
            except httpx.RequestError as e:
                print(f"Error fetching album metadata from Spotify: {e}")
                return None
            except Exception as e:
                print(f"Unexpected error fetching album metadata: {e}")
                return None
