import asyncio
import base64
import time

import httpx

from . import config


class LastFmClient:
    """
    Fetches artist metadata from the Last.fm API.
    - artist_genres: top user-applied tags (genres/styles)
    - related_artists: similar artists based on listening patterns
    """

    BASE_URL = "https://ws.audioscrobbler.com/2.0/"

    def __init__(self):
        self.api_key = config.LASTFM_API_KEY

    async def get_metadata(self, artist: str, album: str) -> dict:
        """
        Returns artist_genres and related_artists.
        Fetches album.getTopTags, artist.getTopTags, and artist.getSimilar in parallel.
        Uses album tags if available, falls back to artist tags.
        Never raises — returns empty lists on failure.
        """
        async with httpx.AsyncClient(timeout=15) as client:
            album_tags_task = client.get(self.BASE_URL, params={
                "method": "album.getTopTags",
                "artist": artist,
                "album": album,
                "api_key": self.api_key,
                "format": "json",
                "autocorrect": 1,
            })
            artist_tags_task = client.get(self.BASE_URL, params={
                "method": "artist.getTopTags",
                "artist": artist,
                "api_key": self.api_key,
                "format": "json",
                "autocorrect": 1,
            })
            similar_task = client.get(self.BASE_URL, params={
                "method": "artist.getSimilar",
                "artist": artist,
                "api_key": self.api_key,
                "format": "json",
                "limit": 10,
                "autocorrect": 1,
            })
            album_tags_resp, artist_tags_resp, similar_resp = await asyncio.gather(
                album_tags_task, artist_tags_task, similar_task, return_exceptions=True
            )

        genres = []
        if isinstance(album_tags_resp, httpx.Response) and album_tags_resp.status_code == 200:
            genres = [t["name"] for t in album_tags_resp.json().get("tags", {}).get("tag", [])[:10]]
        if not genres and isinstance(artist_tags_resp, httpx.Response) and artist_tags_resp.status_code == 200:
            genres = [t["name"] for t in artist_tags_resp.json().get("toptags", {}).get("tag", [])[:10]]

        related = []
        if isinstance(similar_resp, httpx.Response) and similar_resp.status_code == 200:
            related = [a["name"] for a in similar_resp.json().get("similarartists", {}).get("artist", [])[:10]]

        return {"artist_genres": genres, "related_artists": related}


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

