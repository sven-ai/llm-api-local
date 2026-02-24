import base64
import json
import os
import random
from typing import Optional

import aiohttp
from pydantic import BaseModel


class FetchResult(BaseModel):
    html: str
    final_url: str


class FetchFailedError(Exception):
    pass


class Fetch:
    """
    Fetch module that uses the RCWwwApiApp's /fetch endpoint.
    The RCWwwApiApp uses SwiftHeadlessWebKit (WKZombie) for Playwright-like fetching.

    The /fetch endpoint returns:
    - 200: JSON with {"data": <base64-encoded-html>, "final_url": <redirect-url>}
    - 204: Fetch in progress (first request, not cached)
    - 422: Fetch failed (raises FetchFailedError)
    """

    OCAGENT_HOST = os.getenv("OCAGENT_HOST", "getsven.com")
    OCAGENT_TOKEN = os.getenv("OCAGENT_TOKEN")
    if not OCAGENT_TOKEN or not OCAGENT_TOKEN.strip():
        raise RuntimeError(
            "OCAGENT_TOKEN environment variable must be a valid non-empty string"
        )

    _ZONES = ["eu", "us"]
    _current_zone_index = 0

    def __init__(self):
        self.api_url = f"https://{self.OCAGENT_HOST}/fetch"

    def _get_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Authorization": f"Bearer {self.OCAGENT_TOKEN}",
        }

    def _get_next_zone(self) -> str:
        zone = self._ZONES[self._current_zone_index]
        self._current_zone_index = (self._current_zone_index + 1) % len(self._ZONES)
        return zone

    async def fetch(self, url: str) -> Optional[FetchResult]:
        """
        Fetch HTML using the RCWwwApiApp's /fetch endpoint.
        Rotates through zones (eu/us/sg) similar to ocagent_client.

        Args:
            url: URL to fetch (already normalized and tracking params stripped)

        Returns:
            FetchResult with html and final_url, or None if fetching (204)

        Raises:
            FetchFailedError: If fetch failed (422 or other non-200/204 status)
        """
        zone = self._get_next_zone()
        api_url = f"https://{zone}.{self.OCAGENT_HOST}/fetch"

        try:
            headers = self._get_headers()
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    api_url,
                    json={"url": url},
                    headers=headers,
                ) as response:
                    status = response.status

                    if status == 200:
                        data = await response.json()
                        html_b64 = data.get("data")
                        final_url = data.get("final_url", url)
                        if not html_b64:
                            raise FetchFailedError(f"Fetch missing data field: {url}")
                        html = base64.b64decode(html_b64).decode("utf-8")
                        return FetchResult(html=html, final_url=final_url)
                    elif status == 204:
                        return None
                    elif status == 422:
                        raise FetchFailedError(f"Fetch rejected (422): {url}")
                    else:
                        raise FetchFailedError(
                            f"Fetch unexpected status {status}: {url}"
                        )

        except FetchFailedError:
            raise
        except Exception as e:
            raise FetchFailedError(f"Fetch exception: {url} - {str(e)}")
