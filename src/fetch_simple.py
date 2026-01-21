import asyncio
import aiohttp
from typing import Optional
from pydantic import BaseModel

from fetch_local import strip_tracking_params


class FetchResult(BaseModel):
    html: str
    final_url: str


class FetchSimple:
    """
    Lightweight HTTP fetch using aiohttp (no JS execution).
    Faster than Playwright (10s timeout vs 30s), but can't execute JavaScript.
    """

    def __init__(
        self,
        user_agent: str = "SvenBrowser/1.0 (anton@devbrain.io)",
        max_concurrent: int = 15,
    ):
        self.user_agent = user_agent
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch(
        self,
        url: str,
        timeout: int = 10,
    ) -> Optional[FetchResult]:
        """
        Fetch HTML using aiohttp.

        Args:
            url: URL to fetch
            timeout: Request timeout in seconds (default: 10s)

        Returns:
            FetchResult with html and final_url, or None if failed
        """
        print(f"📖 Starting simple fetch: {url}")

        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as session:
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        print(
                            f"✗ Simple fetch failed with status {response.status}: {url}"
                        )
                        return None

                    html = await response.text()
                    final_url = strip_tracking_params(str(response.url))

                    print(
                        f"✓ Simple fetch succeeded: {url} → {final_url} "
                        f"({len(html)} bytes, {timeout}s timeout)"
                    )
                    return FetchResult(html=html, final_url=final_url)

        except asyncio.TimeoutError:
            print(f"✗ Simple fetch timeout after {timeout}s: {url}")
            return None
        except Exception as e:
            print(f"✗ Simple fetch exception: {url} - {str(e)}")
            return None
