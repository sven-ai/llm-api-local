# import urllib.robotparser
import asyncio
from typing import Optional
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import robots

# import requests
from playwright.async_api import async_playwright
from pydantic import BaseModel


class FetchResult(BaseModel):
    html: str
    final_url: str


def strip_tracking_params(url: str) -> str:
    tracking_params = {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_content",
        "utm_term",
        "utm_id",
        "fbclid",
        "gclid",
        "msclkid",
        "mc_cid",
        "mc_eid",
        "_hsenc",
        "_hsmi",
        "ref",
        "source",
    }

    parsed = urlparse(url)
    query_params = parse_qs(parsed.query, keep_blank_values=True)

    cleaned_params = {
        k: v for k, v in query_params.items() if k not in tracking_params
    }

    cleaned_query = urlencode(cleaned_params, doseq=True)

    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            cleaned_query,
            parsed.fragment,
        )
    )


class Fetch:
    def __init__(
        self,
        user_agent: str = "SvenBrowser/1.0 (anton@devbrain.io)",
        max_concurrent: int = 15,
    ):
        self.user_agent = user_agent
        self.max_concurrent = max_concurrent
        self._playwright = None
        self._browser = None
        self._context = None
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def _ensure_browser(self):
        async with self._lock:
            if self._browser is None:
                self._playwright = await async_playwright().start()
                self._browser = await self._playwright.chromium.launch(
                    headless=True,
                    args=[
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-gpu",
                    ],
                )
                self._context = await self._browser.new_context(
                    user_agent=self.user_agent,
                    viewport={"width": 1440, "height": 900},
                )

    async def close(self):
        if self._context:
            await self._context.close()
            self._context = None
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _check_robots_txt(self, url: str) -> bool:
        """
        Checks the domain's robots.txt to see if the URL is allowed for the configured user agent.
        Returns True if allowed, False otherwise (including if robots.txt is inaccessible).
        """
        try:
            parser = robots.RobotsParser.from_uri(url)
            result = parser.can_fetch(self.user_agent, url)

            if result:
                print(
                    f"'{self.user_agent}' is ALLOWED to fetch '{url}' according to robots.txt."
                )
                return True
            else:
                print(
                    f"'{self.user_agent}' is DISALLOWED from fetching '{url}' by robots.txt."
                )
                return False

        except Exception as e:
            print(f"An error occurred during robots.txt check: {e}")
            print(
                "Assuming disallow as robots.txt could not be accessed or parsed."
            )
            return False

    async def _fetch_html_with_playwright(
        self, url: str
    ) -> Optional[FetchResult]:
        await self._ensure_browser()

        async with self._semaphore:
            page = None
            try:
                page = await self._context.new_page()

                print(f"Navigating to {url} with Playwright...")
                await page.goto(
                    url, wait_until="domcontentloaded", timeout=20000
                )
                content = await page.content()
                final_url = strip_tracking_params(page.url)
                print(
                    f"Successfully fetched content from {url}. Final URL: {final_url}"
                )
                return FetchResult(html=content, final_url=final_url)
            except Exception as e:
                print(f"Playwright failed to fetch {url}: {e}")
                return None
            finally:
                if page:
                    await page.close()

    async def fetch(self, url: str) -> Optional[FetchResult]:
        """
        Fetches the content of a given URL after checking the domain's robots.txt.

        Args:
            url (str): The URL to fetch.

        Returns:
            Optional[FetchResult]: FetchResult with html and final_url if allowed and successful,
                                   otherwise None.
        """
        # NOTE: robots.txt check disabled - often returns incorrect NO for sites that allow fetching
        # if await self._check_robots_txt(url):
        #     return await self._fetch_html_with_playwright(url)
        # else:
        #     return None

        return await self._fetch_html_with_playwright(url)
