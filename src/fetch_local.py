# import urllib.robotparser
from typing import Optional
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

import robots
from pydantic import BaseModel

# import requests
from playwright.async_api import async_playwright


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
    def __init__(self, user_agent: str = "SvenBrowser/1.0 (anton@devbrain.io)"):
        self.user_agent = user_agent

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
        """
        Fetches the HTML content of a URL using Playwright.
        Returns FetchResult with html and final_url if successful, None otherwise.
        """
        async with async_playwright() as p:
            browser = None
            try:
                browser = await p.chromium.launch(
                    headless=True
                )
                page = await browser.new_page()
                await page.set_extra_http_headers(
                    {"User-Agent": self.user_agent}
                )

                print(f"Navigating to {url} with Playwright...")
                await page.goto(
                    url,
                    wait_until="networkidle",
                    timeout=30000
                )
                content = await page.content()
                final_url = strip_tracking_params(page.url)
                print(
                    f"Successfully fetched content from {url}. Final URL: {final_url}"
                )
                return FetchResult(
                    html=content,
                    final_url=final_url
                )
            except Exception as e:
                print(f"Playwright failed to fetch {url}: {e}")
                return None
            finally:
                if browser:
                    await browser.close()
                    print("Playwright browser closed.")

    async def fetch(self, url: str) -> Optional[FetchResult]:
        """
        Fetches the content of a given URL after checking the domain's robots.txt.

        Args:
            url (str): The URL to fetch.

        Returns:
            Optional[FetchResult]: FetchResult with html and final_url if allowed and successful,
                                   otherwise None.
        """
        if await self._check_robots_txt(url):
            return await self._fetch_html_with_playwright(url)
        else:
            return None
