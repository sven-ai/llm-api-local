import urllib.parse
# import urllib.robotparser
from typing import Optional

# import requests
from playwright.async_api import async_playwright

import robots


class Fetch:
    def __init__(self, user_agent: str = "SvenBrowser/1.0 (anton@mimecam.com)"):
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
            print("Assuming disallow as robots.txt could not be accessed or parsed.")
            return False

    async def _fetch_html_with_playwright(self, url: str) -> Optional[str]:
        """
        Fetches the HTML content of a URL using Playwright.
        Returns the content as a string if successful, None otherwise.
        """
        async with async_playwright() as p:
            browser = None
            try:
                browser = await p.chromium.launch(                    headless=True
                )  # Run in headless mode
                page = await browser.new_page()
                await page.set_extra_http_headers(
                    {"User-Agent": self.user_agent}
                )  # Set user agent for Playwright

                print(f"Navigating to {url} with Playwright...")
                await page.goto(
                    url, wait_until="domcontentloaded", timeout=20000
                )  # Wait for DOM to be loaded
                content = await page.content()  # Get the full HTML content
                print(f"Successfully fetched content from {url}.")
                return content
            except Exception as e:
                print(f"Playwright failed to fetch {url}: {e}")
                return None
            finally:
                if browser:
                    await browser.close()
                    print("Playwright browser closed.")

    async def fetch(self, url: str) -> Optional[str]:
        """
        Fetches the content of a given URL after checking the domain's robots.txt.

        Args:
            url (str): The URL to fetch.

        Returns:
            Optional[str]: The web page content as a string if allowed and successful,
                           otherwise None.
        """
        if await self._check_robots_txt(url):
            return await self._fetch_html_with_playwright(url)
        else:
            return None