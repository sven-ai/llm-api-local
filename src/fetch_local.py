# import urllib.robotparser
import asyncio
import threading
from typing import Optional
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import robots

# import requests
from playwright.async_api import async_playwright
from pydantic import BaseModel


class FetchResult(BaseModel):
    html: str
    final_url: str


class _FetchInstance:
    def __init__(self, user_agent: str, max_concurrent: int):
        self.user_agent = user_agent
        self.max_concurrent = max_concurrent
        self._playwright = None
        self._browser = None
        self._request_count = 0
        self._max_requests_before_restart = 30
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrent)


_thread_local = threading.local()


def _get_instance(user_agent: str, max_concurrent: int) -> _FetchInstance:
    if not hasattr(_thread_local, "fetch_instance"):
        _thread_local.fetch_instance = _FetchInstance(user_agent, max_concurrent)
    return _thread_local.fetch_instance


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

    cleaned_params = {k: v for k, v in query_params.items() if k not in tracking_params}

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
        max_concurrent: int = 10,
    ):
        self._user_agent = user_agent
        self._max_concurrent = max_concurrent

    def _instance(self) -> _FetchInstance:
        return _get_instance(self._user_agent, self._max_concurrent)

    async def _ensure_browser(self):
        inst = self._instance()
        try:
            async with inst._lock:
                if inst._request_count >= inst._max_requests_before_restart:
                    await self._restart_browser()

                if inst._browser is None:
                    inst._playwright = await async_playwright().start()
                    inst._browser = await inst._playwright.chromium.launch(
                        headless=True,
                        args=[
                            "--no-sandbox",
                            "--disable-dev-shm-usage",
                            "--disable-gpu",
                        ],
                    )
        except RuntimeError as e:
            if "bound to a different event loop" in str(e):
                print("Event loop changed, recreating instance...")
                _thread_local.fetch_instance = _FetchInstance(
                    self._user_agent, self._max_concurrent
                )
                inst = self._instance()
                async with inst._lock:
                    if inst._request_count >= inst._max_requests_before_restart:
                        await self._restart_browser()

                    if inst._browser is None:
                        inst._playwright = await async_playwright().start()
                        inst._browser = await inst._playwright.chromium.launch(
                            headless=True,
                            args=[
                                "--no-sandbox",
                                "--disable-dev-shm-usage",
                                "--disable-gpu",
                            ],
                        )
            else:
                raise

    async def _restart_browser(self):
        inst = self._instance()
        print(
            f"🔄 Restarting browser after {inst._request_count} requests for memory cleanup..."
        )

        if inst._browser:
            await inst._browser.close()
            inst._browser = None
        if inst._playwright:
            await inst._playwright.stop()
            inst._playwright = None

        inst._request_count = 0
        print("✅ Browser restart complete")

    async def close(self):
        inst = self._instance()
        if inst._browser:
            await inst._browser.close()
            inst._browser = None
        if inst._playwright:
            await inst._playwright.stop()
            inst._playwright = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _check_robots_txt(self, url: str) -> bool:
        """
        Checks the domain's robots.txt to see if the URL is allowed for the configured user agent.
        Returns True if allowed, False otherwise (including if robots.txt is inaccessible).
        """
        inst = self._instance()
        try:
            parser = robots.RobotsParser.from_uri(url)
            result = parser.can_fetch(inst.user_agent, url)

            if result:
                print(
                    f"'{inst.user_agent}' is ALLOWED to fetch '{url}' according to robots.txt."
                )
                return True
            else:
                print(
                    f"'{inst.user_agent}' is DISALLOWED from fetching '{url}' by robots.txt."
                )
                return False

        except Exception as e:
            print(f"An error occurred during robots.txt check: {e}")
            print("Assuming disallow as robots.txt could not be accessed or parsed.")
            return False

    async def _fetch_html_with_playwright(self, url: str) -> Optional[FetchResult]:
        await self._ensure_browser()

        # Safety check: ensure browser is initialized
        inst = self._instance()
        if inst._browser is None:
            print("Browser is None after _ensure_browser(), retrying initialization...")
            await self._ensure_browser()
            inst = self._instance()
            if inst._browser is None:
                print("Failed to initialize browser")
                return None

        inst._request_count += 1

        async with inst._semaphore:
            context = None
            page = None
            try:
                context = await inst._browser.new_context(
                    user_agent=inst.user_agent,
                    viewport={"width": 1440, "height": 900},
                )
                page = await context.new_page()

                print(
                    f"Navigating to {url} with Playwright... (req #{inst._request_count})"
                )

                await page.add_init_script("""
                    const style = document.createElement('style');
                    style.textContent = `
                        *, *::before, *::after {
                            animation-duration: 0s !important;
                            animation-delay: 0s !important;
                            transition-duration: 0s !important;
                            transition-delay: 0s !important;
                        }
                    `;
                    document.head.appendChild(style);
                """)

                await page.goto(url, wait_until="networkidle", timeout=30000)
                await page.wait_for_timeout(1000)

                content = await page.content()
                final_url = strip_tracking_params(page.url)

                if not content or len(content) < 1024:
                    print(
                        f"⚠️ Playwright returned insufficient content ({len(content) if content else 0} bytes): {url}"
                    )
                    return None

                print(
                    f"Successfully fetched content from {url}. Final URL: {final_url} ({len(content)} bytes)"
                )
                return FetchResult(html=content, final_url=final_url)
            except Exception as e:
                print(f"Playwright failed to fetch {url}: {e}")
                return None
            finally:
                if page:
                    await page.close()
                if context:
                    await context.close()

    async def fetch(self, url: str) -> Optional[FetchResult]:
        return await self._fetch_html_with_playwright(url)
