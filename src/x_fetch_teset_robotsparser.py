import asyncio
import urllib.parse
from typing import Optional

import requests

# Correct import for robotsparser's RobotsParser class
from robotsparser.robotsparser import RobotsParser


async def test_robotsparser_lib(
    url: str, user_agent: str, target_path: str
) -> Optional[bool]:
    base_url = (
        urllib.parse.urlparse(url).scheme
        + "://"
        + urllib.parse.urlparse(url).netloc
    )
    robots_txt_url = urllib.parse.urljoin(base_url, "/robots.txt")

    print(f"Testing robotsparser for {base_url}")
    print("---")
    print(f"Robots.txt URL: {robots_txt_url}")
    print(f"User-agent: {user_agent}")
    print(f"Target path: {target_path}")

    try:
        response = requests.get(
            robots_txt_url, timeout=5, headers={"User-Agent": user_agent}
        )
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        robots_content = response.text

        # Use the correctly imported RobotsParser
        parser = RobotsParser()
        parser.parse(robots_content)

        allowed = parser.check_access(target_path, user_agent)
        print(f"Is '{user_agent}' allowed to fetch '{target_path}'? {allowed}")
        return allowed
    except requests.exceptions.RequestException as e:
        print(
            f"Error fetching robots.txt with requests (for robotsparser): {e}"
        )
        return None
    except Exception as e:
        print(f"Error parsing robots.txt with robotsparser: {e}")
        return None


async def main():
    my_user_agent = "SvenBrowser/1.0 (anton@mimecam.com)"

    print("### Test Case: HackingWithSwift.com ###")
    allowed_hws = await test_robotsparser_lib(
        "https://www.hackingwithswift.com/",
        my_user_agent,
        "https://www.hackingwithswift.com/articles/278/whats-new-in-swiftui-for-ios-26",
    )
    if allowed_hws is False:
        print(
            "robotsparser confirms DISALLOWED for HackingWithSwift.com articles."
        )
    elif allowed_hws is True:
        print(
            "robotsparser allows HackingWithSwift.com articles (unexpected, review parsing)."
        )
    else:
        print("robotsparser encountered an error for HackingWithSwift.com.")

    print("### Test Case: Example.com ###")
    allowed_example = await test_robotsparser_lib(
        "https://www.example.com/",
        my_user_agent,
        "https://www.example.com/some/path.html",
    )
    if allowed_example is True:
        print("robotsparser confirms ALLOWED for Example.com.")
    elif allowed_example is False:
        print("robotsparser disallows Example.com (unexpected).")
    else:
        print("robotsparser encountered an error for Example.com.")


if __name__ == "__main__":
    asyncio.run(main())
