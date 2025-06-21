import asyncio

from loader import load_module

fetcher = load_module("fetch.yml")


async def main():
    print()
    print("--- Testing Google (likely disallowed for search pages) ---")
    google_content = await fetcher.fetch("https://www.google.com/search?q=test")
    if google_content:
        print(f"Google content length: {len(google_content)}")
    else:
        print("Failed to fetch Google search page (likely robots.txt blocked).")

    print()
    print("--- Testing Example.com (likely allowed) ---")
    example_content = await fetcher.fetch("https://www.example.com")
    if example_content:
        print(f"Example.com content length: {len(example_content)}")
        # print(example_content[:500]) # Print first 500 characters
    else:
        print("Failed to fetch example.com.")

    print()
    print("--- Testing Wikipedia Main Page (likely allowed) ---")
    wiki_content = await fetcher.fetch(
        "https://en.wikipedia.org/wiki/Main_Page"
    )
    if wiki_content:
        print(f"Wikipedia content length: {len(wiki_content)}")
    else:
        print("Failed to fetch Wikipedia main page.")


if __name__ == "__main__":
    asyncio.run(main())
