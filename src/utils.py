from collections import OrderedDict


def normalize_url(url: str) -> str:
    from urllib.parse import urlparse, urlunparse

    if "://" in url:
        url = url.split("://", 1)[1]
    elif ":" in url:
        url = url.split(":", 1)[1]

    parts = url.split("/", 1)
    domain = parts[0].lower()
    path = parts[1] if len(parts) > 1 else ""

    if path:
        normalized = f"https://{domain}/{path}"
    else:
        normalized = f"https://{domain}"

    parsed = urlparse(normalized)
    return urlunparse(("https", parsed.netloc, parsed.path, "", "", ""))


# If dict has a value by key, then return it, otherwise insert new. Limit dict size to N items.
class LimitedDict:
    def __init__(self, max_size=5):
        self.max_size = max_size
        self.data = OrderedDict()

    # def contains(self, key):
    #     return key in self.data

    def get_or_insert(self, key, value_func):
        if key in self.data:
            return self.data[key]
        else:
            if len(self.data) >= self.max_size:
                self.data.popitem(last=False)  # Remove the oldest item

            value = value_func(key)
            self.data[key] = value
            return value


# import itertools

# def flatMap(list_of_lists):
#     return list(itertools.chain.from_iterable(list_of_lists))

# # def flatMap(l):
# #     return [item for sublist in l for item in sublist]
