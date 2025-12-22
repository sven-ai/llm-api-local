from collections import OrderedDict


def normalize_url(url: str) -> str:
    from urllib.parse import urlparse, urlunparse

    parsed = urlparse(url.lower())  # Lowercase everything

    # Strip port from netloc if present (only split last occurrence)
    netloc = parsed.netloc
    if ':' in netloc and '@' not in netloc.split(':')[-1]:  # Make sure last part is port, not userinfo
        netloc = netloc.rsplit(':', 1)[0]

    # Normalize path - remove duplicate slashes
    path = parsed.path
    while '//' in path:
        path = path.replace('//', '/')

    # Ensure trailing slash
    if not path.endswith('/'):
        path = path + '/' if path else '/'

    return urlunparse(('https', netloc, path, "", "", ""))


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