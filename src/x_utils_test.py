import unittest
from utils import normalize_url


class TestNormalizeUrl(unittest.TestCase):
    def test_normal_url(self):
        url = "https://example.com/path/to/page"
        result = normalize_url(url)
        self.assertEqual(result, "https://example.com/path/to/page")

    def test_malformed_url_missing_double_slash(self):
        url = "https:www.example.com/path/to/page"
        result = normalize_url(url)
        self.assertEqual(result, "https://www.example.com/path/to/page")

    def test_url_with_query_params(self):
        url = "https://example.com/path?arg=value&foo=bar"
        result = normalize_url(url)
        self.assertEqual(result, "https://example.com/path")

    def test_url_with_fragment(self):
        url = "https://example.com/path#section"
        result = normalize_url(url)
        self.assertEqual(result, "https://example.com/path")

    def test_url_with_query_and_fragment(self):
        url = "https://example.com/path?arg=value#section"
        result = normalize_url(url)
        self.assertEqual(result, "https://example.com/path")

    def test_url_uppercase(self):
        url = "https://EXAMPLE.COM/PATH/To/PAGE"
        result = normalize_url(url)
        self.assertEqual(result, "https://example.com/PATH/To/PAGE")

    def test_malformed_with_query(self):
        url = "https:www.example.com/path?arg=value"
        result = normalize_url(url)
        self.assertEqual(result, "https://www.example.com/path")

    def test_root_path(self):
        url = "https://example.com"
        result = normalize_url(url)
        self.assertEqual(result, "https://example.com")

    def test_root_path_trailing_slash(self):
        url = "https://example.com/"
        result = normalize_url(url)
        self.assertEqual(result, "https://example.com")

    def test_url_with_port(self):
        url = "https://example.com:443/path"
        result = normalize_url(url)
        self.assertEqual(result, "https://example.com:443/path")

    def test_url_with_auth(self):
        url = "https://user:pass@example.com/path"
        result = normalize_url(url)
        self.assertEqual(result, "https://user:pass@example.com/path")

    def test_malformed_with_colon_only(self):
        url = "https:example.com/path"
        result = normalize_url(url)
        self.assertEqual(result, "https://example.com/path")

    def test_html_file_url(self):
        url = "https://example.com/path/to/file.html"
        result = normalize_url(url)
        self.assertEqual(result, "https://example.com/path/to/file.html")

    def test_malformed_html_file_url(self):
        url = "https:example.com/path/to/file.html"
        result = normalize_url(url)
        self.assertEqual(result, "https://example.com/path/to/file.html")


if __name__ == "__main__":
    unittest.main()
