import unittest
import os
import tempfile
import sqlite3
from datetime import datetime
from cache import init_db, update_cache, get_cached_url_data

class TestCacheFunctions(unittest.TestCase):
    def setUp(self):
        # Create a temporary database and dummy files
        self.db_fd, self.db_path = tempfile.mkstemp()
        self.html_file = tempfile.NamedTemporaryFile(delete=False)
        self.html_file.write(b"<html>Example</html>")
        self.html_file.close()

        self.text_file = tempfile.NamedTemporaryFile(delete=False)
        self.text_file.write(b"Example")
        self.text_file.close()

        self.test_url = "https://example.com/page"
        self.content_type = "text/html"
        self.url_file_path = self.html_file.name
        self.url_file_size = os.path.getsize(self.html_file.name)
        self.text_file_path = self.text_file.name
        self.text_file_size = os.path.getsize(self.text_file.name)
        self.test_hash = "abc123deadbeef"
        self.test_time = datetime.now()

        init_db(db_path=self.db_path)

    def tearDown(self):
        os.close(self.db_fd)
        os.remove(self.db_path)
        os.remove(self.html_file.name)
        os.remove(self.text_file.name)

    def test_update_cache_creates_entry(self):
        update_cache(
            cleaned_url=self.test_url,
            content_type=self.content_type,
            url_file_path=self.url_file_path,
            url_file_size=self.url_file_size,
            text_file_path=self.text_file_path,
            text_file_size=self.text_file_size,
            hash=self.test_hash,
            download_time=self.test_time,
            db_path=self.db_path
        )

        row = get_cached_url_data(self.db_path, self.test_url)

        self.assertIsNotNone(row)
        self.assertEqual(row[0], self.test_url)
        self.assertEqual(row[1], self.content_type)
        self.assertEqual(row[2], self.url_file_path)
        self.assertEqual(row[3], self.url_file_size)
        self.assertEqual(row[4], self.text_file_path)
        self.assertEqual(row[5], self.text_file_size)
        self.assertEqual(row[6], self.test_hash)
        #self.assertEqual(row[7], self.test_time)  # @TODO: Fix me later


if __name__ == "__main__":
    unittest.main()
