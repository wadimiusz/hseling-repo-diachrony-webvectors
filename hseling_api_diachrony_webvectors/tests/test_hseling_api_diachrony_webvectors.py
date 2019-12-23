import unittest

import hseling_api_diachrony_webvectors


class HSELing_API_Diachrony_webvectorsTestCase(unittest.TestCase):

    def setUp(self):
        self.app = hseling_api_diachrony_webvectors.app.test_client()

    def test_index(self):
        rv = self.app.get('/healthz')
        self.assertIn('Application Shiftry', rv.data.decode())


if __name__ == '__main__':
    unittest.main()
