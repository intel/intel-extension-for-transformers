from utils.english_normalizer import EnglishNormalizer
import unittest

class TestTTS(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.normalizer = EnglishNormalizer()

    @classmethod
    def tearDownClass(self):
        pass

    def test_correct_number(self):
        text = "3000 people among 1.2 billion people"
        result = self.normalizer.correct_number(text)
        self.assertEqual(result, "three thousand people among one point two billion people")

    def test_correct_abbreviation(self):
        text = "SATG AIA a great department"
        result = self.normalizer.correct_abbreviation(text)
        self.assertEqual(result, "ess Eigh tee jee Eigh I Eigh a great department")

if __name__ == "__main__":
    unittest.main()
