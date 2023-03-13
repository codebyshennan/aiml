import unittest
from src.helpers import fahrenheitToCelsius


class TestFahrenheitToCelsius(unittest.TestCase):
    def test_freezing_point(self):
        self.assertEqual(fahrenheitToCelsius(32), 0)

    def test_boiling_point(self):
        self.assertEqual(fahrenheitToCelsius(212), 100)

    def test_body_temperature(self):
        self.assertAlmostEqual(fahrenheitToCelsius(98.6), 37, delta=0.1)

    def test_arbitrary_temperature(self):
        self.assertAlmostEqual(fahrenheitToCelsius(50), 10, delta=0.1)


if __name__ == "__main__":
    unittest.main()
