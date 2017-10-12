import unittest

from neurons.activations import unipolar, bipolar


class TestActivations(unittest.TestCase):
    def test_unipolar_positive_values(self):
        self.assertEqual(unipolar(5), 1)
        self.assertEqual(unipolar(1), 1)
        self.assertEqual(unipolar(2), 1)
        self.assertEqual(unipolar(20000), 1)

    def test_unipolar_negative_values(self):
        self.assertEqual(unipolar(-5), 0)
        self.assertEqual(unipolar(-1), 0)
        self.assertEqual(unipolar(-2), 0)
        self.assertEqual(unipolar(-20000), 0)

    def test_unipolar_zero_value(self):
        self.assertEqual(unipolar(0), 1)

    def test_bipolar_positive_values(self):
        self.assertEqual(bipolar(5), 1)
        self.assertEqual(bipolar(1), 1)
        self.assertEqual(bipolar(2), 1)
        self.assertEqual(bipolar(20000), 1)

    def test_bipolar_negative_values(self):
        self.assertEqual(bipolar(-5), -1)
        self.assertEqual(bipolar(-1), -1)
        self.assertEqual(bipolar(-2), -1)
        self.assertEqual(bipolar(-20000), -1)

    def test_bipolar_zero_value(self):
        self.assertEqual(bipolar(0), 1)

if __name__ == '__main__':
    unittest.main()
