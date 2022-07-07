import torch
import torch.nn.functional as functional
import unittest


class TestMSELoss(unittest.TestCase):
    def test_similar(self):
        test_vector1 = torch.ones(91)
        test_vector2 = torch.ones(91)
        self.assertEqual(functional.mse_loss(test_vector1, test_vector2), 0)

    def test_different(self):
        test_vector1 = torch.ones(91)
        test_vector2 = torch.zeros(91)
        self.assertEqual(functional.mse_loss(test_vector1, test_vector2), 1)

    def test_value(self):
        test_vector1 = torch.ones(91)
        test_vector2 = torch.tensor([0.01*i for i in range(91)])
        result = 0
        for i in range(91):
            result += pow(test_vector1[i] - test_vector2[i], 2)
        result = (1/91)*result
        self.assertEqual(functional.mse_loss(test_vector1, test_vector2), result)
