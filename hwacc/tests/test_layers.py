#!/bin/python3
import sys
sys.path.append("../..")
import hwacc


import unittest

class Testhwacc(unittest.TestCase):
    def test_conv_hwacc(self):
        self.assertEqual(hwacc.ConvLayer(16, 16, 1, 2, insize=28, inchannels=1, incapsules=1, kernsize=9, outsize=20, outchannels=256, outcapsules= 1).get_cycles(), 1794)
        self.assertEqual(hwacc.ConvLayer(16, 16, 16, 2, insize=28, inchannels=1, incapsules=1, kernsize=9, outsize=20, outchannels=256, outcapsules= 1).get_cycles(), 672)
        self.assertEqual(hwacc.ConvLayer(1, 1, 6, 2, insize=28, inchannels=3, incapsules=1, kernsize=9, outsize=20, outchannels=256, outcapsules= 3).get_cycles(), 63664)
        self.assertEqual(hwacc.ConvLayer(1, 2, 6, 2, insize=10, inchannels=4, incapsules=4, kernsize=25, outsize=20, outchannels=256, outcapsules= 1).get_cycles(), 433238)
        self.assertEqual(hwacc.ConvLayer(4, 5, 5, 2, insize=10, inchannels=5, incapsules=3, kernsize=25, outsize=25, outchannels=256, outcapsules= 2).get_cycles(), 249455)
        self.assertEqual(hwacc.ConvLayer(7, 1, 8, 2, insize=28, inchannels=1, incapsules=1, kernsize=25, outsize=20, outchannels=256, outcapsules= 5).get_cycles(), 114872)

    def test_primary_hwacc(self):
        self.assertEqual(hwacc.PrimaryLayer(16, 16, 1, 2, insize=20, inchannels=256, incapsules=1, kernsize=9, outsize=6, outchannels=32, outcapsules= 8).get_cycles(), 361745)
        self.assertEqual(hwacc.PrimaryLayer(16, 16, 16, 2, insize=20, inchannels=256, incapsules=1, kernsize=9, outsize=6, outchannels=32, outcapsules= 8).get_cycles(), 31265)
        self.assertEqual(hwacc.PrimaryLayer(16, 16, 1, 2, insize=28, inchannels=1, incapsules=1, kernsize=9, outsize=20, outchannels=256, outcapsules= 1).get_cycles(), 1794)
        self.assertEqual(hwacc.PrimaryLayer(16, 16, 16, 2, insize=28, inchannels=1, incapsules=1, kernsize=9, outsize=20, outchannels=256, outcapsules= 1).get_cycles(), 672)
        self.assertEqual(hwacc.PrimaryLayer(1, 1, 6, 2, insize=28, inchannels=3, incapsules=1, kernsize=9, outsize=20, outchannels=256, outcapsules= 3).get_cycles(), 63664)
        self.assertEqual(hwacc.PrimaryLayer(1, 2, 6, 2, insize=10, inchannels=4, incapsules=4, kernsize=25, outsize=20, outchannels=256, outcapsules= 1).get_cycles(), 433238)
        self.assertEqual(hwacc.PrimaryLayer(4, 5, 5, 2, insize=10, inchannels=5, incapsules=3, kernsize=25, outsize=25, outchannels=256, outcapsules= 2).get_cycles(), 249455)
        self.assertEqual(hwacc.PrimaryLayer(7, 1, 8, 2, insize=28, inchannels=1, incapsules=1, kernsize=25, outsize=20, outchannels=256, outcapsules= 5).get_cycles(), 114872)
    
    def test_class_hwacc(self):
        self.assertEqual(hwacc.ClassLayer(16, 16, 1, 2, insize=6, inchannels=32, incapsules=8, kernsize=6, outsize=1, outchannels=10, outcapsules= 16).get_cycles(), 98006)
        self.assertEqual(hwacc.ClassLayer(16, 16, 16, 2, insize=6, inchannels=32, incapsules=8, kernsize=6, outsize=1, outchannels=10, outcapsules= 16).get_cycles(), 6138)
        self.assertEqual(hwacc.ClassLayer(16, 16, 1, 2, insize=28, inchannels=1, incapsules=1, kernsize=9, outsize=20, outchannels=256, outcapsules= 1).get_cycles(), 1395)
        self.assertEqual(hwacc.ClassLayer(16, 16, 16, 2, insize=28, inchannels=1, incapsules=1, kernsize=9, outsize=20, outchannels=256, outcapsules= 1).get_cycles(), 273)
        self.assertEqual(hwacc.ClassLayer(1, 1, 6, 2, insize=28, inchannels=3, incapsules=1, kernsize=9, outsize=20, outchannels=256, outcapsules= 3).get_cycles(), 62465)
        self.assertEqual(hwacc.ClassLayer(1, 2, 6, 2, insize=10, inchannels=4, incapsules=4, kernsize=25, outsize=20, outchannels=256, outcapsules= 1).get_cycles(), 426839)
        self.assertEqual(hwacc.ClassLayer(4, 5, 5, 2, insize=10, inchannels=5, incapsules=3, kernsize=25, outsize=25, outchannels=256, outcapsules= 2).get_cycles(), 240081)
        self.assertEqual(hwacc.ClassLayer(7, 1, 8, 2, insize=28, inchannels=1, incapsules=1, kernsize=25, outsize=20, outchannels=256, outcapsules= 5).get_cycles(), 114473)


    def test_sum_update_hwacc(self):
        self.assertEqual(hwacc.SumUpdateLayer(16, 16, 1, 2, insize=6, inchannels=32, incapsules=8, kernsize=6, outsize=1, outchannels=10, outcapsules= 16).get_cycles(), 1531)
        self.assertEqual(hwacc.SumUpdateLayer(16, 16, 16, 2, insize=6, inchannels=32, incapsules=8, kernsize=6, outsize=1, outchannels=10, outcapsules= 16).get_cycles(), 1531)
        self.assertEqual(hwacc.SumUpdateLayer(16, 16, 1, 2, insize=28, inchannels=1, incapsules=1, kernsize=9, outsize=20, outchannels=256, outcapsules= 1).get_cycles(), 22033)
        self.assertEqual(hwacc.SumUpdateLayer(16, 16, 16, 2, insize=28, inchannels=1, incapsules=1, kernsize=9, outsize=20, outchannels=256, outcapsules= 1).get_cycles(), 22033)
        self.assertEqual(hwacc.SumUpdateLayer(1, 1, 6, 2, insize=28, inchannels=3, incapsules=1, kernsize=9, outsize=20, outchannels=256, outcapsules= 3).get_cycles(), 124417)
        self.assertEqual(hwacc.SumUpdateLayer(1, 2, 6, 2, insize=10, inchannels=4, incapsules=4, kernsize=25, outsize=20, outchannels=256, outcapsules= 1).get_cycles(), 160001)
        self.assertEqual(hwacc.SumUpdateLayer(4, 5, 5, 2, insize=10, inchannels=5, incapsules=3, kernsize=25, outsize=25, outchannels=256, outcapsules= 2).get_cycles(), 266671)
        self.assertEqual(hwacc.SumUpdateLayer(7, 1, 8, 2, insize=28, inchannels=1, incapsules=1, kernsize=25, outsize=20, outchannels=256, outcapsules= 5).get_cycles(), 1280001)



if __name__ == '__main__':
    unittest.main()