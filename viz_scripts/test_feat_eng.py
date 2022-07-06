"""
Author: Stanley Y
Purpose:
    To test functions in scaffolding

Credit to:
https://docs.python.org/3.10/library/unittest.html
"""

import unittest
import pandas as pd
import numpy as np
import scaffolding

class TestFeatEng(unittest.TestCase):
    """
    A unit test for feat_eng function in 
    the scaffolding.py file
    """

    def setUp(self):
        self.constants = pd.DataFrame({
            'mode': ['car', 'bus', 'train'],
            'vals': [12,5,2],
            'test': [0,0,0],
            'energy_intensity_factor': [0, 1, 2],
            'CO2_factor': [1, 2, 3],
            '(kWH)/trip': [0.5, 0.2, 0.3],
            'C($/PMT)': [1,2,3],
            'D(hours/PMT)': [3,2,1]
        })

        self.data = pd.DataFrame({
            'mode': ['car', 'bus', 'train', 'car'],
            'repm': ['car', 'car', 'bus', 'train'],
            'vals': [1,2,3,4],
            'test': [0.5,3,0,8]
        })


    def test_process(self):
        expect = [('car', 12), ('bus', 5), ('train', 2)]
        zipped = zip(self.constants['mode'], self.constants['vals'])
        listed = list(zipped)
        self.assertEqual(expect, listed,
                            'Zip malfunction')

        expect = {
            'car': 12, 
            'bus': 5, 
            'train': 2
        }
        zipped = zip(self.constants['mode'], self.constants['vals'])
        a_dict = dict(zipped)
        self.assertEqual(expect, a_dict,
                            'Dict malfunction')
    
        expect = pd.Series(
            [12, 12, 5, 2]
        )
        a_dict = dict(zip(self.constants['mode'], self.constants['vals']))
        output = self.data['repm'].map(a_dict)
        self.assertTrue(expect.equals(output),
                            'Map malfunction')


    def test_function(self):
        expect = pd.DataFrame({
            'mode': ['car', 'bus', 'train', 'car'],
            'repm': ['car', 'car', 'bus', 'train'],
            'vals': [1,2,3,4],
            'test': [0.5,3,0,8],
            'ei_mode': [0, 1, 2, 0],
            'ei_repm': [0, 0, 1, 2],
            'CO2_mode': [1, 2, 3, 1],
            'CO2_repm': [1, 1, 2, 3],
            'ei_trip_mode': [0.5, 0.2, 0.3, 0.5],
            'ei_trip_repm': [0.5, 0.5, 0.2, 0.3], 
        })

        output = scaffolding.feat_eng(
            self.data,
            self.constants,
            ['energy_intensity_factor', 'CO2_factor', '(kWH)/trip'],
            ['ei_', 'CO2_', 'ei_trip_'],
            'mode',
            'repm')
        self.assertTrue(expect.equals(output),
                            f"feat_eng failed:\n{output[['ei_mode','ei_repm','CO2_mode','CO2_repm','ei_trip_mode','ei_trip_repm']]}")


if __name__ == '__main__':
    unittest.main()
