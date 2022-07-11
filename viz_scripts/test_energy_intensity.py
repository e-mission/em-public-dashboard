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

class TestEnergyIntensity(unittest.TestCase):
    """
    A unit test for energy_intensity function in 
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
        output = scaffolding.energy_intensity(self.data, self.constants, 'repm', 'mode')
        self.assertTrue(expect.equals(output),
                            f"energy_intensity failed:\n{output[['ei_mode','ei_repm','CO2_mode','CO2_repm','ei_trip_mode','ei_trip_repm']]}")
        
        expect = pd.DataFrame({
            'mode': ['car', 'bus', 'train', 'car'],
            'repm': ['car', 'car', 'bus', 'train'],
            'vals': [1,2,3,4],
            'test': [0.5,3.0,0.0,8.0],
            'cost__trip_mode': [1,2,3,1],
            'cost__trip_repm': [1,1,2,3],
        })
        output = scaffolding.cost(self.data, self.constants, 'repm', 'mode')
        self.assertTrue(expect.equals(output),
                            f"cost failed:\n{output}")

        expect = pd.DataFrame({
            'mode': ['car', 'bus', 'train', 'car'],
            'repm': ['car', 'car', 'bus', 'train'],
            'vals': [1,2,3,4],
            'test': [0.5,3,0,8],
            'dura__trip_mode': [3,2,1,3],
            'dura__trip_repm': [3,3,2,1],
        })
        output = scaffolding.time(self.data, self.constants, 'repm', 'mode')
        self.assertTrue(expect.equals(output),
                            f"time failed:\n{output}")


if __name__ == '__main__':
    unittest.main()
