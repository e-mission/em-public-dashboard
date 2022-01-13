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
    
    def setUp(self):
        self.constants = pd.DataFrame({
            'mode': ['car', 'bus', 'train'],
            'vals': [12,5,2],
            'test': [0,0,0],
            'energy_intensity_factor': [0, 1, 2],
            'CO2_factor': [1, 2, 3],
            '(kWH)/trip': [0.5, 0.2, 0.3]
        })
        # Inputs
        self.data = pd.DataFrame({
            'mode': ['car', 'bus', 'train', 'car'],
            'repm': ['car', 'car', 'bus', 'train'],
            'vals': [1,2,3, 4],
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
            'vals': [1,2,3, 4],
            'test': [0.5,3,0,8],
            'ei_mode': [0, 1, 2, 0],
            'CO2_mode': [1, 2, 3, 1],
            'ei_trip_mode': [0.5, 0.2, 0.3, 0.5],
            'ei_repm': [0, 0, 1, 2],
            'CO2_repm': [1, 1, 2, 3],
            'ei_trip_repm': [0.5, 0.5, 0.2, 0.3], 
        })
        output = scaffolding.energy_intensity(self.data, self.constants, '', 'mode', 'repm')
        self.assertTrue(expect.equals(output),
                            f'{output}')




if __name__ == '__main__':
    unittest.main()