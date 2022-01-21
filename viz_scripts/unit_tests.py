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
            '(kWH)/trip': [0.5, 0.2, 0.3]
        })

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


class TestEnergyImpact(unittest.TestCase):
    """
    A unit test for energy_impact_kWH function in 
    the scaffolding.py file
    """

    def setUp(self):
        self.conditions = np.array([
            [True, False, False],
            [False,True,False],
            [False,False,True]
        ])
        self.values = np.array([
            [8, 0, 3],
            [3,5,7],
            [4,2,9]
        ])
        self.data = pd.DataFrame({
            'mode': ['car', 'bus', 'train', 'car'],
            'repm': ['car', 'car', 'bus', 'train'],
            'dist': [1.5,2.5,3.5,4.5],
            'ei_mode': [1,2,3,1],
            'ei_repm': [1,1,2,3],
            'ei_trip_mode': [7,8,9,7],
            'ei_trip_repm': [7,7,8,9],
            'Mode_confirm_fuel': ['gasoline','diesel','electric','gasoline'],
            'Replaced_mode_fuel': ['gasoline','gasoline','diesel','electric']
        })


    def test_process(self):
        expect = np.array([8, 5, 9])
        output = np.select(self.conditions, self.values)
        if(len(expect) != len(output)):
            self.assertTrue(False, 
                            f'Select Malfunction (out: {output})')
        else:
            for i in range(len(expect)):
                self.assertEqual(expect[i], output[i],
                                f'Select Malfunction (out: {output})')
    
    def test_function(self):
        expect = pd.DataFrame({
            'mode': ['car', 'bus', 'train', 'car'],
            'repm': ['car', 'car', 'bus', 'train'],
            'dist': [1.5,2.5,3.5,4.5],
            'ei_mode': [1,2,3,1],
            'ei_repm': [1,1,2,3],
            'ei_trip_mode': [7,8,9,7],
            'ei_trip_repm': [7,7,8,9],
            'Mode_confirm_fuel': ['gasoline','diesel','electric','gasoline'],
            'Replaced_mode_fuel': ['gasoline','gasoline','diesel','electric'],
            'repm_EI(kWH)':[1.5*1*0.000293071,
                            2.5*1*0.000293071,
                            3.5*2*0.000293071,
                            4.5*3+9],
            'mode_EI(kWH)':[1.5*1*0.000293071,
                            2.5*2*0.000293071,
                            3.5*3+9,
                            4.5*1*0.000293071],
            'Energy_Impact(kWH)':[round(1.5*1*0.000293071-1.5*1*0.000293071,3), 
                                  round(2.5*1*0.000293071-2.5*2*0.000293071,3),
                                  round(3.5*2*0.000293071-(3.5*3+9),3), 
                                  round(4.5*3+9-4.5*1*0.000293071,3)]
        })
        output = scaffolding.energy_impact_kWH(self.data,'dist','repm', 'mode')
        self.assertTrue(np.isclose(expect['Energy_Impact(kWH)'],
                                   output['Energy_Impact(kWH)']).all(),
                            f'Error in function')



class TestCalcAvgSpeed(unittest.TestCase):
    """
    A unit test for calc_avg_speed function in 
    the scaffolding.py file
    """

    def setUp(self):
        pd.DataFrame({
            'mode': ['car', 'bus', 'train', 'car'],
            'dist': [1,2,3,4],
            'time': []
        })


    def test_process(self):
        None


    def test_function(self):
        None



if __name__ == '__main__':
    unittest.main()