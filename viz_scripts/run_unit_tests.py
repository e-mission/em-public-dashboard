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
from viz_scripts import scaffolding

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


class TestCalcAvgDura(unittest.TestCase):
    """
    A unit test for calc_avg_dura function in 
    the scaffolding.py file
    """

    def setUp(self):
        self.data = pd.DataFrame({
            'mode': ['car', 'bus', 'train', 'car'],
            'dist': [1,2,3,4],
            'time': [1,2,3,4]
        })


    def test_process(self):
        # Average speed of each trip
        expect = pd.Series([1.0,1.0,1.0,1.0])
        speeds = self.data['dist'] / self.data['time']
        self.assertTrue(expect.equals(speeds),
                        f'Calc speed failed.\n{expect}\n{speeds}')
        
        # Aggregate by mode
        self.data['sped'] = self.data['dist'] / self.data['time']
        expect = pd.Series({
            'bus': 1.0,
            'car': 1.0,
            'train': 1.0
        })
        groupd = self.data.groupby('mode')
        speedm = groupd['sped'].mean()
        self.assertTrue(expect.equals(speedm),
                        f'Agg by mean failed.\n{expect}\n{speedm}')
        

        speedm = groupd['sped'].median()
        self.assertTrue(expect.equals(speedm),
                        f'Agg by median failed.\n{expect}\n{speedm}')


        None


    def test_function(self):
        expect1 = pd.DataFrame({
            'mode': ['car', 'bus', 'train', 'car'],
            'dist': [1,2,3,4],
            'time': [1,2,3,4],
            'D(time/PMT)': [1.0, 1.0, 1.0, 1.0]
        })
        expect2 = pd.Series(
            data = [1.0, 1.0, 1.0],
            index = ['bus', 'car', 'train'],
            name = 'D(time/PMT)',
            dtype=np.float64
        )
        result1, result2 = scaffolding.calc_avg_dura(self.data,'dist','time','mode','average')
        self.assertTrue(expect1.equals(result1),
                        f'calc_avg_dura with average failed.[1]\n{result1}')
        self.assertTrue(expect2.equals(result2),
                        f'calc_avg_dura with average failed.[2]\n{expect2}\n{result2}')

        result1, result2 = scaffolding.calc_avg_dura(self.data,'dist','time','mode','median')
        self.assertTrue(expect1.equals(result1),
                        f'calc_avg_dura with median failed.[1]')
        self.assertTrue(expect2.equals(result2),
                        f'calc_avg_dura with median failed.[2]')

        expect2 = None
        result1, result2 = scaffolding.calc_avg_dura(self.data,'dist','time','mode','break')
        self.assertTrue(expect1.equals(result1),
                        f'calc_avg_dura with incorrect method failed.[1]')
        self.assertEqual(expect2, result2,
                          f'calc_avg_dura with incorrect method failed.[2]')


if __name__ == '__main__':
    unittest.main()