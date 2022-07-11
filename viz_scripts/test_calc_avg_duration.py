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