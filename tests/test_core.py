import pandas as pd
import numpy as np
import pytest
from spyc.helpers import convert_to_timestamp, get_data_subset
from spyc import SPC

def test_spc_initialisation():

    
    data = {
        'date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
        'measurement': np.random.randn(100)
    }
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    
    spc = SPC(data_in=df, target_col='measurement')
    
    assert spc.data_in.equals(df)
    assert spc.target_col == 'measurement'
    assert spc.fix_control_start_dt == df.index.min()
    assert spc.fix_control_end_dt == df.index.max()
    assert spc.seasonal == False
    assert spc.seasonal_periods is None
    assert spc.control_line_dates_dict == {
        df.index.min(): {
            "cl_start_data": df.index.min(),
            "cl_end_data": df.index.max()
        }
    }

def test_add_process_change_date():

    
    data = {
        'date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
        'measurement': np.random.randn(100)
    }
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    
    spc = SPC(data_in=df, target_col='measurement')
    
    spc.add_process_change_date(change_date='2023-02-01')
    
    expected_dict = {
        pd.Timestamp('2023-01-01'): {
            'cl_start_data': pd.Timestamp('2023-01-01'),
            'cl_end_data': pd.Timestamp('2023-02-01')
        },
        pd.Timestamp('2023-02-01'): {
            'cl_start_data': pd.Timestamp('2023-02-01'),
            'cl_end_data': pd.Timestamp('2023-04-10')
        }
    }
    
    assert spc.control_line_dates_dict == expected_dict

def test_add_seasonality():

    data = {
        'date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
        'measurement': np.random.randn(100)
    }
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    
    spc = SPC(data_in=df, target_col='measurement')
    
    def season_func(index):
        return index.month
    
    spc.add_seasonality(season_func=season_func)
    
    assert spc.seasonal == True
    assert spc.seasonal_periods.tolist() == [1, 2, 3, 4]

def test_calculate_spc():

    data = {
        'date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
        'measurement': np.random.randn(100)
    }
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    
    spc = SPC(data_in=df, target_col='measurement')
    
    def dummy_spc_calc_func(data, target_col, cl_start_dt, cl_end_dt):
        return {
            'process': data[target_col],
            'CL': data[target_col].mean(),
            'UCL': data[target_col].mean() + 3 * data[target_col].std(),
            'LCL': data[target_col].mean() - 3 * data[target_col].std()
        }
    
    result_df = spc.calculate_spc(spc_calc_func=dummy_spc_calc_func)
    
    expected_columns = ['CL', 'LCL', 'UCL', 'process']
    
    assert all([col in result_df.columns for col in expected_columns])
