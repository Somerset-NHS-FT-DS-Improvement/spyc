import numpy as np
from .helpers import get_data_subset


def x_chart(data, target_col, cl_start_dt=None, cl_end_dt=None):
    """
    Calculate control lines for X-chart.

    Inputs:
        data (Pandas.DataFrame): Input data
        target_col (str): Name of column to calculate within data
        cl_start_dt (str): Start date of control line calculation
        cl_end_dt (str): End date of control line calculation
    Returns:
        Dictionary in required format
    """

    data_subset = get_data_subset(data, cl_start_dt, cl_end_dt)

    cl = data_subset[target_col].mean()
    mr_cl = np.abs(data_subset[target_col].diff()).mean()

    d2 = 1.128  # Individual chart constant

    ucl = cl + 3 * (mr_cl / d2)
    lcl = cl - 3 * (mr_cl / d2)

    return {'process': data[target_col], 'CL': cl, "UCL": ucl, "LCL": lcl}


def mr_chart(data, target_col, cl_start_dt=None, cl_end_dt=None):
    """
    Calculate control lines for the moving-range chart.

    Inputs:
        data (Pandas.DataFrame): Input data
        target_col (str): Name of column to calculate within data
        cl_start_dt (str): Start date of control line calculation
        cl_end_dt (str): End date of control line calculation
    Returns:
        Dictionary in required format
    """

    data_subset = get_data_subset(data, cl_start_dt, cl_end_dt)

    mr_cl = np.abs(data_subset[target_col].diff()).mean()

    mr_lcl = 0
    mr_ucl = 3.27 * mr_cl

    return {'process': np.abs(data[target_col].diff()), 'CL': mr_cl, "UCL": mr_ucl, "LCL": mr_lcl}
