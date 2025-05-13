import numpy as np
from .helpers import get_data_subset

"""
All chart function must accept exactly four input value:

    - data (Pandas.DataFrame): Input dataset with DateTime index.
    - target_col (str): Name of target column in data.
    - cl_start_dt (str): Start date of control line calculation (must be in data.index).
    - cl_end_dt (str): End date of control line calculation (must be in data.index).
    
Returns:
    Dictionary with control chart "process" (observed data), "CL", "UCL", "LCL".
    
Some charts have additional complexity (eg p-chart) which require further information (such as sample size col).
These are handled within the function (eg data must have column "sample_size") to maintain only the four inputs.
"""


def x_chart(data, target_col, cl_start_dt=None, cl_end_dt=None):
    """
    Calculate control lines for X (I)-chart.
    """

    data_subset = get_data_subset(data, cl_start_dt, cl_end_dt)

    cl = data_subset[target_col].mean()
    mr_cl = np.abs(data_subset[target_col].diff()).mean()

    d2 = 1.128  # Individual chart constant

    ucl = cl + 3 * (mr_cl / d2)
    lcl = cl - 3 * (mr_cl / d2)

    return {"process": data[target_col], "CL": cl, "UCL": ucl, "LCL": lcl}


def mr_chart(data, target_col, cl_start_dt=None, cl_end_dt=None):
    """
    Calculate control lines for the moving-range chart.
    """

    data_subset = get_data_subset(data, cl_start_dt, cl_end_dt)

    mr_cl = np.abs(data_subset[target_col].diff()).mean()

    mr_lcl = 0
    mr_ucl = 3.27 * mr_cl

    return {
        "process": np.abs(data[target_col].diff()),
        "CL": mr_cl,
        "UCL": mr_ucl,
        "LCL": mr_lcl,
    }


def p_chart(data, target_col, cl_start_dt=None, cl_end_dt=None):
    """
    Calculate control lines for P-chart (with varying control limits if sample size varies).

    data must have column named "sample_size" and the target column must be the proportion!
    """

    data_subset = get_data_subset(data, cl_start_dt, cl_end_dt)

    data_post_change = get_data_subset(data, cl_start_dt, data.index.max())

    p_bar = data_subset[target_col].mean()

    n = data_post_change["sample_size"]
    sigma_p = np.sqrt((p_bar * (1 - p_bar)) / n)

    cl = p_bar
    ucl = p_bar + 3 * sigma_p
    lcl = (p_bar - 3 * sigma_p).clip(lower=0)

    return {
        "process": data_post_change[target_col],
        "CL": cl,
        "UCL": ucl,
        "LCL": lcl,
    }
