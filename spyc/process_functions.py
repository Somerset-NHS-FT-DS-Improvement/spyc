import numpy as np
from .helpers import get_data_subset
from .spc_constants import constants


def infer_subgroup_size(data):
    """
    Infers the subgroup size based on the repeated index in the data.

    Args:
        data (Pandas.DataFrame): Input data with repeated index for subgroups.

    Returns:
        int: Subgroup size.
    """
    # Count occurrences of each index value
    subgroup_sizes = data.index.value_counts()

    # Ensure all subgroups are of the same size
    if subgroup_sizes.nunique() != 1:
        raise ValueError("Subgroup sizes are inconsistent.")

    # Return the common subgroup size
    return subgroup_sizes.iloc[0]


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


def xbar_chart(data, target_col, cl_start_dt=None, cl_end_dt=None):
    """
    Calculate control lines for X-bar chart.

    Inputs:
        data (Pandas.DataFrame): Input data
        target_col (str): Name of column to calculate within data
        subgroup_size (int): Size of each subgroup (e.g., 2, 3, ... 25)
        cl_start_dt (str): Start date of control line calculation (optional)
        cl_end_dt (str): End date of control line calculation (optional)
    Returns:
        Dictionary in required format
    """
    # Ensure subgroup_size is valid
    # if subgroup_size not in range(2, 26):
    #     raise ValueError("Subgroup size must be between 2 and 25 inclusive.")

    subgroup_size = infer_subgroup_size(data[target_col])

    # Filter data by date range if provided
    if cl_start_dt or cl_end_dt:
        data_subset = data[(data.index >= cl_start_dt) & (data.index <= cl_end_dt)]

    # Calculate subgroup means and ranges
    subgroup_means = data_subset[target_col].groupby(data_subset.index).mean()
    subgroup_ranges = (
        data_subset[target_col]
        .groupby(data_subset.index)
        .apply(lambda x: x.max() - x.min())
    )

    # Calculate X-bar chart control limits
    cl = subgroup_means.mean()
    r_bar = subgroup_ranges.mean()
    a2 = constants["A2"][subgroup_size]
    ucl = cl + a2 * r_bar
    lcl = cl - a2 * r_bar

    return {
        "process": data[target_col]
        .groupby(data.index)
        .mean(),  # Process data (subgroup means)
        "CL": cl,  # Center line
        "UCL": ucl,  # Upper control limit
        "LCL": lcl,  # Lower control limit
    }
