import numpy as np
import pandas as pd
import pytest

from spychart.spc_functions import mr_chart, p_chart, x_chart


@pytest.fixture
def df():
    """
    Example dataset.
    """

    data = {
        "date": pd.date_range(start="2023-01-01", periods=100, freq="D"),
        "measurement": np.random.randn(100),
    }
    df = pd.DataFrame(data)
    df.set_index("date", inplace=True)
    return df


def test_x_chart(df):
    """
    Test keys exist in output dictionary.
    """

    cl_start_dt = "2023-01-01"
    cl_end_dt = "2023-01-31"

    result = x_chart(
        df, target_col="measurement", cl_start_dt=cl_start_dt, cl_end_dt=cl_end_dt
    )

    expected_keys = ["process", "CL", "UCL", "LCL"]

    assert all([key in result for key in expected_keys])


def test_mr_chart(df):
    """
    Test keys exist in output dictionary.
    """

    cl_start_dt = "2023-01-01"
    cl_end_dt = "2023-01-31"

    result = mr_chart(
        df, target_col="measurement", cl_start_dt=cl_start_dt, cl_end_dt=cl_end_dt
    )

    expected_keys = ["process", "CL", "UCL", "LCL"]

    assert all([key in result for key in expected_keys])


def test_p_chart(df):
    """
    Test keys exist in output dictionary.
    """

    df["sample_size"] = np.random.randint(50, 100, size=100)  # Add sample size col

    cl_start_dt = "2023-01-01"
    cl_end_dt = "2023-01-31"

    result = p_chart(
        df, target_col="measurement", cl_start_dt=cl_start_dt, cl_end_dt=cl_end_dt
    )

    expected_keys = ["process", "CL", "UCL", "LCL"]

    assert all([key in result for key in expected_keys])
