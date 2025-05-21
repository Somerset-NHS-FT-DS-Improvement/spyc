import numpy as np
import pandas as pd
import pytest

from spychart.helpers import convert_to_timestamp, get_data_subset


def test_convert_to_timestamp():
    """
    Converting dict. of dates to pd.Timestep test
    """

    input_dict = {
        "2023-01-01": {"cl_start_data": "2023-01-01", "cl_end_data": "2023-01-03"}
    }
    expected_output = {
        pd.Timestamp("2023-01-01"): {
            "cl_start_data": pd.Timestamp("2023-01-01"),
            "cl_end_data": pd.Timestamp("2023-01-03"),
        }
    }

    result = convert_to_timestamp(input_dict)

    assert result == expected_output


def test_get_data_subset():
    """
    Subset a sample dataframe for testing.
    """

    data = {
        "date": pd.date_range(start="2023-01-01", periods=100, freq="D"),
        "measurement": np.random.randn(100),
    }
    df = pd.DataFrame(data)
    df.set_index("date", inplace=True)

    # Test get_data_subset function
    subset = get_data_subset(df, cl_start_dt="2023-01-10", cl_end_dt="2023-01-20")

    expected_subset = df.loc["2023-01-10":"2023-01-20"]

    pd.testing.assert_frame_equal(subset, expected_subset)
