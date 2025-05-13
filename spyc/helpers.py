import pandas as pd


def convert_to_timestamp(data_dict):
    # Convert all date and time values to Timestamp objects
    consistent_data = {}
    for key, value in data_dict.items():
        new_key = pd.Timestamp(key)
        new_value = {
            "cl_start_data": pd.Timestamp(value["cl_start_data"]),
            "cl_end_data": pd.Timestamp(value["cl_end_data"])
        }
        consistent_data[new_key] = new_value

    return consistent_data


def get_data_subset(data, cl_start_dt=None, cl_end_dt=None):
    # Helper function to subset data based on start and end control line dates.

    if cl_start_dt is None:
        cl_start_dt = data.index.min()
    if cl_end_dt is None:
        cl_end_dt = data.index.max()

    return data.loc[cl_start_dt:cl_end_dt]
