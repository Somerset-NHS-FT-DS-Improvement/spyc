import numpy as np
import pandas as pd
import warnings

from .control_rules import _rules_func
from .seasonal_formatting import (
    _add_seasonal_column,
    _group_data_by_season,
    _data_points_per_season,
)
from .helpers import convert_to_timestamp


class SPC:

    def __init__(
        self,
        data_in,
        target_col,
        seasonal=None,
        fix_control_start_dt=None,
        fix_control_end_dt=None,
    ):
        """
        A tool for streamlining Statistical Process Control (SPC) analytics.

        Functionality:
            - Generalisable for custom/default control chart types.
            - Allows for seasonal patterns to be captured (daily/weekly/monthly/quarterly). Note, trends are not
            automatically handled and should ideally be removed prior to feeding in dataset, if present
            - Allows for control lines to be re-calculated following a process change.

        Inputs:
            - data_in (pd.DataFrame): The input data containing the process measurements, with DateTime index.
            - target_col (str): The column name in the data containing the target measurements.
            - seasonal (int, optional): The number of time steps for repeating seasonal patterns (e.g., 24 for hourly
              data, 7 for daily data). Defaults to None.
            - fix_control_start_dt (str): If None, reverts to start of data/start of previous value in
            process_change_dates.
            - fix_control_end_dt (str): If None, reverts to end of data/end of next value in process_change_dates.
        """

        self.control_line_dates = None
        self.seasonal_periods = None
        self.seasonal = None
        self.target_col = None
        self.data_in = None
        self.data_in = data_in.copy()
        self.target_col = target_col
        self.seasonal = seasonal

        # If values not set, default to start/end dates of data_in
        self.fix_control_start_dt = (
            fix_control_start_dt
            if fix_control_start_dt is not None
            else self.data_in.index.min()
        )
        self.fix_control_end_dt = (
            fix_control_end_dt
            if fix_control_end_dt is not None
            else self.data_in.index.max()
        )

        assert target_col in data_in.columns, "Column not found in input dataframe!"
        assert seasonal in [None, 24, 7, 12, 4], (
            "seasonal parameter only accepts 4 (quarterly), 12 (monthly), "
            "7 (daily) or 24 (hourly)."
        )
        assert isinstance(
            self.data_in.index, pd.DatetimeIndex
        ), "Dataframe must have datetime index"

        # Add seasonal column to data_in and store unique seasonal periods
        if self.seasonal:
            self.data_in["season"] = _add_seasonal_column(
                self.data_in, self.seasonal
            )  # Add season column
            self.seasonal_periods = self.data_in[
                "season"
            ].unique()  # Store unique seasonal values

        # Track process change dates & control line calculation start/end dates, if specified
        self.control_line_dates = {
            self.data_in.index.min(): {
                "cl_start_data": self.fix_control_start_dt,
                "cl_end_data": self.fix_control_end_dt,
                "process_names": None,
            }
        }

        self.__change_dates_list = None
        self.__seasonal_adjustment_passed = (
            {}
        )  # Storing bools for whether seasonal adjustment will be made.

    def add_process_change_date(
        self,
        change_date,
        fix_control_start_dt=None,
        fix_control_end_dt=None,
        process_name=None,
    ) -> None:
        """
        Optional Method. Allows control lines to be re-calculated following a systematic change to the measured process.
        It can be called multiple times if a series of process changes occurs during the observed period.

        Inputs:
            - change_date (str or pd.Timestamp): The date when the process change occurred.
            - fix_control_start_dt (str or pd.Timestamp, optional): The start date for the control line calculation.
              If None, it defaults to the change_date.
            - fix_control_end_dt (str or pd.Timestamp, optional): The end date for the control line calculation.
              If None, it defaults to the end of the data.
            - process_name (str, optional): A name for the process change. This can be used to identify different
              process changes.
        """

        fix_control_start_dt = fix_control_start_dt or change_date
        fix_control_end_dt = fix_control_end_dt or self.data_in.index.max()

        # Default max. of previous fix_control_end_dt to change_date, if none provided, or date exceeds change_date.
        if (
            self.fix_control_end_dt == self.data_in.index.max()
            or self.fix_control_end_dt > fix_control_start_dt
        ):
            most_recent_key = max(
                [
                    pd.Timestamp(key)
                    for key in self.control_line_dates.keys()
                    if pd.Timestamp(key) < pd.Timestamp(change_date)
                ],
                default=None,
            )
            self.control_line_dates[most_recent_key][
                "cl_end_data"
            ] = change_date  # Update end date

        # Add new change_date to dict.
        self.control_line_dates[change_date] = {
            "cl_start_data": fix_control_start_dt,
            "cl_end_data": fix_control_end_dt,
            "process_names": process_name,
        }

        self.control_line_dates = convert_to_timestamp(self.control_line_dates)

    def calculate_spc(
        self,
        spc_calc_func,
        rule_1=True,
        rule_2=False,
        rule_3=False,
        rule_4=False,
        rule_5=False,
    ) -> None:
        """
        Calculates control lines (including re-calculation, if specified) & tests data for control.

        Inputs:

            - spc_name (str): The name of the SPC chart.
            - spc_calc_func (function): A function that calculates the SPC metrics. It must return a dictionary
              containing the following keys:
                - 'process' (pd.Series): The measured process being tracked using SPC (e.g., individuals/moving-range).
                - 'CL' (float or int): The center/control line.
                - 'UCL' (float or int): The upper control line.
                - 'LCL' (float or int): The lower control line.
            - rule_1 (bool, optional): Rule 1 - Outside of control limits. Defaults to True.
            - rule_2 (bool, optional): Rule 2 - 8 (or more) consecutive points above/below the center line. Defaults to
              False.
            - rule_3 (bool, optional): Rule 3 - 6 (or more) consecutive points increasing/decreasing. Defaults to False.
            - rule_4 (bool, optional): Rule 4 - 2 out of 3 consecutive points outside +/- 2 sigma limits. Defaults to
              False.
            - rule_5 (bool, optional): Rule 5 - 15 (or more) consecutive points within the +/- 1 sigma limits. Defaults
              to False.
        """

        rules_dict = {
            "Rule 1": rule_1,
            "Rule 2": rule_2,
            "Rule 3": rule_3,
            "Rule 4": rule_4,
            "Rule 5": rule_5,
        }

        # Obtain list of datasets for each process_change_dates periods, if add_process_change() called.
        self.__change_dates_list = self.__get_change_dates()

        self.__calculate_control_lines(spc_calc_func=spc_calc_func)

        return self.__test_for_control(rules_dict=rules_dict)

    def __get_change_dates(self):
        """
        Returns list of tuples of type (start_change_date, end_change_date). If add_process_change_date() not called,
        a single element is returned with start/ end date of data_in.
        """

        data_start, data_end = self.data_in.index.min(), self.data_in.index.max()

        if len(self.control_line_dates.keys()) == 1:
            return [(data_start, data_end)]  # No change dates, return start/ end date

        change_dates = sorted(
            list(self.control_line_dates.keys())[1:]
        )  # Process change dates (excluding first val.)
        change_dates_list = []  # Store data blocks in list
        prev_date = data_start  # Set previous change date to start of data

        for cd in change_dates:
            change_dates_list.append((prev_date, cd))
            prev_date = cd  # Update prev_date to the current change date

        # Append last data block
        change_dates_list.append((cd, data_end))

        return change_dates_list

    def __calculate_control_lines(self, spc_calc_func):
        """
        Calculates control lines for each "block" dataset (only one dataset if no process_change_dates set).
        """

        cl_df = pd.DataFrame()
        cl_df.index = self.data_in.index
        cl_df[["CL", "LCL", "UCL", "process", "period", "period_name"]] = np.nan
        cl_df["period"] = (
            cl_df["period"].fillna(-1).astype(int)
        )  # Ensure int is stored rather than float
        cl_df["period_name"] = cl_df["period_name"].astype(object)
        if self.seasonal:
            cl_df["season"] = self.data_in["season"].values  # Add season column

        # Store control lines in dictionary for quick reference
        cl_dictionary = {}

        # Use slices instead of copying data frames
        for idx, date_tuple in enumerate(self.__change_dates_list):

            period_start_date, period_end_date = date_tuple

            df = self.data_in.loc[
                period_start_date:period_end_date
            ]  # Access dataframe for given period

            start_date_dict = self.control_line_dates.get(df.index[0], None)
            end_date_dict = self.control_line_dates.get(df.index[0], None)

            if start_date_dict is None or end_date_dict is None:
                raise ValueError(
                    f"Warning: Control line dates not found for change date {df.index[0]}. Make sure "
                    f"change date is of same type as index."
                )

            start_date = start_date_dict["cl_start_data"]
            end_date = end_date_dict["cl_end_data"]

            # Check if data points per season are sufficient
            check_data_points_per_season = _data_points_per_season(
                _group_data_by_season(df.loc[start_date:end_date], self.seasonal)
            )

            if self.seasonal and not check_data_points_per_season:
                self.__seasonal_adjustment_passed[idx] = True
                warnings.warn(
                    "Not enough data to build reliable estimate of mean for each season. Global mean will be "
                    "estimated until more data is added (control line calculation period is changed, or "
                    "more data collected)",
                    UserWarning,
                )

            if self.seasonal and check_data_points_per_season:
                seasonal_grouped = _group_data_by_season(df, self.seasonal)

                # Calculate control lines for each seasonal period
                cl_dict = seasonal_grouped.apply(
                    lambda group: spc_calc_func(
                        group, self.target_col, start_date, end_date
                    )
                )
                cl_dictionary[idx] = cl_dict

                # Update control lines and process columns efficiently
                cl_df.loc[df.index, "CL"] = df["season"].map(lambda s: cl_dict[s]["CL"])
                cl_df.loc[df.index, "LCL"] = df["season"].map(
                    lambda s: cl_dict[s]["LCL"]
                )
                cl_df.loc[df.index, "UCL"] = df["season"].map(
                    lambda s: cl_dict[s]["UCL"]
                )

                for sn in self.seasonal_periods:
                    cl_df.loc[
                        (cl_df["season"] == sn) & (cl_df.index.isin(df.index)),
                        "process",
                    ] = cl_dict[sn]["process"]

            else:
                cl_dict = spc_calc_func(df, self.target_col, start_date, end_date)
                cl_dictionary[idx] = cl_dict

                cl_df.loc[df.index, "CL"] = cl_dict["CL"]
                cl_df.loc[df.index, "LCL"] = cl_dict["LCL"]
                cl_df.loc[df.index, "UCL"] = cl_dict["UCL"]
                cl_df.loc[df.index, "process"] = cl_dict["process"]

            cl_df.loc[df.index, "period"] = idx

        first_key = next(iter(self.control_line_dates))  # Get the first key
        cl_df["period_name"] = cl_df.index.map(
            lambda date: (
                self.control_line_dates.get(date, {}).get("process_names", np.nan)
                if date != first_key
                else np.nan
            )
        )

        self.__cl_dict = cl_dictionary
        self.__cl_data = cl_df

    def __test_for_control(self, rules_dict):
        """
        Tests dataset for control, using the set rules.
        """

        out_of_control_dict = {
            f"{rule} violation": [] for rule, enabled in rules_dict.items() if enabled
        }

        for idx in self.__cl_data.period.unique():

            df_period_filtered = self.__cl_data[self.__cl_data.period == idx]

            if self.seasonal and not self.__seasonal_adjustment_passed.get(idx, False):

                for s in self.seasonal_periods:

                    seasonal_filtered = df_period_filtered[
                        df_period_filtered.season == s
                    ]  # Filter for each season

                    # Test control for each season
                    violations = _rules_func(
                        seasonal_filtered["process"],
                        self.__cl_dict[idx].loc[s]["CL"],
                        self.__cl_dict[idx].loc[s]["LCL"],
                        self.__cl_dict[idx].loc[s]["UCL"],
                        rules_dict,
                    )
                    for rule, timestamps in violations.items():
                        out_of_control_dict[rule].extend(timestamps)

            else:

                violations = _rules_func(
                    df_period_filtered["process"],
                    self.__cl_dict[idx]["CL"],
                    self.__cl_dict[idx]["LCL"],
                    self.__cl_dict[idx]["UCL"],
                    rules_dict,
                )

            for rule, timestamps in violations.items():
                out_of_control_dict[rule].extend(timestamps)

        # Store unique dates with "out of control" variation.
        violations_dates_set = set()
        for rule, dates in out_of_control_dict.items():
            violations_dates_set.update(dates)

        violations_dates_set = sorted(violations_dates_set)  # Ensure dates are sorted
        df = pd.DataFrame(
            0, index=violations_dates_set, columns=out_of_control_dict.keys()
        )
        for rule, dates in out_of_control_dict.items():
            df.loc[dates, rule] = 1

        merged_df = self.__cl_data.merge(
            df, how="left", left_index=True, right_index=True
        )

        merged_df[list(out_of_control_dict.keys())] = (
            merged_df[list(out_of_control_dict.keys())].fillna(0).astype(int)
        )

        return merged_df.dropna(axis=1, how="all").drop(columns=["period"])
