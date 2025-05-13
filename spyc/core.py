import numpy as np
import pandas as pd
import warnings

from .control_rules import _rules_func
from .helpers import convert_to_timestamp


class SPC:

    def __init__(
        self,
        data_in: pd.DataFrame,
        target_col: str,
        fix_control_start_dt: str | None = None,
        fix_control_end_dt: str | None = None,
    ):
        """
        A tool for streamlining Statistical Process Control (SPC) analytics.

        Functionality:
            - Re-calculates control lines following process changes. The control lines can be calculated
              on specific periods (to avoid training on non-representitive data).
            - Generalisable for custom or default control chart types, via custom functions.
            - Captures seasonal patterns (e.g., daily, weekly, monthly, or custom, such as weekend) to negate the need
              of separate charts for each seasonal period. Note: Trends are not automatically handled and should ideally
              be removed prior to feeding in the dataset, if present.

        Inputs:
            - data_in (pd.DataFrame): The input data containing the process measurements (with DateTimeIndex).
            - target_col (str): The column name in the data containing the target measurements.
            - fix_control_start_dt (str, optional): If None, reverts to the start of data (defaults to None).
            - fix_control_end_dt (str, optional): If None, reverts to the end of data (defaults to None).

        Attributes:
            - seasonal (Bool): Whether seasonal patterns are used (set to True if add_seasonality called).
            - seasonal_periods (list of int, float, or bool): List of unique seasonal periods (if seasonal).
            - control_line_dates_dict (dict): Store dates of process changes (if add_process_change_date called).
        """

        assert isinstance(data_in, pd.DataFrame), "Input is not a pandas DataFrame."
        assert target_col in data_in.columns, "Column not found in input dataframe."
        assert isinstance(
            data_in.index, pd.DatetimeIndex
        ), "Dataframe must have DateTime Index"

        self.data_in = data_in.copy()
        self.target_col = target_col

        # Defaults to start/end dates of data_in if control start/end values not set.
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

        self.seasonal = False
        self.seasonal_periods = None
        self.control_line_dates_dict = {
            self.data_in.index.min(): {
                "cl_start_data": self.fix_control_start_dt,
                "cl_end_data": self.fix_control_end_dt,
            }
        }
        self.__change_dates_list = None
        self.__seasonal_adjustment_passed = {}

    def add_process_change_date(
        self,
        change_date: str,
        fix_control_start_dt: str | None = None,
        fix_control_end_dt: str | None = None,
    ) -> None:
        """
        Optional Method. Allows control lines to be re-calculated following a systematic change to the measured process.

        Inputs:
            - change_date (str or pd.Timestamp): The date when the process change occurred.
            - fix_control_start_dt (str or pd.Timestamp, optional): The start date for the control line calculation.
              If None, it defaults to the change_date.
            - fix_control_end_dt (str or pd.Timestamp, optional): The end date for the control line calculation.
              If None, it defaults to the end of the data.
        """

        #  Check change_date are within data range.
        assert (
            pd.Timestamp(change_date) < self.data_in.index.max()
        ), "Change date occurs after dataset!"
        assert (
            pd.Timestamp(change_date) > self.data_in.index.min()
        ), "Change date occurs before dataset!"

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
                    for key in self.control_line_dates_dict.keys()
                    if pd.Timestamp(key) < pd.Timestamp(change_date)
                ]
            )

            self.control_line_dates_dict[most_recent_key][
                "cl_end_data"
            ] = change_date  # Update end date

        # Add new change_date to dict.
        self.control_line_dates_dict[change_date] = {
            "cl_start_data": fix_control_start_dt,
            "cl_end_data": fix_control_end_dt,
        }

        self.control_line_dates_dict = convert_to_timestamp(
            self.control_line_dates_dict
        )

    def add_seasonality(self, season_func: callable, min_data_req: int = 15):
        """
        Optional Method. Include seasonal variation in control line calculations.

        Inputs:
            - season_func (function): A function that accepts DateTime Index and returns unique value
              for each seasonal period.
        """

        self.seasonal_func = season_func
        self.min_data_req = min_data_req

        # Add seasonal column to data_in and store unique seasonal periods
        self.data_in["season"] = season_func(self.data_in.index)

        self.seasonal_periods = self.data_in["season"].unique()

        self.seasonal = True

    def calculate_spc(
        self,
        spc_calc_func: callable,
        rule_1: bool = True,
        rule_2: bool = False,
        rule_3: bool = False,
        rule_4: bool = False,
        rule_5: bool = False,
    ) -> pd.DataFrame:
        """
        Convenience function for calculating control lines (including re-calculation, if specified) and
        tests data for control (up to 5 tests, defined in control_rules.py).

        Not all rules are appropriate for all charts.

        Inputs:
            - spc_calc_func (function): A function that calculates the SPC metrics. It must return a dictionary
              containing the following keys:
                - 'process' (pd.Series): The measured process being tracked using SPC (e.g., moving-range).
                - 'CL' (float, int or pd.Series): The center/control line.
                - 'UCL' (float, int or pd.Series): The upper control line.
                - 'LCL' (float, int or pd.Series): The lower control line.
            - rule_1 (bool, optional): Rule 1 - Outside of control limits. Defaults to True.
            - rule_2 (bool, optional): Rule 2 - 8 (or more) consecutive points above/below the center line. Defaults to
              False.
            - rule_3 (bool, optional): Rule 3 - 6 (or more) consecutive points increasing/decreasing. Defaults to False.
            - rule_4 (bool, optional): Rule 4 - 2 out of 3 consecutive points outside +/- 2 sigma limits. Defaults to
              False.
            - rule_5 (bool, optional): Rule 5 - 15 (or more) consecutive points within the +/- 1 sigma limits. Defaults
              to False.

          Returns:
              - SPC data (pd.DataFrame): A time series dataframe containing the control lines, process, process change
                flag (if present), season category (if present) and rule violations binary flag columns.
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

    def __get_change_dates(self) -> list[tuple]:
        """
        Returns list of tuples of type (start_change_date, end_change_date). If add_process_change_date() not called,
        a single element is returned with start/ end date of data_in.
        """

        data_start, data_end = self.data_in.index.min(), self.data_in.index.max()

        if len(self.control_line_dates_dict.keys()) == 1:
            return [(data_start, data_end)]  # No change dates, return start/ end date

        change_dates = sorted(
            list(self.control_line_dates_dict.keys())[1:]
        )  # Process change dates (excluding first val.)
        change_dates_list = []  # Store data blocks in list
        prev_date = data_start  # Set previous change date to start of data

        for cd in change_dates:
            change_dates_list.append((prev_date, cd))
            prev_date = cd  # Update prev_date to the current change date

        change_dates_list.append((cd, data_end))

        return change_dates_list

    def __data_points_per_season_check(self, df, start_date, end_date) -> bool:
        """
        Checks sufficient data to estimate control line calculation.
        """

        data_points_per_season = (
            df.loc[start_date:end_date]
            .groupby(by="season")
            .size()
            .reindex(self.seasonal_periods, fill_value=0)
        )
        data_above_threshold_bool = (data_points_per_season >= self.min_data_req).all()

        if not data_above_threshold_bool:
            self.__seasonal_adjustment_passed[idx] = True
            warnings.warn(
                f"Not enough data to build reliable estimate of mean for each season following {start_date} change period. Global mean will be "
                "estimated until more data is added (control line calculation period is changed, or "
                "more data collected)",
                UserWarning,
            )

        return data_above_threshold_bool

    def __calculate_control_lines(self, spc_calc_func: callable) -> None:
        """
        Calculates control lines for each period (single dataset if no process_change_dates set).
        """

        cl_df = pd.DataFrame()

        # Removing dupliate dates if calculating subgroup charts
        cl_df.index = self.data_in.index
        cl_df[["CL", "LCL", "UCL", "process", "period"]] = np.nan
        cl_df["period"] = (
            cl_df["period"].fillna(-1).astype(int)
        )  # Ensure int is stored rather than float
        if self.seasonal:
            cl_df["season"] = self.seasonal_func(cl_df.index)

        # Control lines stored in dictionary
        cl_dictionary = {}

        for idx, date_tuple in enumerate(self.__change_dates_list):

            period_start_date, period_end_date = date_tuple

            df = self.data_in.loc[
                period_start_date:period_end_date
            ]  # Access dataframe for given period

            # Period control lines calculated on
            cl_date_dict = self.control_line_dates_dict.get(df.index[0], None)

            if cl_date_dict is None:
                raise ValueError(
                    f"Warning: Control line dates not found for change date {df.index[0]}. Make sure "
                    f"change date is of same type as index."
                )

            start_date = cl_date_dict["cl_start_data"]
            end_date = cl_date_dict["cl_end_data"]

            # Check number of data points per season exceeds min_data_req.
            if self.seasonal:
                data_above_threshold_bool = self.__data_points_per_season_check(
                    df, start_date, end_date
                )

            # Defining dictionary of required kwargs for spc_calc_func.
            kwargs = {
                "target_col": self.target_col,
                "cl_start_dt": start_date,
                "cl_end_dt": end_date,
            }

            if self.seasonal and data_above_threshold_bool:  # Season variation applied
                seasonal_grouped = df.groupby(by="season")

                cl_dict = seasonal_grouped.apply(
                    lambda group: spc_calc_func(data=group, **kwargs)
                )

                cl_dictionary[idx] = cl_dict

                for sn in self.seasonal_periods:

                    cl_df.loc[
                        (cl_df["season"] == sn) & (cl_df.index.isin(df.index)),
                        "process",
                    ] = cl_dict[sn]["process"]
                    cl_df.loc[
                        (cl_df["season"] == sn) & (cl_df.index.isin(df.index)), "CL"
                    ] = cl_dict[sn]["CL"]
                    cl_df.loc[
                        (cl_df["season"] == sn) & (cl_df.index.isin(df.index)), "LCL"
                    ] = cl_dict[sn]["LCL"]
                    cl_df.loc[
                        (cl_df["season"] == sn) & (cl_df.index.isin(df.index)), "UCL"
                    ] = cl_dict[sn]["UCL"]

            else:  # No seasonality
                cl_dict = spc_calc_func(data=df, **kwargs)

                cl_dictionary[idx] = cl_dict

                cl_df.loc[df.index, "CL"] = cl_dict["CL"]
                cl_df.loc[df.index, "LCL"] = cl_dict["LCL"]
                cl_df.loc[df.index, "UCL"] = cl_dict["UCL"]
                cl_df.loc[df.index, "process"] = cl_dict["process"]

            cl_df.loc[df.index, "period"] = idx

        self.__cl_dict = cl_dictionary
        self.__cl_data = cl_df

    def __test_for_control(self, rules_dict: dict[str, bool]) -> pd.DataFrame:
        """
        Tests dataset for control, using defined SPC rules.
        """

        out_of_control_dict = {
            f"{rule} violation": [] for rule, enabled in rules_dict.items() if enabled
        }

        for idx in self.__cl_data.period.unique():

            df_period_filtered = self.__cl_data[self.__cl_data.period == idx]

            if self.seasonal and not self.__seasonal_adjustment_passed.get(idx, False):

                for _, seasonal_filtered in df_period_filtered.groupby("season"):
                    violations = _rules_func(
                        seasonal_filtered["process"],
                        seasonal_filtered["CL"],
                        seasonal_filtered["LCL"],
                        seasonal_filtered["UCL"],
                        rules_dict,
                    )
                    for rule, timestamps in violations.items():
                        out_of_control_dict[rule].extend(timestamps)

            else:

                violations = _rules_func(
                    df_period_filtered["process"],
                    df_period_filtered["CL"],
                    df_period_filtered["LCL"],
                    df_period_filtered["UCL"],
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
