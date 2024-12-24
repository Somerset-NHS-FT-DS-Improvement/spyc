import numpy as np
import pandas as pd
from .control_rules import _rules_func
from .seasonal_formatting import _add_seasonal_column, _group_data_by_season, _data_points_per_season
from .helpers import convert_to_timestamp
from .plotly_chart import plotly_chart, combine_figures


class SPC:

    def __init__(self,
                 data_in,
                 target_col,
                 fix_control_start_dt=None,
                 fix_control_end_dt=None,
                 seasonal=None):

        """
        Statistical Process Control automation, which can generalise for any custom/default chart type (defined using
        the spc_calc_func parameter).
        Seasonal adjustments (control defined for each season) can be set by using the seasonal parameter, if present.
        Trends are not automatically handled and should ideally be removed prior to feeding in dataset, if present.
        Control lines can be re-calculated following a process change, using the add_process_change_date() method.

        - data_in (Pandas.DataFrame): Datetime index with frequency attribute.
        - target_col (str): Must exist in data_in.
        - fix_control_start_dt (str): If None, reverts to start of data/start of previous value in process_change_dates.
        - fix_control_end_dt (str): If None, reverts to end of data/end of next value in process_change_dates.
        - seasonal (int): Time steps for repeating seasonal pattern (i.e., 24 for hourly data, 7 for daily etc...)
        """

        self.spc_outputs_dict = {}
        self.data_in = data_in.copy()
        self.target_col = target_col

        # If values not set, default to start/end dates of data_in
        self.fix_control_start_dt = fix_control_start_dt if fix_control_start_dt is not None \
            else self.data_in.index.min()
        self.fix_control_end_dt = fix_control_end_dt if fix_control_end_dt is not None else self.data_in.index.max()
        self.seasonal = seasonal

        assert target_col in data_in.columns, "Column not found in input dataframe!"
        assert seasonal in [None, 24, 7, 12, 4], "seasonal parameter only accepts 4 (quarterly), 12 (monthly), " \
                                                 "7 (daily) or 24 (hourly)."
        assert isinstance(self.data_in.index, pd.DatetimeIndex), "Dataframe must have datetime index"

        # Add seasonal column to data_in and store unique seasonal periods
        if self.seasonal:
            self.data_in['season'] = _add_seasonal_column(self.data_in, self.seasonal)  # Add season column
            self.seasonal_periods = self.data_in['season'].unique()  # Store unique seasonal values

        # Track process change dates & control line calculation start/end dates, if specified
        self.control_line_dates = {self.data_in.index.min(): {'cl_start_data': self.fix_control_start_dt,
                                                              'cl_end_data': self.fix_control_end_dt,
                                                              'process_names': None}}
        self.__data_blocks_list = None
        self.__process_change_dates = None
        self.__seasonal_adjustment_passed = {}

    def add_process_change_date(self,
                                change_date,
                                fix_control_start_dt=None,
                                fix_control_end_dt=None,
                                process_name=None) -> None:

        """
        Optional method, which allows control lines to be re-calculated following a systematic change to the measured
        process. Can be called multiple times if a series of process changes occurs during the observed period.
        """

        fix_control_start_dt = fix_control_start_dt or change_date
        fix_control_end_dt = fix_control_end_dt or self.data_in.index.max()

        # Default max. of previous fix_control_end_dt to change_date, if none provided, or date exceeds change_date/
        if self.fix_control_end_dt == self.data_in.index.max() or self.fix_control_end_dt > fix_control_start_dt:
            most_recent_key = list(self.control_line_dates.keys())[-1]  # Get previous change date
            self.control_line_dates[most_recent_key]['cl_end_data'] = change_date  # Update end date

        # Add new change_date to dict.
        self.control_line_dates[change_date] = {'cl_start_data': fix_control_start_dt,
                                                'cl_end_data': fix_control_end_dt,
                                                'process_names': process_name}

        self.control_line_dates = convert_to_timestamp(self.control_line_dates)

    def add_spc(self, spc_name, spc_calc_func, rule_1=True, rule_2=False, rule_3=False, rule_4=False, rule_5=False):

        """
        Calculates control lines (including re-calculation, if specified), tests data for control.

        Inputs:
        - spc_calc_func (function): Must return dictionary containing keys:
            * 'process' (pandas.Series): The measured process being tracked using SPC (e.g., individuals/moving-range)
            * 'CL' (float/int): The center/control line
            * 'UCL' (float/int): The upper control line
            * 'LCL' (float/int): The lower control line
        - rule_1 (bool, default=True): Outside of control limits.
        - rule_2 (bool, default=False): 8 (or more) consecutive points above/below center line.
        - rule_3 (bool, default=False): 6 (or more) consecutive points increasing/decreasing.
        - rule_4 (bool, default=False): 2 out of 3 consecutive points outside +/- 2 sigma limits
        - rule_5 (bool, default=False): 15 (or more) consecutive points within the +/- 1 sigma limits.

        Outputs:
            - Plotly.Figure
        """

        process_change_dates = list(self.control_line_dates.keys())

        rules_dict = {'Rule 1': rule_1, 'Rule 2': rule_2, 'Rule 3': rule_3, 'Rule 4': rule_4,
                      'Rule 5': rule_5}

        # Get list of process change dates, while excluding first element
        self.__process_change_dates = process_change_dates[1:] if len(process_change_dates) > 1 else None

        # Obtain list of datasets for each process_change_dates periods, if add_process_change() called.
        self.__data_blocks_list = self.__split_data_by_change_dates()

        self._calculate_control_lines(spc_calc_func=spc_calc_func)
        spc_data = self._test_for_control(rules_dict=rules_dict)

        self.spc_outputs_dict[spc_name] = {'data': spc_data,
                                           'figure': plotly_chart(spc_data,
                                                                  self.__process_change_dates,
                                                                  self.control_line_dates,
                                                                  figure_title=spc_name)}

    def plot_spc(self, save_to_html=False, file_name='SPC Analytics Chart'):

        figs = [output['figure'] for output in self.spc_outputs_dict.values()]
        fig_names = list(self.spc_outputs_dict.keys())

        return combine_figures(figures=figs,
                               fig_names=fig_names,
                               process_change_dates=self.__process_change_dates,
                               process_change_dict=self.control_line_dates,
                               save_to_html=save_to_html,
                               file_name=file_name)

    def __split_data_by_change_dates(self):

        """
        Splits dataframe into data "blocks", following each process change date
        (if process_change_dates values provided). Otherwise, the original dataframe is returned, as the single element
        in a list.
        """

        if not self.__process_change_dates:
            return [self.data_in]
        else:

            change_dates = sorted(self.__process_change_dates)

            data_blocks_list = []  # Store each data "block" in list
            prev_date = self.data_in.index.min()  # Initialise previous change date to start of data

            for idx, cd in enumerate(change_dates):
                block = self.data_in[(self.data_in.index >= pd.to_datetime(prev_date)) & \
                                     (self.data_in.index < pd.to_datetime(cd))].copy()
                data_blocks_list.append(block)
                prev_date = cd  # Update prev_date variable

            # Append last data block to data_blocks_list
            final_period = self.data_in[(self.data_in.index >= pd.to_datetime(cd))].copy()
            data_blocks_list.append(final_period)

            return data_blocks_list  # Store list of data block

    def _calculate_control_lines(self, spc_calc_func):

        """
        Calculates control lines for each "block" dataset (only one dataset if no process_change_dates set).
        """

        cl_df = pd.DataFrame()
        cl_df.index = self.data_in.index
        cl_df[['CL', 'LCL', 'UCL', 'process', 'period']] = np.nan
        if self.seasonal:
            cl_df['season'] = self.data_in['season'].values  # Add season column

        # Store control lines in dictionary for quick reference
        cl_dictionary = {}

        for idx, df in enumerate(self.__data_blocks_list):

            # Filter data for start/end control line fixed dates
            start_date_dict = self.control_line_dates.get(df.index[0], None)
            end_date_dict = self.control_line_dates.get(df.index[0], None)

            if start_date_dict is None or end_date_dict is None:
                raise ValueError(f"Warning: Control line dates not found for change date {df.index[0]}. Make sure "
                                 f"change date is of same type as index.")

            else:
                start_date = start_date_dict['cl_start_data']
                end_date = end_date_dict['cl_end_data']

            check_data_points_per_season = _data_points_per_season(_group_data_by_season(df.loc[start_date: end_date],
                                                                                         self.seasonal))

            if self.seasonal and not check_data_points_per_season:
                self.__seasonal_adjustment_passed[idx] = True
                print('Not enough data to build reliable estimate of mean for each season. Global mean will be '
                      'estimated until more data is added (control line calculation period is changed, or more data '
                      'collected)')

            if self.seasonal and check_data_points_per_season:

                seasonal_grouped = _group_data_by_season(df, self.seasonal)

                # Calculate control lines for each seasonal period
                cl_dict = seasonal_grouped.apply(
                    lambda group: spc_calc_func(group, self.target_col, start_date, end_date))
                cl_dictionary[idx] = cl_dict

                cl_df.loc[df.index, 'CL'] = df['season'].map(lambda s: cl_dict[s]['CL'])
                cl_df.loc[df.index, 'LCL'] = df['season'].map(lambda s: cl_dict[s]['LCL'])
                cl_df.loc[df.index, 'UCL'] = df['season'].map(lambda s: cl_dict[s]['UCL'])

                for sn in self.seasonal_periods:
                    df.loc[df['season'] == sn, 'process'] = cl_dict[sn]['process']
                cl_df.loc[df.index, 'process'] = df['process'].values

            else:

                cl_dict = spc_calc_func(df, self.target_col, start_date, end_date)
                cl_dictionary[idx] = cl_dict

                cl_df.loc[df.index, 'CL'] = cl_dict['CL']
                cl_df.loc[df.index, 'LCL'] = cl_dict['LCL']
                cl_df.loc[df.index, 'UCL'] = cl_dict['UCL']
                cl_df.loc[df.index, 'process'] = cl_dict['process']

            cl_df.loc[df.index, 'period'] = idx

        self.__cl_dict = cl_dictionary
        self.__cl_data = cl_df

    def _test_for_control(self, rules_dict):

        """
        Tests dataset for control, using the set rules.
        """

        out_of_control_dict = {f"{rule} violation": [] for rule, enabled in rules_dict.items() if enabled}

        for idx in self.__cl_data.period.unique():

            df_period_filtered = self.__cl_data[self.__cl_data.period == idx]

            if self.seasonal and not self.__seasonal_adjustment_passed.get(idx, False):

                for s in self.seasonal_periods:

                    seasonal_filtered = df_period_filtered[df_period_filtered.season == s]  # Filter for each season

                    # Test control for each season
                    violations = _rules_func(seasonal_filtered['process'],
                                             self.__cl_dict[idx].loc[s]['CL'],
                                             self.__cl_dict[idx].loc[s]['LCL'],
                                             self.__cl_dict[idx].loc[s]['UCL'],
                                             rules_dict)
                    for rule, timestamps in violations.items():
                        out_of_control_dict[rule].extend(timestamps)

            else:

                violations = _rules_func(df_period_filtered['process'],
                                         self.__cl_dict[idx]['CL'],
                                         self.__cl_dict[idx]['LCL'],
                                         self.__cl_dict[idx]['UCL'],
                                         rules_dict)

            for rule, timestamps in violations.items():
                out_of_control_dict[rule].extend(timestamps)

        # Store unique dates with "out of control" variation.
        violations_dates_set = set()
        for rule, dates in out_of_control_dict.items():
            violations_dates_set.update(dates)

        violations_dates_set = sorted(violations_dates_set)  # Ensure dates are sorted
        df = pd.DataFrame(0, index=violations_dates_set, columns=out_of_control_dict.keys())
        for rule, dates in out_of_control_dict.items():
            df.loc[dates, rule] = 1

        merged_df = self.__cl_data.merge(df, how='left', left_index=True, right_index=True)

        merged_df[list(out_of_control_dict.keys())] = merged_df[list(out_of_control_dict.keys())].fillna(0).astype(int)

        return merged_df
