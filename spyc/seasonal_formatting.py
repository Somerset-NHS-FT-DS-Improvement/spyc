def _group_data_by_season(data, seasonal_int):
    if seasonal_int == 7:
        grouped_data = data.groupby(by=data.index.weekday)
    elif seasonal_int == 24:
        grouped_data = data.groupby(by=data.index.hour)
    elif seasonal_int == 12:
        grouped_data = data.groupby(by=data.index.month)
    else:
        grouped_data = data.groupby(by=data.index.quarter)

    return grouped_data


def _add_seasonal_column(data, seasonal_int):
    if seasonal_int == 7:
        return data.index.dayofweek
    elif seasonal_int == 24:
        return data.index.hour
    elif seasonal_int == 12:
        return data.index.month
    elif seasonal_int == 4:
        return data.index.quarter
    else:
        print('none')


def _data_points_per_season(seasonal_grouped_data):
    """
    If each season small size, use global mean until more data is available.
    Arbitrarily set to 15 points, can be adjusted.
    """
    return (seasonal_grouped_data.count().values > 15).all()
