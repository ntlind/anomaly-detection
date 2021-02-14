from fbprophet import Prophet
import pandas as pd
import numpy as np
import pytest
from anomaly_detection.utilities import _ensure_is_list


class AnomalyDetector:
    """
    Base class for AnomalyDetector.

    Parameters
    ----------
    data: pd.Series or pd.DataFrame,
        The pandas DataFrame or Series that you wish to model
    target: str, default None
        The target column for which to build the prediction. If None, class will assert
        that your data is a Series.
    datetime_column: str, default None
        The column to use as your datetime column. If None, class wlil asser that your 
        index is a datetime index.
    additional_regressors: list or str, default None
        If a list of column names or a str of a single column is passed, will include
        those additional columns as regressors in the Prophet model.
    *args, **kwargs
        Additional arguments are passed directly into Facebook's Prophet() module. See
        below for examples of what you might want to include
    """

    from anomaly_detection.io import save, load
    from anomaly_detection.main import (
        fit,
        predict,
        detect_anomalies,
        get_results,
        plot_forecasts,
        plot_components,
        plot_anomalies,
    )

    def __init__(
        self, data, target=None, datetime_column=None, additional_regressors=None,
    ):

        self.set_target(target, data)
        self.set_datetime(datetime_column, data)
        self.additional_regressors = _ensure_is_list(additional_regressors)

        self.set_data(data, additional_regressors)

        self.model = None
        self.results = None

    def set_target(self, target, data):
        if not target:
            assert (
                "y" in data.columns
            ), "Please pass your dependent variable name to the target argument."
            self.target = "y"
        else:
            self.target = target

    def set_datetime(self, datetime_column, data):
        if not datetime_column:
            if isinstance(data.index, pd.DatetimeIndex):
                index_name = data.index.name
                data.reset_index(inplace=True)
                data.rename({index_name: "ds"}, inplace=True, axis=1)

            assert (
                "ds" in data.columns
            ), "Please pass datetime_column if not including a 'ds' column"

            self.datetime_column = "ds"
        else:
            self.datetime_column = datetime_column

    def set_data(self, data, additional_regressors):

        # https://github.com/facebook/prophet/blob/3c69ce3312fcf91d15fd7e56812aa28cabf6f9f2/python/fbprophet/forecaster.py#L298
        data = data.sort_values(self.datetime_column)

        columns = [self.target, self.datetime_column]

        if additional_regressors:
            columns += additional_regressors

        data_copy = data.copy(deep=True)

        self.data = data_copy.loc[:, columns].rename(
            {self.datetime_column: "ds", self.target: "y"}, axis=1
        )

