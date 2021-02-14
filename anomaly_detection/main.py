"""
All core methods for anomaly-detector, including plotting functions
"""

from fbprophet import Prophet
import pandas as pd
import numpy as np
import pytest
import altair as alt


def fit(self, interval_width=0.95, changepoint_range=1, *args, **kwargs):
    """
    Fit Prophet to the data.

    Parameters (passed as *args or **kwargs to Prophet)
    ----------
    interval_width: float, default .95
        Float, width of the uncertainty intervals provided
        for the forecast. If mcmc_samples=0, this will be only the uncertainty
        in the trend using the MAP estimate of the extrapolated generative
        model. If mcmc.samples>0, this will be integrated over all model
        parameters, which will include uncertainty in seasonality. In this library,
        we override FB's default from .8 to .95 to provide more stringer
        anomaly detection.
    growth: str, default "linear"
        String 'linear' or 'logistic' to specify a linear or logistic trend.
    changepoints: list, default None 
        List of dates at which to include potential changepoints. If
        not specified, potential changepoints are selected automatically.
    n_changepoints: int, default 25
        Number of potential changepoints to include. Not used
        if input `changepoints` is supplied. If `changepoints` is not supplied,
        then n_changepoints potential changepoints are selected uniformly from
        the first `changepoint_range` proportion of the history.
    changepoint_range: float, default .8 
        Proportion of history in which trend changepoints will
        be estimated. Defaults to 0.8 for the first 80%. Not used if
        `changepoints` is specified.
    yearly_seasonality: bool, str, or int, default "auto" 
        If true, adds Fourier terms to model changes in annual seasonality. Pass 
        an int to manually control the number of Fourier terms added, where 10 
        is the default and 20 creates a more flexible model but increases the 
        risk of overfitting.
    weekly_seasonality: bool, str, or int, default "auto"
        Fit weekly seasonality.
        Can be 'auto', True, False, or a number of Fourier terms to generate.
    daily_seasonality: bool, str, or int, default "auto"  
        If true, adds Fourier terms to model changes in daily seasonality. Pass 
        an int to manually control the number of Fourier terms added, where 10 
        is the default and 20 creates a more flexible model but increases the 
        risk of overfitting.
    holidays: bool, default None
        pd.DataFrame with columns holiday (string) and ds (date type)
        and optionally columns lower_window and upper_window which specify a
        range of days around the date to be included as holidays.
        lower_window=-2 will include 2 days prior to the date as holidays. Also
        optionally can have a column prior_scale specifying the prior scale for
        that holiday.
    seasonality_mode: str, default "additive"
        'additive' (default) or 'multiplicative'. Multiplicative seasonality implies
        that each season applies a scaling effect to the overall trend, while additive 
        seasonality implies adding seasonality to trend to arrive at delta.
    seasonality_prior_scale: float, default 10.0
        Parameter modulating the strength of the
        seasonality model. Larger values allow the model to fit larger seasonal
        fluctuations, smaller values dampen the seasonality. Can be specified
        for individual seasonalities using add_seasonality.
    holidays_prior_scale: float, default 10.0
        Parameter modulating the strength of the holiday
        components model, unless overridden in the holidays input.
    changepoint_prior_scale: float, default 0.05 
        Parameter modulating the flexibility of the
        automatic changepoint selection. Large values will allow many
        changepoints, small values will allow few changepoints.
    mcmc_samples: int, default 0
        Integer, if greater than 0, will do full Bayesian inference
        with the specified number of MCMC samples. If 0, will do MAP
        estimation, which only measures uncertainty in the trend and 
        observation noise but is much faster to run.
    uncertainty_samples: int, default 1000
        Number of simulated draws used to estimate
        uncertainty intervals. Settings this value to 0 or False will disable
        uncertainty estimation and speed up the calculation.
    stan_backend: str, default None
        str as defined in StanBackendEnum default: None - will try to
        iterate over all available backends and find the working one
    """
    model = Prophet(
        interval_width=interval_width,
        changepoint_range=changepoint_range,
        *args,
        **kwargs
    )

    if self.additional_regressors:
        (model.add_regressor(regressor) for regressor in self.additional_regressors)

    model = model.fit(self.data)

    self.model = model


def predict(self, df=None, *args, **kwargs):
    """
    Predict using the prophet model.
    
    Parameters
    ----------
    df: pd.DataFrame with dates for predictions (column ds), and capacity
        (column cap) if logistic growth. If not provided, predictions are
        made on the history.
    """
    if not self.model:
        self.fit(*args, **kwargs)

    if df is None:
        df = self.model.history.copy()
    else:
        if df.shape[0] == 0:
            raise ValueError("Dataframe has no rows.")
        df = self.model.setup_dataframe(df.copy())

    df["trend"] = self.model.predict_trend(df)
    seasonal_components = self.model.predict_seasonal_components(df)
    if self.model.uncertainty_samples:
        intervals = self.model.predict_uncertainty(df)
    else:
        intervals = None

    df2 = pd.concat((df, intervals, seasonal_components), axis=1)
    df2["yhat"] = (
        df2["trend"] * (1 + df2["multiplicative_terms"]) + df2["additive_terms"]
    )

    columns_to_keep = list(self.model.history.columns)
    for col in ["y_scaled", "t"]:
        columns_to_keep.remove(col)

    for col in [
        "trend",
        "trend_lower",
        "trend_upper",
        "yhat_upper",
        "yhat_lower",
        "yhat",
    ]:
        columns_to_keep.append(col)

    if "cap" in df:
        columns_to_keep.append("cap")
    if self.model.logistic_floor:
        columns_to_keep.append("floor")

    self.results = df2[columns_to_keep]


def detect_anomalies(self, *args, **kwargs):
    """
    Leverages Prophet's predictions to flag actuals greater than their upper confidence interval or 
    lower than their lower confidence interval.  Also adds a "changepoint_flag" for dates where Prophet 
    detected a changepoint. If no predictions exist to build these variables off of, this
    function will automatically call self.predict(). 
    """

    def _calc_percent_deviation(boundary, value):
        return ((value - boundary) / boundary).astype(np.float32)

    if not isinstance(self.results, pd.DataFrame):
        self.predict(*args, **kwargs)

    results = self.results

    large_anoms = results["y"] > results["yhat_upper"]
    small_anoms = results["y"] < results["yhat_lower"]

    results["anomaly_score"] = 0
    results.loc[large_anoms, "anomaly_score"] = _calc_percent_deviation(
        results.loc[large_anoms, "yhat_upper"], results.loc[large_anoms, "y"],
    )

    results.loc[small_anoms, "anomaly_score"] = _calc_percent_deviation(
        results.loc[small_anoms, "yhat_lower"], results.loc[small_anoms, "y"],
    )

    results["changepoint_flag"] = 0
    results.loc[results["ds"].isin(self.model.changepoints), "changepoint_flag"] = 1

    self.results = results


def get_results(self, *args, **kwargs):
    """
    Return your original dataframe with added anomaly_score and changepoint_flag columns
    """

    if not isinstance(self.results, pd.DataFrame):
        self.predict(*args, **kwargs)

    if "anomaly_score" not in self.results.columns:
        self.detect_anomalies(*args, **kwargs)

    results = self.results.rename(
        {"ds": self.datetime_column, "y": self.target}, axis=1
    )

    return results.drop(
        ["floor", "trend", "yhat_upper", "yhat_lower", "yhat"], axis=1, errors="ignore"
    )


@pytest.mark.skip(reason="plotting function")
def plot_forecasts(self):
    """
    Wrapper function for Prophet's default plotting functionality.
    """
    return self.model.plot(self.results)


@pytest.mark.skip(reason="plotting function")
def plot_components(self):
    """
    Wrapper function for Prophet's default plotting functionality.
    """
    return self.model.plot_components(self.results)


@pytest.mark.skip(reason="plotting function")
def plot_anomalies(self, width=870, height=450):
    """
    Plot your outliers, changepoints, actuals, and confidence intervals
     on one convenient graph.

    Parameters (passed as *args or **kwargs to Prophet)
    ----------
    width : int, default 870
        Set the plot's width.
    Height : int, default 450
        Set the plot's height.
    """
    alt.data_transformers.disable_max_rows()

    results = self.results

    color_scheme = ["#000000", "#999999", "#00b0ff", "#ff4f00"]

    interval = (
        alt.Chart(results)
        .transform_calculate(name='"Confidence Bounds"')
        .mark_area(interpolate="basis")
        .encode(
            x=alt.X("ds:T", title="date"),
            y="yhat_upper",
            y2="yhat_lower",
            tooltip=["ds", "y", "yhat_lower", "yhat_upper"],
            color=alt.Color(
                "name:N",
                scale=alt.Scale(range=color_scheme),
                legend=alt.Legend(title=None),
            ),
        )
        .interactive()
    )

    anomaly_mask = results.anomaly_score != 0
    actuals = (
        alt.Chart(results[~anomaly_mask])
        .transform_calculate(name='"Actuals"')
        .mark_circle(size=15, opacity=0.7, color="Black")
        .encode(
            x="ds:T",
            y=alt.Y("y", title=self.target),
            tooltip=["ds", "y", "yhat_lower", "yhat_upper"],
            color=alt.Color("name:N", legend=alt.Legend(title=None),),
        )
        .interactive()
    )

    results["abs_anomaly_score"] = abs(results["anomaly_score"])
    anomalies = (
        alt.Chart(results[anomaly_mask])
        .transform_calculate(name='"Outliers"')
        .mark_circle(size=30, color="Red")
        .encode(
            x="ds:T",
            y=alt.Y("y", title=self.target),
            tooltip=["ds", "y", "yhat_lower", "yhat_upper", "anomaly_score"],
            size=alt.Size("abs_anomaly_score", legend=None),
            color=alt.Color("name:N", legend=alt.Legend(title=None),),
        )
        .interactive()
    )

    changepoints = results.loc[results["changepoint_flag"] == 1]

    changepoints = (
        alt.Chart(changepoints)
        .transform_calculate(name='"Changepoint"')
        .mark_rule(color="lightgray", size=0.7, strokeDash=[3, 5])
        .encode(
            x="ds:T",
            tooltip=["ds"],
            color=alt.Color("name:N", legend=alt.Legend(title=None),),
        )
        .interactive()
    )

    return (
        alt.layer(interval, actuals, anomalies, changepoints)
        .properties(width=width, height=height)
        .configure_axis(grid=False)
    )

