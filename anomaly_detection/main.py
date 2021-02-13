from fbprophet import Prophet
import pandas as pd
import numpy as np
import pytest
import altair as alt


def fit(self, interval_width=0.95, *args, **kwargs):
    """
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
    model = Prophet(interval_width=interval_width, *args, **kwargs)

    if self.additional_regressors:
        (model.add_regressor(regressor) for regressor in self.additional_regressors)

    model = model.fit(self.data)

    self.model = model


def predict(self, predict_data=None, *args, **kwargs):
    if not self.model:
        self.fit(*args, **kwargs)

    if not predict_data:
        predict_data = self.data

    self.results = self.model.predict(predict_data)


def detect_anomalies(self):
    def _calc_percent_deviation(boundary, value):
        return ((value - boundary) / boundary).astype(np.float32)

    if not isinstance(self.results, pd.DataFrame):
        self.predict()

    results = self.results
    results["actuals"] = self.data["y"]

    large_anoms = results["actuals"] > results["yhat_upper"]
    small_anoms = results["actuals"] < results["yhat_lower"]

    results.loc[large_anoms, "anomaly_score"] = _calc_percent_deviation(
        results.loc[large_anoms, "yhat_upper"], results.loc[large_anoms, "actuals"],
    )

    results.loc[small_anoms, "anomaly_score"] = _calc_percent_deviation(
        results.loc[small_anoms, "yhat_lower"], results.loc[small_anoms, "actuals"],
    )

    self.results = results


@pytest.mark.skip(reason="plotting function")
def plot_forecasts(self):
    return self.model.plot(self.results)


@pytest.mark.skip(reason="plotting function")
def plot_components(self):
    return self.model.plot_components(self.results)


def plot_anomalies(self, width=870, height=450, fontSize=20):

    alt.data_transformers.disable_max_rows()

    results = self.results

    interval = (
        alt.Chart(results)
        .mark_area(interpolate="basis", color="#00b0ff")
        .encode(
            x=alt.X("ds:T", title="date"),
            y="yhat_upper",
            y2="yhat_lower",
            tooltip=["ds", "actuals", "yhat_lower", "yhat_upper"],
        )
        .interactive()
        .properties(title="Anomaly Detection")
    )

    actuals = (
        alt.Chart(results[results.anomaly_score.isna()])
        .mark_circle(size=15, opacity=0.7, color="Black")
        .encode(
            x="ds:T",
            y=alt.Y("actuals", title=self.target),
            tooltip=["ds", "actuals", "yhat_lower", "yhat_upper"],
        )
        .interactive()
    )

    anomalies = (
        alt.Chart(results[~results.anomaly_score.isna()])
        .mark_circle(size=30, color="Red")
        .encode(
            x="ds:T",
            y=alt.Y("actuals", title=self.target),
            tooltip=["ds", "actuals", "yhat_lower", "yhat_upper"],
            size=alt.Size("anomaly_score", legend=None),
        )
        .interactive()
    )

    return (
        alt.layer(interval, actuals, anomalies)
        .properties(width=width, height=height)
        .configure_title(fontSize=fontSize)
    )

