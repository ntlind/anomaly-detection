from fbprophet import Prophet
from utilities import _ensure_is_list


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

    def __init__(
        self, data, target=None, datetime_column=None, additional_regressors=None,
    ):

        self.target = assert_target(target)
        self.datetime_column = assert_datetime(datetime_column)
        self.additional_regressors = _ensure_is_list(additional_regressors)

        self.data = _format_dataframe(data)

        self.model = None

        def _assert_target(self, target, data):
            if not target:
                assert isinstance(
                    data, pd.Series
                ), "Please pass your dependent variable name to target when not using a Series"
                self.target = data.name
            else:
                self.target = target

        def _assert_datetime(self, datetime_column, data):
            if not datetime_column:
                assert isinstance(
                    data.index, pd.DatetimeIndex
                ), "Please pass datetime_column if not using a Dataframe or Series with a time-series index"
                self.datetime_column = data.index.name
                data.reset_index(inplace=True)
            else:
                self.datetime_column = datetime_column

        def _format_dataframe(self, data):
            columns = [self.target, self.datetime_column]

            if additional_regressors:
                columns.append(additional_regressors, inplace=True)

            self.data = data.loc[:, columns].rename(
                {self.datetime_column: "ds", self.target: "y"}
            )

    def fit_predict_model(self, interval_width=0.95, *args, **kwargs):
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
            'additive' (default) or 'multiplicative'.
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
            estimation.
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
        forecast = model.predict(self.data)
        # forecast["fact"] = dataframe["y"].reset_index(drop=True)
        return forecast


# pred = fit_predict_model(df1)


# def detect_anomalies(forecast):
#     forecasted = forecast[['ds','trend', 'yhat', 'yhat_lower', 'yhat_upper', 'fact']].copy()
#     #forecast['fact'] = df['y']

#     forecasted['anomaly'] = 0
#     forecasted.loc[forecasted['fact'] > forecasted['yhat_upper'], 'anomaly'] = 1
#     forecasted.loc[forecasted['fact'] < forecasted['yhat_lower'], 'anomaly'] = -1

#     #anomaly importances
#     forecasted['importance'] = 0
#     forecasted.loc[forecasted['anomaly'] ==1, 'importance'] = \
#         (forecasted['fact'] - forecasted['yhat_upper'])/forecast['fact']
#     forecasted.loc[forecasted['anomaly'] ==-1, 'importance'] = \
#         (forecasted['yhat_lower'] - forecasted['fact'])/forecast['fact']

#     return forecasted

# pred = detect_anomalies(pred)


# def plot_anomalies(forecasted):
#     interval = alt.Chart(forecasted).mark_area(interpolate="basis", color = '#7FC97F').encode(
#     x=alt.X('ds:T',  title ='date'),
#     y='yhat_upper',
#     y2='yhat_lower',
#     tooltip=['ds', 'fact', 'yhat_lower', 'yhat_upper']
#     ).interactive().properties(
#         title='Anomaly Detection'
#     )

#     fact = alt.Chart(forecasted[forecasted.anomaly==0]).mark_circle(size=15, opacity=0.7, color = 'Black').encode(
#         x='ds:T',
#         y=alt.Y('fact', title='sales'),
#         tooltip=['ds', 'fact', 'yhat_lower', 'yhat_upper']
#     ).interactive()

#     anomalies = alt.Chart(forecasted[forecasted.anomaly!=0]).mark_circle(size=30, color = 'Red').encode(
#         x='ds:T',
#         y=alt.Y('fact', title='sales'),
#         tooltip=['ds', 'fact', 'yhat_lower', 'yhat_upper'],
#         size = alt.Size( 'importance', legend=None)
#     ).interactive()

#     return alt.layer(interval, fact, anomalies)\
#               .properties(width=870, height=450)\
#               .configure_title(fontSize=20)

#   plot_anomalies(pred)
