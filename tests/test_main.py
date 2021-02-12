import pandas as pd
import numpy as np
import anomaly_detection as ad
import pystan
import fbprophet


def test_pystan():
    """Check pystan for build errors"""
    import pystan

    model_code = "parameters {real y;} model {y ~ normal(0,1);}"
    model = pystan.StanModel(model_code=model_code)  # this will take a minute
    y = model.sampling(n_jobs=1).extract()["y"]
    assert np.round(y.mean(), 0) == 0


def test__format_dataframe():
    simple_example = ad.utilities._get_prophet_example()
    example = ad.utilities._get_test_example()

    # should pass
    detector = ad.AnomalyDetector(data=simple_example)
    detector = ad.AnomalyDetector(
        data=example, datetime_column="datetime", target="sales_float"
    )
    detector = ad.AnomalyDetector(
        data=example.set_index("datetime"), target="sales_float"
    )

    # should fail
    try:
        detector = ad.AnomalyDetector(data=example)
    except AssertionError:
        pass

    try:
        detector = ad.AnomalyDetector(data=example, target="sales_float")
    except AssertionError:
        pass

    assert set(("y", "ds")).issubset(set(detector.data.columns))


def test_fit():
    detector = ad.utilities._get_detector_example()
    detector.fit()

    assert detector.model


def test_predict():
    detector = ad.utilities._get_detector_example()
    detector.predict()

    forecasts = self.forecasts

    assert (forecasts["yhat"] >= 100000).all() & (forecasts["yhat"] <= 518253).all()


def test_detect_anomalies():
    detector = ad.utilities._get_detector_example()
    forecasts = detector.detect_anomalies()

    total_rows = len(forecasts)

    null_count = sum(forecasts["anomaly_score"].isnull())

    assert (null_count / total_rows) >= 0.95
    print(forecasts["anomaly_score"].describe())


if __name__ == "__main__":
    # test_pystan()
    test__format_dataframe()
    test_fit()
    test_predict()
    test_detect_anomalies()

