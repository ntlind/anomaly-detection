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
    prophet_example = ad.utilities._get_prophet_example()
    example = ad.utilities._get_test_example()

    # should pass
    detector = ad.AnomalyDetector(data=prophet_example)
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

    assert set(("sales_float", "datetime")).issubset(set(example))
    assert set(("y", "ds")).issubset(set(detector.data.columns))


def test_fit():
    detector = ad.utilities._get_detector_example()
    detector.fit()

    assert detector.model


def test_predict():
    detector = ad.utilities._get_detector_example()
    detector.predict()

    forecasts = detector.results

    assert (forecasts["yhat"] >= 100000).all() & (forecasts["yhat"] <= 518253).all()


def test_detect_anomalies():
    detector = ad.utilities._get_detector_example()
    detector.detect_anomalies()

    total_rows = len(detector.results)

    null_count = sum(detector.results["anomaly_score"] == 0)

    # As a rough guideline, there shouldn't be that many anomalies
    assert (null_count / total_rows) >= 0.92


def test_get_results():
    example = ad.utilities._get_test_example()
    prophet_example = ad.utilities._get_prophet_example()

    detector = ad.utilities._get_detector_example(type_="test")
    prophet_detector = ad.utilities._get_detector_example(type_="prophet")

    for data, detect in [(example, detector), (prophet_example, prophet_detector)]:

        detect.predict()

        detect.detect_anomalies()

        final_output = detect.get_results()

        # check that changepoints are assigned correctly
        output_changepoints = set(
            final_output.loc[
                final_output["changepoint_flag"] == 1, detect.datetime_column
            ]
        )
        model_changepoints = set(detect.model.changepoints)
        print(detect.results)
        print(final_output)
        assert output_changepoints == model_changepoints

        # check that column / date order hasn't changed
        final_output_set = set(final_output[[detect.target, detect.datetime_column]])
        data_set = set(data[[detect.target, detect.datetime_column]])
        assert final_output_set == data_set

        # check that correct columns are included
        assert set(("anomaly_score", "changepoint_flag")).issubset(
            set(final_output.columns)
        )


def test_plot_forecasts():
    detector = ad.utilities._get_detector_example()
    results = detector.detect_anomalies()

    print(detector.plot_forecasts())


def test_plot_components():
    detector = ad.utilities._get_detector_example()
    results = detector.detect_anomalies()

    print(detector.plot_components())


def test_plot_anomalies():
    detector = ad.utilities._get_detector_example()
    results = detector.detect_anomalies()

    print(detector.plot_anomalies())


if __name__ == "__main__":
    test_pystan()
    test__format_dataframe()
    test_fit()
    test_predict()
    test_detect_anomalies()
    test_get_results()
    test_plot_forecasts()
    test_plot_components()
    test_plot_anomalies()
    print("Passed!")

