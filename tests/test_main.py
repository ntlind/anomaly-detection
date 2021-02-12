import pandas as pd
import numpy as np
import anomaly_detection as ad

def test__format_dataframe():
    simple_example = ad.utilities._get_prophet_example()
    example = ad.utilities._get_prophet_example()

    # should pass
    detector = ad.AnomalyDetector(data=simple_example)
    detector = ad.AnomalyDetector(data=example.set_index("datetime"), target="sales_float")
    detector = ad.AnomalyDetector(data=example, datetime_column="datetime", target="sales_float")

    # should fail
    try:
        detector = ad.AnomalyDetector(data=example)     
    except AssertionError:
        pass

    try:
        detector = ad.AnomalyDetector(data=example, target="sales_float")
    except AssertionError:
        pass 

    assert set("y", "ds").issubset(set(detector.data.columns))


def test_pystan():
    """Check pystan for build errors"""
    import pystan

    model_code = "parameters {real y;} model {y ~ normal(0,1);}"
    model = pystan.StanModel(model_code=model_code)  # this will take a minute
    y = model.sampling(n_jobs=1).extract()["y"]
    assert np.round(y.mean(), 0) == 0 


def test_main():
    # # Single series, no hierarchy
    # data = pd.read_csv(
    #     "https://raw.githubusercontent.com/facebookincubator/prophet/master/examples/example_retail_sales.csv"
    # )

    # detector = ad.AnomalyDetector(target="y", datetime_column="ds")

    # print(dector.data)


def test_series():
    pass


def test_dataframe():
    pass


if __name__ == "__main__":
    test_pystan()
    test_main()
