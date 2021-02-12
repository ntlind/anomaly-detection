import pandas as pd


def test_main():
    # Single series, no hierarchy
    data = pd.read_csv(
        "https://raw.githubusercontent.com/facebookincubator/prophet/master/examples/example_retail_sales.csv"
    ).set_index("ds")


def test_series():
    pass


def test_dataframe():
    pass


if __name__ == "__main__":
    test_main()
