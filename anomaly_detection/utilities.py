import pandas as pd
import numpy as np
import pytest


@pytest.mark.skip(reason="simple python functionality")
def _ensure_is_list(obj):
    """
    Return an object in a list if not already wrapped. Useful when you 
    want to treat an object as a collection,even when the user passes a string
    """
    if obj:
        if not isinstance(obj, list):
            return [obj]
        else:
            return obj
    else:
        return None


@pytest.mark.skip(reason="simple pandas command")
def _get_prophet_example():
    return pd.read_csv(
        "https://raw.githubusercontent.com/facebookincubator/prophet/master/examples/example_retail_sales.csv"
    )


@pytest.mark.skip(reason="creates a simple pandas dataframe")
def _get_test_example(convert_dtypes=True):
    """
    Return a made-up dataframe that can be used for testing purposes
    """

    column_names = [
        "datetime",
        "category",
        "sales_int",
        "product",
        "state",
        "store",
        "sales_float",
    ]

    example = pd.DataFrame(
        [
            ["2020-01-01", "Cat_1", 113, "Prod_3", "CA", "Store_1", 113.21],
            ["2020-01-02", "Cat_1", 10000, "Prod_3", "CA", "Store_1", 10000.00],
            ["2020-01-03", "Cat_1", 214, "Prod_3", "CA", "Store_1", np.nan],
            ["2020-01-05", "Cat_1", 123, "Prod_3", "CA", "Store_1", 123.21],
            ["2019-12-30", "Cat_2", 5, "Prod_4", "CA", "Store_1", 5.1],
            ["2019-12-31", "Cat_2", np.nan, "Prod_4", "CA", "Store_1", np.nan],
            ["2020-01-01", "Cat_2", 0, "Prod_4", "CA", "Store_1", 0],
            ["2020-01-02", "Cat_2", -20, "Prod_4", "CA", "Store_1", -20.1],
            ["2019-12-30", "Cat_2", 2, "Prod_5", "CA", "Store_1", 2.1],
            ["2019-12-31", "Cat_2", 4, "Prod_5", "CA", "Store_1", 4.1],
            ["2020-01-01", "Cat_2", 10, "Prod_5", "CA", "Store_1", 10.2],
            ["2020-01-02", "Cat_2", -10, "Prod_5", "CA", "Store_1", -10.1],
        ],
        columns=column_names,
    )

    if convert_dtypes:
        example["datetime"] = pd.to_datetime(example["datetime"])

    return example
