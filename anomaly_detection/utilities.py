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
