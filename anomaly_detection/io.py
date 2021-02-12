"""
Functions to import or export AnomalyDetector.
"""

import pickle
import os


def save(self, name="AnomalyDetector.pkl", path=os.getcwd() + "\\"):
    """
    Save the AnomalyDetector as a pickle.

    Parameters
    ----------
    name : string, default "AnomalyDetector.pickle"
        The filename to save to.
    path : Path, default os.getcwd()
        The path you want to save the pickle to.
    """
    with open(path + name, "wb") as file:
        pickle.dump(self, file)


def load(name="AnomalyDetector.pkl", path=os.getcwd() + "\\"):
    """
    Load the AnomalyDetector from a pickle.

    Parameters
    ----------
    name : string, default "AnomalyDetector.pickle"
        The filename to load from.
    path : Path, default os.getcwd()
        The path you want to load the pickle from.
    """
    with (open(path + name, "rb")) as file:
        detector = pickle.load(file)

    return detector
