import pickle
import json
import os
from typing import List
import os
import struct
import hashlib
import ast
import warnings
from loguru import logger
import pandas as pd
from femr.datasets import PatientDatabase
import datetime
import random
import collections
import femr


def save_data(data, filename):
    """
    Saves Python object to either pickle or JSON file, depending on file extension.

    Parameters:
    data (object): The Python object to be saved.
    filename (str): The name of the file to save to, including the extension.
    """

    # Determine file extension
    file_extension = filename.split(".")[-1]

    # Save to pickle file if extension is .pkl
    if file_extension == "pkl":
        with open(filename, "wb") as f:
            pickle.dump(data, f)
    # Save to JSON file if extension is .json
    elif file_extension == "json":
        with open(filename, "w") as f:
            json.dump(data, f)
    # Dump it to pickle
    else:
        warnings.warn("There is no file extension, so saving it as pickle")
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        # raise ValueError("Unsupported file extension. Only .pkl and .json are supported.")


def load_data(filename):
    """
    Loads Python object from either pickle or JSON file, depending on file extension.

    Parameters:
    filename (str): The name of the file to load from, including the extension.

    Returns:
    The loaded Python object.
    """

    # Determine file extension
    file_extension = filename.split(".")[-1]

    # Load from pickle file if extension is .pkl
    if file_extension == "pkl":
        with open(filename, "rb") as f:
            return pickle.load(f)
    # Load from JSON file if extension is .json
    elif file_extension == "json":
        with open(filename, "r") as f:
            return json.load(f)
    # Raise error if file extension is not supported
    else:
        warnings.warn("There is no file extension, so loading it as pickle")
        with open(filename, "rb") as f:
            return pickle.load(f)
        # raise ValueError("Unsupported file extension. Only .pkl and .json are supported.")


def get_labels_directly_csv(cohort_lines, PATIENT_ID_COLUMN, TIME_COLUMN, label_column):
    labels = collections.defaultdict(list)
    for row in cohort_lines:
        labels[row[PATIENT_ID_COLUMN]].append(
            femr.labelers.Label(
                time=row[TIME_COLUMN] - datetime.timedelta(days=1),
                value=row[label_column] == "True",
            )
        )

    for _, v in labels.items():
        v.sort(key=lambda a: a.time)

    labeled_patients = femr.labelers.LabeledPatients(labels, "boolean")

    return labeled_patients
