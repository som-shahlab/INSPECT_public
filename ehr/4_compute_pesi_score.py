import argparse
import os
import pickle
from typing import List, Tuple, Optional
import multiprocessing
from collections import Counter
import datetime
import collections

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    LogisticRegressionCV,
)
from sklearn.preprocessing import MaxAbsScaler
from loguru import logger
from utils import load_data, save_data
import collections
from sklearn.model_selection import GridSearchCV, PredefinedSplit, ParameterGrid
from scipy.sparse import issparse
import scipy
import csv
import lightgbm as ltb

import femr
import femr.datasets
import femr.labelers.omop


def get_simplified_pesi_score(data) -> int:
    """
    Simplified PESI (Pulmonary Embolism Severity Index)

    Predicts 30-day outcome of patients with PE, with fewer criteria than the original PESI.
    https://www.mdcalc.com/calc/1247/simplified-pesi-pulmonary-embolism-severity-index

    Input is a data dictionary containing these variables:
    - age: int          Age in years
    - pmh_cancer: bool  History of cancer
    - pmh_ccd: bool     History of chronic cardiopulmonary disease
    - heart_rate: int   Heart rate, bpm
    - systolic_bp: int  Systolic BP, mmHg
    - o2_sat: float     O₂ saturation

    returns int score in interval [0,6]
    """
    # assert all variables are present
    names = {"age", "pmh_cancer", "pmh_ccd", "heart_rate", "systolic_bp", "o2_sat"}
    assert names.issubset(data.keys())

    score = 0
    if data["age"] > 80:
        score += 1
    if data["pmh_cancer"]:
        score += 1
    if data["pmh_ccd"]:
        score += 1
    if data["heart_rate"] >= 110:
        score += 1
    if data["systolic_bp"] < 100:
        score += 1
    if data["o2_sat"] < 90:
        score += 1
    return score


start = datetime.timedelta(days=10)
end = datetime.timedelta(days=2)


def compute_pesi(args):
    path_to_database, patient_ids, label_times = args
    values = []

    database = femr.datasets.PatientDatabase(path_to_database)
    ontology = database.get_ontology()

    cancer_codes = list(
        femr.labelers.omop.map_omop_concept_codes_to_femr_codes(
            ontology, {"SNOMED/55342001"}, is_ontology_expansion=True
        )
    )
    cardiopulmonary_codes = list(
        femr.labelers.omop.map_omop_concept_codes_to_femr_codes(
            ontology, {"SNOMED/39785005"}, is_ontology_expansion=True
        )
    )
    o2_sat_codes = list(
        femr.labelers.omop.map_omop_concept_codes_to_femr_codes(
            ontology, {"SNOMED/104847001"}, is_ontology_expansion=True
        )
    ) + ["LOINC/20564-1", "LOINC/2713-6"]

    cancer_codes = set(cancer_codes)
    cardiopulmonary_codes = set(cardiopulmonary_codes)
    o2_sat_codes = set(o2_sat_codes)

    missing_count = collections.defaultdict(int)

    for i, (patient_id, label_time) in enumerate(zip(patient_ids, label_times)):
        if i % 1000 == 0:
            print(i, len(patient_ids))

        min_time = label_time - start
        label_time = label_time + end
        patient = database[patient_id]
        data = {}
        data["age"] = (label_time - patient.events[0].start).days / 365
        data["pmh_ccd"] = False
        data["pmh_cancer"] = False

        for event in patient.events:
            if event.start > label_time:
                break

            if event.code in cardiopulmonary_codes:
                data["pmh_ccd"] = True

            if event.code in cancer_codes:
                data["pmh_cancer"] = True

            if event.start > min_time:
                if event.code == "LOINC/8867-4":
                    data["heart_rate"] = event.value

                if event.code == "LOINC/8480-6":
                    data["systolic_bp"] = event.value

                if event.code in o2_sat_codes and event.value is not None:
                    data["o2_sat"] = event.value

        names = {"age", "pmh_cancer", "pmh_ccd", "heart_rate", "systolic_bp", "o2_sat"}
        if not all(n in data for n in names):
            for n in names:
                if n not in data:
                    missing_count[n] += 1
            values.append(float("nan"))
        else:
            values.append(get_simplified_pesi_score(data))

    return values, missing_count


def main(args):
    os.mkdir(args.path_to_output_dir)

    patient_ids = []
    label_times = []
    values = []

    with open(os.path.join(args.path_to_cohort)) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["pe_positive_nlp"] != "True":
                continue
            patient_ids.append(int(row["patient_id"]))
            label_times.append(datetime.datetime.fromisoformat(row["procedure_time"]))

    patient_ids = np.array(patient_ids)
    label_times = np.array(label_times)

    indices = list(range(len(patient_ids)))

    result_counts = collections.defaultdict(int)

    task_indices = np.array_split(indices, args.num_threads)

    tasks = []
    for i in range(args.num_threads):
        tasks.append(
            (
                args.path_to_database,
                patient_ids[task_indices[i]],
                label_times[task_indices[i]],
            )
        )

    with multiprocessing.Pool(args.num_threads) as pool:
        for v, c in pool.imap(compute_pesi, tasks):
            values.extend(v)
            for n, count in c.items():
                result_counts[n] += count

    values = np.array(values)
    print(start, end)
    print(result_counts)
    print("Num cases", np.sum(values == values))
    print("Num patients", len(set(patient_ids[values == values])))

    unique, counts = np.unique(values, return_counts=True)
    print("sPESI distribution", dict(zip(unique, counts)))

    with open(os.path.join(args.path_to_output_dir, "pesi_scores.pkl"), "wb") as f:
        pickle.dump([values, patient_ids, label_times], f)

    logger.success("DONE!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PESI score")
    parser.add_argument(
        "--path_to_cohort", required=True, type=str, help="Path to femr database"
    )
    parser.add_argument(
        "--path_to_database", required=True, type=str, help="Path to femr database"
    )
    parser.add_argument(
        "--path_to_output_dir",
        required=True,
        type=str,
        help="Path to save labeles and featurizers",
    )
    parser.add_argument(
        "--num_threads",
        required=True,
        type=int,
        help="Path to save labeles and featurizers",
    )

    args = parser.parse_args()
    main(args)
