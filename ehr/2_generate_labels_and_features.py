import argparse
import datetime
import os
import pickle
import json
import csv
import numpy as np
from loguru import logger
from utils import load_data, save_data
import pandas as pd
from typing import Any, Callable, List, Optional, Set, Tuple
import collections

import random
import femr
from femr.datasets import PatientDatabase
from femr.labelers.core import LabeledPatients
from femr.featurizers.core import FeaturizerList
from femr.featurizers.featurizers import AgeFeaturizer, CountFeaturizer
from femr.labelers.core import NLabelsPerPatientLabeler, TimeHorizon
from femr.labelers.omop import (
    MortalityCodeLabeler,
)

from femr.labelers.omop_inpatient_admissions import InpatientReadmissionLabeler

PH_codes = [
    a.strip()
    for a in """
1029634
I27.21
I27.22
416
416.1
416.2
I27.2
I27.29
I27.83
I27.0
416
416.0
416.8
I27.89
I27.82
1170535
I27.1
I27.20
I27.23
416.9
I27.81
I27.9
I27.9
I27.0
I27.1
67294
142308
I27.24
2065632
I27.2
""".split()
]


class CodeLabeler(femr.labelers.TimeHorizonEventLabeler):
    def __init__(
        self,
        cohort_rows,
        time_horizon,
        codes,
        ontology,
        offset=datetime.timedelta(days=0),
    ):
        self.codes = codes
        self.time_horizon = time_horizon

        self.prediction_times_map = collections.defaultdict(set)

        for row in cohort_rows:
            prediction_time: datetime.datetime = row[TIME_COLUMN]
            self.prediction_times_map[row[PATIENT_ID_COLUMN]].add(
                prediction_time - offset
            )

        super().__init__()

    def get_prediction_times(self, patient):
        return sorted(list(self.prediction_times_map[patient.patient_id]))

    def get_time_horizon(self):
        return self.time_horizon

    def allow_same_time_labels(self):
        return False

    def get_outcome_times(self, patient):
        outcome_times = set()

        for event in patient.events:
            if event.code in self.codes:
                outcome_times.add(event.start)

        return sorted(list(outcome_times))


class SourceCodeLabeler(femr.labelers.TimeHorizonEventLabeler):
    def __init__(self, cohort_rows, time_horizon, codes, ontology):
        self.codes = codes
        self.time_horizon = time_horizon

        self.prediction_times_map = collections.defaultdict(set)

        for row in cohort_rows:
            prediction_time: datetime.datetime = row[TIME_COLUMN]
            self.prediction_times_map[row[PATIENT_ID_COLUMN]].add(prediction_time)

        super().__init__()

    def get_prediction_times(self, patient):
        return sorted(list(self.prediction_times_map[patient.patient_id]))

    def get_time_horizon(self):
        return self.time_horizon

    def allow_same_time_labels(self):
        return False

    def get_outcome_times(self, patient):
        outcome_times = set()

        for event in patient.events:
            if (
                event.omop_table == "condition_occurrence"
                and event.source_code is not None
                and event.source_code in self.codes
            ):
                outcome_times.add(event.start)

        return sorted(list(outcome_times))


class ModifiedReadmission(InpatientReadmissionLabeler):
    def __init__(self, cohort_rows, time_horizon, ontology):
        self.prediction_times_map = collections.defaultdict(set)

        for row in cohort_rows:
            prediction_time: datetime.datetime = row[TIME_COLUMN]
            self.prediction_times_map[row[PATIENT_ID_COLUMN]].add(prediction_time)

        super().__init__(ontology, time_horizon)

    def get_prediction_times(self, patient):
        return sorted(list(self.prediction_times_map[patient.patient_id]))

    def allow_same_time_labels(self):
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run femr featurization")
    parser.add_argument(
        "--path_to_cohort", required=True, type=str, help="Path to cohort dataframe"
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
        "--labeling_function",
        required=True,
        type=str,
        help="Name of labeling function to create.",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        help="The number of threads to use",
        default=1,
    )

    args = parser.parse_args()

    PATIENT_ID_COLUMN = "PatientID"  # "patient_id"
    TIME_COLUMN = "StudyTime"  # "procedure_time"

    with open(args.path_to_cohort) as f:
        if args.path_to_cohort.endswith(".csv"):
            reader = csv.DictReader(f)
        elif args.path_to_cohort.endswith(".tsv"):
            reader = csv.DictReader(f, delimiter="\t")
        cohort_lines = list(reader)
        for row in cohort_lines:
            row[TIME_COLUMN] = datetime.datetime.fromisoformat(row[TIME_COLUMN])
            row[PATIENT_ID_COLUMN] = int(row[PATIENT_ID_COLUMN])

    PATH_TO_PATIENT_DATABASE = args.path_to_database
    PATH_TO_OUTPUT_DIR = args.path_to_output_dir
    NUM_THREADS: int = args.num_threads

    # Logging
    path_to_log_file: str = os.path.join(PATH_TO_OUTPUT_DIR, "info.log")
    if os.path.exists(path_to_log_file):
        os.remove(path_to_log_file)
    logger.add(path_to_log_file, level="INFO")  # connect logger to file
    logger.info(f"Labeling function: {args.labeling_function}")
    logger.info(f"Loading patient database from: {PATH_TO_PATIENT_DATABASE}")
    logger.info(f"Saving output to: {PATH_TO_OUTPUT_DIR}")
    logger.info(f"# of threads: {NUM_THREADS}")
    with open(os.path.join(PATH_TO_OUTPUT_DIR, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # create directories to save files
    PATH_TO_SAVE_LABELED_PATIENTS: str = os.path.join(
        PATH_TO_OUTPUT_DIR, "labeled_patients.csv"
    )
    PATH_TO_SAVE_PREPROCESSED_FEATURIZERS: str = os.path.join(
        PATH_TO_OUTPUT_DIR, "preprocessed_featurizers.pkl"
    )
    PATH_TO_SAVE_FEATURIZED_PATIENTS: str = os.path.join(
        PATH_TO_OUTPUT_DIR, "featurized_patients.pkl"
    )
    os.makedirs(PATH_TO_OUTPUT_DIR, exist_ok=True)

    # Load PatientDatabase + Ontology
    logger.info(f"Start | Load PatientDatabase")
    database = PatientDatabase(PATH_TO_PATIENT_DATABASE)
    ontology = database.get_ontology()
    logger.info(f"Finish | Load PatientDatabase")

    patient_ids = {row[PATIENT_ID_COLUMN] for row in cohort_lines}

    mortality_codes = list(
        femr.labelers.omop.map_omop_concept_codes_to_femr_codes(
            ontology,
            femr.labelers.omop.get_death_concepts(),
            is_ontology_expansion=True,
        )
    )

    if args.labeling_function == "12_month_mortality":
        labeler = CodeLabeler(
            cohort_lines,
            TimeHorizon(datetime.timedelta(minutes=1), datetime.timedelta(days=365)),
            mortality_codes,
            database.get_ontology(),
        )
    elif args.labeling_function == "6_month_mortality":
        labeler = CodeLabeler(
            cohort_lines,
            TimeHorizon(datetime.timedelta(minutes=1), datetime.timedelta(days=180)),
            mortality_codes,
            database.get_ontology(),
        )
    elif args.labeling_function == "1_month_mortality":
        labeler = CodeLabeler(
            cohort_lines,
            TimeHorizon(datetime.timedelta(minutes=1), datetime.timedelta(days=30)),
            mortality_codes,
            database.get_ontology(),
        )
    elif args.labeling_function == "1_month_readmission":
        labeler = ModifiedReadmission(
            cohort_lines,
            TimeHorizon(datetime.timedelta(minutes=1), datetime.timedelta(days=30)),
            database.get_ontology(),
        )
    elif args.labeling_function == "6_month_readmission":
        labeler = ModifiedReadmission(
            cohort_lines,
            TimeHorizon(datetime.timedelta(minutes=1), datetime.timedelta(days=180)),
            database.get_ontology(),
        )
    elif args.labeling_function == "12_month_readmission":
        labeler = ModifiedReadmission(
            cohort_lines,
            TimeHorizon(datetime.timedelta(minutes=1), datetime.timedelta(days=365)),
            database.get_ontology(),
        )
    elif args.labeling_function == "12_month_PH":
        # We have to disable this since we aren't currently releasing source codes due to PHI issues.
        # We hope to re-enable this shortly in the future
        # labeler = SourceCodeLabeler(
        #     cohort_lines,
        #     femr.labelers.TimeHorizon(
        #         datetime.timedelta(days=1), datetime.timedelta(days=365)
        #     ),
        #     PH_codes,
        #     database.get_ontology(),
        # )

        labeler = None
        labels = collections.defaultdict(list)
        for row in cohort_lines:
            label = row[args.labeling_function]
            if label == "Censored":
                continue

            labels[row[PATIENT_ID_COLUMN]].append(
                femr.labelers.Label(time=row[TIME_COLUMN], value=label == "True")
            )

        for _, v in labels.items():
            v.sort(key=lambda a: a.time)

        labeled_patients = femr.labelers.LabeledPatients(labels, "boolean")
    elif args.labeling_function == "PE":
        labeler = None
        labels = collections.defaultdict(list)
        for row in cohort_lines:
            labels[row[PATIENT_ID_COLUMN]].append(
                femr.labelers.Label(
                    time=row[TIME_COLUMN] - datetime.timedelta(days=1),
                    value=row["pe_positive_nlp"] == "True",
                )
            )

        for _, v in labels.items():
            v.sort(key=lambda a: a.time)

        labeled_patients = femr.labelers.LabeledPatients(labels, "boolean")
    else:
        raise ValueError(
            f"Labeling function `{args.labeling_function}` not supported.."
        )

    # Determine how many labels to keep per patient
    logger.info(f"Start | Label {len(patient_ids)} patients")

    if labeler is not None:
        labeled_patients = labeler.apply(
            path_to_patient_database=PATH_TO_PATIENT_DATABASE,
            num_threads=NUM_THREADS,
            patient_ids=patient_ids,
        )

    labeled_patients.save(PATH_TO_SAVE_LABELED_PATIENTS)
    logger.info("Finish | Label patients")
    logger.info(
        "LabeledPatient stats:\n"
        f"Total # of patients = {labeled_patients.get_num_patients()}\n"
        f"Total # of patients with at least one label = {labeled_patients.get_num_patients()}\n"
        f"Total # of labels = {labeled_patients.get_num_labels()}"
    )

    # Lets use both age and count featurizer
    age = AgeFeaturizer()
    count = CountFeaturizer(is_ontology_expansion=True)
    featurizer_age_count = FeaturizerList([age, count])

    # Preprocessing the featurizers, which includes processes such as normalizing age.
    logger.info("Start | Preprocess featurizers")
    featurizer_age_count.preprocess_featurizers(
        PATH_TO_PATIENT_DATABASE, labeled_patients, NUM_THREADS
    )
    save_data(featurizer_age_count, PATH_TO_SAVE_PREPROCESSED_FEATURIZERS)
    logger.info("Finish | Preprocess featurizers")

    logger.info("Start | Featurize patients")
    results = featurizer_age_count.featurize(
        PATH_TO_PATIENT_DATABASE, labeled_patients, NUM_THREADS
    )
    save_data(results, PATH_TO_SAVE_FEATURIZED_PATIENTS)
    logger.info("Finish | Featurize patients")
    feature_matrix, patient_ids, label_values, label_times = (
        results[0],
        results[1],
        results[2],
        results[3],
    )
    label_set, counts_per_label = np.unique(label_values, return_counts=True)
    logger.info(
        "FeaturizedPatient stats:\n"
        f"feature_matrix={repr(feature_matrix)}\n"
        f"patient_ids={repr(patient_ids)}\n"
        f"label_values={repr(label_values)}\n"
        f"label_set={repr(label_set)}\n"
        f"counts_per_label={repr(counts_per_label)}\n"
        f"label_times={repr(label_times)}"
    )

    with open(os.path.join(PATH_TO_OUTPUT_DIR, "done.txt"), "w") as f:
        f.write("done")

    logger.info("Done!")
