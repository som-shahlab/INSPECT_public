import os
import argparse
from loguru import logger
import femr.datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run femr text featurizer")
    parser.add_argument(
        "--path_to_input",
        type=str,
        help="Path to folder with all the csv files",
    )
    parser.add_argument(
        "--path_to_target",
        type=str,
        help="Path to your target directory to save femr",
    )
    parser.add_argument(
        "--athena_download",
        type=str,
        help="Path to athena download",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Number of threads",
    )

    args = parser.parse_args()
    INPUT_DIR = args.path_to_input
    TARGET_DIR = args.path_to_target
    athena_download = args.athena_download
    num_threads = args.num_threads
    
    os.makedirs(TARGET_DIR)

    LOG_DIR = os.path.join(TARGET_DIR, "logs")
    EXTRACT_DIR = os.path.join(TARGET_DIR, "extract")

    os.system(f"etl_simple_femr {INPUT_DIR} {EXTRACT_DIR} {LOG_DIR} --num_threads {num_threads} --athena_download {athena_download}")

    logger.info(f"Femr database saved in path: {TARGET_DIR}")
    logger.info("Testing the database")

    database = femr.datasets.PatientDatabase(EXTRACT_DIR)
    logger.info("Num patients", len(database))
    all_patient_ids = list(database)
    omop_id = all_patient_ids[0]
    patient = database[omop_id]
    events = patient.events
    logger.info(f"Number of events in patients with omop_id {omop_id}: {events}")
    logger.info(f"First event of the patient: {events[0]}")
