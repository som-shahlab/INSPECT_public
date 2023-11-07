import argparse
import os

"""
example to run:

python run_all_ehr.py \
--path_to_data /share/pi/nigam/projects/zphuo/data/PE/inspect \
--path_to_output /share/pi/nigam/projects/zphuo/data/PE/inspect/output \
--path_to_athena /share/pi/nigam/projects/zphuo/data/omop_extract_deid/athena_download \
--path_to_motor /share/pi/nigam/projects/zphuo/models/motor_release 

"""

parser = argparse.ArgumentParser(
    description="Run all the EHR experiments for the INSPECT paper",
)
parser.add_argument("--path_to_data", type=str, required=True)
parser.add_argument("--path_to_output", type=str, required=True)
parser.add_argument("--path_to_athena", type=str, required=True)
parser.add_argument("--path_to_motor", type=str, required=True)
parser.add_argument(
    "--test_GBM",
    default=False,
    action="store_true",
    help="if you want to run GBM for test split only",
)

args = parser.parse_args()

source_csvs_path = os.path.join(args.path_to_data, "timelines_smallfiles")
cohort_path = os.path.join(args.path_to_data, "cohort_0.2.0_master_file_anon.csv")
database_path = os.path.join(args.path_to_output, "inspect_femr_extract")

tasks = [
    "12_month_mortality",
    "6_month_mortality",
    "1_month_mortality",
    "1_month_readmission",
    "6_month_readmission",
    "12_month_readmission",
    "12_month_PH",
    "PE",
]


extract_path = os.path.join(database_path, "extract")
label_features_dirs = os.path.join(args.path_to_output, "labels_and_features")
gbm_model_results = os.path.join(args.path_to_output, "gbm_model_results")
motor_batches = os.path.join(args.path_to_output, "MOTOR_batches")

"""
# 1. Generate database

if not os.path.exists(database_path):
    os.system(
        f"python 1_csv_to_database.py --path_to_input {source_csvs_path} --path_to_target {database_path} --athena_download {args.path_to_athena} --num_threads 20"
    )
"""

# 2. Generate labels and features

os.makedirs(label_features_dirs, exist_ok=True)

for task in tasks:
    task_dir = os.path.join(label_features_dirs, task)
    if not os.path.exists(task_dir):
        os.system(
            f"python 2_generate_labels_and_features.py --path_to_cohort {cohort_path} --path_to_database {extract_path} --path_to_output_dir {task_dir} --labeling_function {task} --num_threads 20"
        )
    else:
        print(
            f"Skipping generating labels/features for {task} because it already exists in {task_dir}"
        )

# 3. Run GBM models
os.makedirs(gbm_model_results, exist_ok=True)

for task in tasks:
    task_labels_features_dir = os.path.join(label_features_dirs, task)
    task_dir = os.path.join(gbm_model_results, task)
    if args.test_GBM:
        os.system(
            f"python 3_train_gbm.py --path_to_cohort {cohort_path} --path_to_database {extract_path} --path_to_output_dir {task_dir} --path_to_label_features {task_labels_features_dir} --num_threads 20 --test_GBM"
        )
    elif not os.path.exists(task_dir) and not args.test_GBM:
        os.system(
            f"python 3_train_gbm.py --path_to_cohort {cohort_path} --path_to_database {extract_path} --path_to_output_dir {task_dir} --path_to_label_features {task_labels_features_dir} --num_threads 20"
        )
    else:
        print(f"Skipping GRM {task} because it already exists in {task_dir}")


# 3.1. Generate MOTOR batches
os.makedirs(motor_batches, exist_ok=True)

for task in tasks:
    task_labels_features_dir = os.path.join(label_features_dirs, task)
    task_dir = os.path.join(motor_batches, task)
    if not os.path.exists(task_dir):
        command = f"clmbr_create_batches {task_dir} --data_path {extract_path} --task labeled_patients --labeled_patients_path {task_labels_features_dir}/labeled_patients.csv --val_start 80 --dictionary_path {args.path_to_motor}/dictionary --is_hierarchical"
        os.system(command)
    else:
        print(f"Skipping MOTOR batches {task} because it already exists in {task_dir}")


# 3.2 Train logistic regression models on MOTOR representations
motor_results = os.path.join(args.path_to_output, "motor_results")
os.makedirs(motor_results, exist_ok=True)

for task in tasks:
    task_batches_dir = os.path.join(motor_batches, task)
    task_dir = os.path.join(motor_results, task)
    if not os.path.exists(task_dir):
        command = f"clmbr_train_linear_probe {task_dir} --data_path {extract_path} --model_dir {args.path_to_motor}/model --batches_path {task_batches_dir} "
        os.system(command)
    else:
        print(
            f"Skipping MOTOR linear probe {task} because it already exists in {task_dir}"
        )

# 4. Compute simplified PESI

pesi_folder = os.path.join(args.path_to_output, "simplified_pesi")
if not os.path.exists(pesi_folder):
    os.system(
        f"python 4_compute_pesi_score.py --path_to_cohort {cohort_path} --path_to_database {extract_path} --path_to_output_dir {pesi_folder} --num_threads 20"
    )
else:
    print(f"Skipping PESI computation because it already exists in {pesi_folder}")
