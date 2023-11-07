import os
import math
import sklearn.metrics
import pickle
import numpy as np
import sklearn.linear_model
import argparse
import scipy
import collections
import datetime
import csv

parser = argparse.ArgumentParser(
    description="Get model performance and simplified pesi score performance",
)
parser.add_argument("--path_to_data", type=str, required=True)
parser.add_argument("--path_to_output", type=str, required=True)
parser.add_argument("--compare_vs_pesi", action=argparse.BooleanOptionalAction)
parser.add_argument("--require_pe_status", type=str, required=False, default=None)
parser.add_argument("--compute_confidence", action=argparse.BooleanOptionalAction)
parser.add_argument("--relative_confidence", action=argparse.BooleanOptionalAction)
parser.add_argument("--metric", type=str, default="auroc")

args = parser.parse_args()

if args.require_pe_status is not None:
    args.require_pe_status = args.require_pe_status.lower() == "true"

tasks = (
    "PE",
    "1_month_mortality",
    "6_month_mortality",
    "12_month_mortality",
    "1_month_readmission",
    "6_month_readmission",
    "12_month_readmission",
    "12_month_PH",
)
ct_names = (
    "pe",
    "mort_1m",
    "mort_6m",
    "mort_12m",
    "read_1m",
    "read_6m",
    "read_12m",
    "ph_12m",
)

ct_name_map = {k: v for k, v in zip(tasks, ct_names)}

if args.require_pe_status != None:
    tasks = tasks[1:]

performances = {}

has_pe = {}
pid_split_assignment = {}
pesi_map = {}

with open(os.path.join(args.path_to_data, "cohort_0.2.0_master_file.csv")) as f:
    reader = csv.DictReader(f)
    for row in reader:
        pid = int(row["patient_id"])
        pid_split_assignment[pid] = row["split"]
        time = datetime.datetime.fromisoformat(row["procedure_time"])
        has_pe[(pid, time)] = row["pe_positive_nlp"] == "True"
        pesi_map[(pid, time)] = float("nan")


with open(f"{args.path_to_output}/simplified_pesi/pesi_scores.pkl", "rb") as f:
    pesi_predictions, pesi_patient_ids, pesi_times = pickle.load(f)

for prediction, patient_id, time in zip(pesi_predictions, pesi_patient_ids, pesi_times):
    pesi_map[(patient_id, time)] = prediction

print("Have pesi scores for ", sum(v == v for k, v in pesi_map.items()), " labels")

from itertools import chain, combinations


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def load_data(name, task, split):
    if name == "GBM":
        with open(
            f"{args.path_to_output}/gbm_model_results/{task}/predictions.pkl", "rb"
        ) as f:
            result = pickle.load(f)
    elif name == "MOTOR":
        with open(
            f"{args.path_to_output}/motor_results/{task}/predictions.pkl", "rb"
        ) as f:
            result = pickle.load(f)
    elif name == "sPESI":
        baseline = load_data("GBM", task, split)
        for i, (pid, time) in enumerate(zip(baseline[1], baseline[3])):
            if task == "PE":
                time = time + datetime.timedelta(days=1)
            baseline[0][i] = scipy.special.expit(pesi_map[(pid, time)])
        return baseline
    elif name == "CT":
        ct_name = ct_name_map[task]
        pids = []
        prediction_dates = []
        labels = []
        probs = []

        with open(f"{args.path_to_output}/ct_results/{ct_name}_pred_proba.csv") as f:
            reader = csv.DictReader(f)
            for line in reader:
                pid = int(line["patient_id"])
                prediction_date = datetime.datetime.fromisoformat(
                    line["procedure_time"]
                )

                if task == "PE":
                    prediction_date = prediction_date - datetime.timedelta(days=1)

                label = float(line["label"]) == 1
                prob = float(line["prob"])
                pids.append(pid)
                prediction_dates.append(prediction_date)
                labels.append(label)
                probs.append(prob)

        result = [probs, pids, labels, prediction_dates]

    if not isinstance(result[3][0], datetime.datetime):
        result[3] = result[3].astype(datetime.datetime)

    o = np.lexsort((result[1], result[3]))
    result = [np.array(v)[o] for v in result]
    valid_indices = []

    for i, (pid, time) in enumerate(zip(result[1], result[3])):
        if task == "PE":
            time = time + datetime.timedelta(days=1)
        good = pid_split_assignment[pid] == split
        if args.compare_vs_pesi:
            if pesi_map[(pid, time)] != pesi_map[(pid, time)]:
                good = False

        if args.require_pe_status is not None:
            if has_pe[(pid, time)] != args.require_pe_status:
                good = False

        if good:
            valid_indices.append(i)
    result = [a[valid_indices] for a in result]

    return result


baseline_model = "MOTOR"

baseline_performance = {}


def get_performance(labels, probabilities):
    if args.metric == "auroc":
        return sklearn.metrics.roc_auc_score(labels, probabilities)
    elif args.metric == "auprc":
        return sklearn.metrics.average_precision_score(labels, probabilities)
    elif args.metric == "ece":
        bins = np.linspace(0, 1, num=10)
        start_bin = bins[:-1]
        end_bin = bins[1:]

        total = 0

        for start, end in zip(start_bin, end_bin):
            total_true = 0
            total_prob = 0
            num_in_bin = 0
            for label, prob in zip(labels, probabilities):
                if start <= prob < end:
                    total_true += label
                    total_prob += prob
                    num_in_bin += 1
            if num_in_bin != 0:
                total += num_in_bin * abs(
                    total_true / num_in_bin - total_prob / num_in_bin
                )

        return total / len(labels)
    else:
        assert False, f"{args.metric} is not a valid metric"


if args.compute_confidence:
    for task in tasks:
        predictions, pids, labels, times = load_data(baseline_model, task, "test")

        samples = []
        for _ in range(1000):
            indices = sklearn.utils.resample(list(range(len(predictions))))
            l_s, p_s = labels[indices], predictions[indices]
            performance = get_performance(l_s, p_s)
            samples.append((indices, performance))

        baseline_performance[task] = samples


def compute_score_and_bootstrap(task, labels, predictions):
    total_score = get_performance(labels, predictions)

    if args.compute_confidence:
        samples = []
        for indices, baseline_perf in baseline_performance[task]:
            l_s, p_s = labels[indices], predictions[indices]
            perf = get_performance(l_s, p_s)
            if args.relative_confidence:
                perf -= baseline_perf
            samples.append(perf)
        confidence = np.quantile(samples, (0.025, 0.975))
    else:
        confidence = None

    return total_score, confidence


def get_model_performance(task, name):
    predictions, pids, labels, times = load_data(name, task, "test")

    return compute_score_and_bootstrap(task, labels, predictions)


def get_combined_model_performance(task, names):
    pids = {}
    times = {}
    labels = {}
    probs = {}

    for split in ["valid", "test"]:
        for name in names:
            k = (name, split)
            probs[k], pids[k], labels[k], times[k] = load_data(name, task, split)

            if name != names[0]:
                assert len(pids[(names[0], split)]) == len(
                    pids[(name, split)]
                ), f"{len(pids[(names[0], split)])} {len(pids[(name, split)])}"

                assert np.all(pids[(names[0], split)] == pids[(name, split)])
                assert np.all(times[(names[0], split)] == times[(name, split)])
                assert np.all(labels[(names[0], split)] == labels[(name, split)])

    all_probs = np.stack([probs[(name, "valid")] for name in names], axis=-1)
    features = scipy.special.logit(all_probs)
    features = np.clip(features, -1e10, 1e10)

    model = sklearn.linear_model.LogisticRegression(penalty=None, solver="newton-cg")

    model.fit(features, labels[(names[0], "valid")])

    if np.any(np.isnan(model.coef_)):
        # Could not converge. Fallback to average
        model.coef_ = np.ones_like(model.coef_)
        model.intercept_ = 0

    test_probs = np.stack([probs[(name, "test")] for name in names], axis=-1)
    test_features = scipy.special.logit(test_probs)
    test_features = np.clip(test_features, -1e10, 1e10)

    computed_probs = model.predict_proba(test_features)[:, 1]

    return compute_score_and_bootstrap(task, labels[(names[0], "test")], computed_probs)


# models = ["CT", "MOTOR", "GBM"]
models = ["MOTOR", "GBM"]
if args.compare_vs_pesi:
    models.append("sPESI")

aurocs = collections.defaultdict(dict)
fusion_aurocs = collections.defaultdict(dict)

best_perfs = {}
best_m_perfs = {}


def compare(a, b):
    if args.metric == "ece":
        return a > b
    else:
        return a < b


for task in tasks:
    for model in models:
        aurocs[model][task] = get_model_performance(task, model)
        if not args.compare_vs_pesi:
            if task not in best_perfs or compare(
                best_perfs[task][0][0], aurocs[model][task][0]
            ):
                best_perfs[task] = aurocs[model][task], model
        if task not in best_m_perfs or compare(
            best_m_perfs[task][0][0], aurocs[model][task][0]
        ):
            best_m_perfs[task] = aurocs[model][task], model

    if not args.compare_vs_pesi:
        for names in powerset(models):
            if len(names) > 1:
                fusion_aurocs[names][task] = get_combined_model_performance(task, names)

                if task not in best_m_perfs or compare(
                    best_m_perfs[task][0][0], fusion_aurocs[names][task][0]
                ):
                    best_m_perfs[task] = fusion_aurocs[names][task], names


print(aurocs)

print("MAIN TABLE")

for model, performances in aurocs.items():
    row = " "
    for i, model_column in enumerate(models):
        if i != 0:
            row += " &"
        if model == model_column:
            row += " $\\checkmark$"
    for task in tasks:
        row += "& "
        value = f"{performances[task][0]:0.3f} "
        if task in best_perfs and model == best_perfs[task][1]:
            value = f"\\underline{{" + value + f"}}"
        if task in best_m_perfs and model == best_m_perfs[task][1]:
            value = f"\\textbf{{" + value + f"}}"

        row += value
    row += "\\\\"
    print(row)


if not args.compare_vs_pesi:
    print("\\midrule")

    for eval_models, performances in fusion_aurocs.items():
        row = " "
        for i, model_column in enumerate(models):
            if i != 0:
                row += " &"
            if model_column in eval_models:
                row += "$\\checkmark$"
        for task in tasks:
            row += "& "
            if eval_models == best_m_perfs[task][1]:
                row += f"\\textbf{{" + f"{performances[task][0]:0.3f}" + f"}}"
            else:
                row += f"{performances[task][0]:0.3f} "
        row += "\\\\"
        print(row)

if args.compute_confidence:
    print("CONFIDENCE INTERVAL TABLE")

    for model, performances in aurocs.items():
        row = " "
        for i, model_column in enumerate(models):
            if i != 0:
                row += " &"
            if model == model_column:
                row += " $\\checkmark$"
        for task in tasks:
            row += "& "
            text = f"({performances[task][1][0]:0.2f}, {performances[task][1][1]:0.2f})"
            if args.relative_confidence and (
                all(v < 0 for v in performances[task][1])
                or all(v > 0 for v in performances[task][1])
            ):
                text = f"\\textbf{{" + text + f"}}"
            row += text
        row += "\\\\"
        print(row)

    print("\\midrule")

    for eval_models, performances in fusion_aurocs.items():
        row = " "
        for i, model_column in enumerate(models):
            if i != 0:
                row += " &"
            if model_column in eval_models:
                row += "$\\checkmark$"
        for task in tasks:
            row += "& "
            text = f"({performances[task][1][0]:0.2f}, {performances[task][1][1]:0.2f})"
            if args.relative_confidence and (
                all(v < 0 for v in performances[task][1])
                or all(v > 0 for v in performances[task][1])
            ):
                text = f"\\textbf{{" + text + f"}}"
            row += text
        row += "\\\\"
        print(row)
