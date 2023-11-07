import os
import csv
import femr.labelers
import collections
import argparse

parser = argparse.ArgumentParser(
    description='Get statistics about the labeling functions',
)
parser.add_argument('--path_to_data', type=str, required=True)
parser.add_argument('--path_to_output', type=str, required=True)

args = parser.parse_args()

tasks = ("PE", "1_month_mortality", "6_month_mortality", "12_month_mortality", "1_month_readmission", "6_month_readmission", "12_month_readmission", "12_month_PH")

def get_category(task):
    if 'PE' in task:
        return 'PE'
    elif 'mortality' in task:
        return 'Mort'
    elif 'readmission' in task:
        return 'Re-ad'
    else:
        return 'PH'

def get_subtype(task):
    if '1_month' in task:
        return '1 m'
    elif '6_month' in task:
        return '6 m'
    elif '12_month' in task:
        return '12 m'
    else:
        return 'N/A'

pid_split_assignment = {}

with open(os.path.join(args.path_to_data, 'inspect_cohort.csv')) as f:
    reader = csv.DictReader(f)
    for row in reader:
        pid_split_assignment[int(row['patient_id'])] = row['split']

raw_totals = {}

for category in ('PE', 'Mort', 'Re-ad', 'PH'):
    cat_tasks = [task for task in tasks if get_category(task) == category]

    rows = []
    for task_i, task in enumerate(cat_tasks):
        labels = femr.labelers.load_labeled_patients(f'{args.path_to_output}/labels_and_features/{task}/labeled_patients.csv')
        totals = collections.defaultdict(int)
        positives = collections.defaultdict(int)
        for pid, ls in labels.items():
            split = pid_split_assignment[pid]
            for l in ls:
                totals[split] += 1
                positives[split] += l.value

        if category == 'PE':
            raw_totals = totals

        subrows = []
        for value in ('pos.', 'neg.', 'cen.'):
            row = '' 
            row += f'& {value}'

            if value == 'pos.':
                counts = positives
            elif value == 'neg.':
                counts = {k: totals[k] - positives[k] for k in positives}
            else:
                counts = {k: raw_totals[k] - totals[k] for k in positives}

            if sum(counts.values()) == 0:
                continue

            row += f' & {sum(counts.values()):,}'

            for split in ('train', 'valid', 'test'):
                row += f'& {counts[split]:,} & ({ 100 * counts[split] / raw_totals[split]:0.1f} \%)'

            subrows.append(row + ' \\\\')
        for i, row in enumerate(subrows):
            if i == 0:
                row = ' & ' + '\\multirow{' + str(len(subrows)) + '}{*}{ ' + get_subtype(task)  + '}' + row
            else:
                row = ' & ' + row

            if i == len(subrows) - 1 and task_i != len(cat_tasks) - 1:
                row = row + ' \n \\cline{2-10}'

            subrows[i] = row

        rows.extend(subrows)

    for i, row in enumerate(rows):
        if i == 0:
            row = '\\multirow{' + str(len(rows)) + '}{*}{\\textbf{' + category  + '}}' + row
        else:
            row = row

        if i == len(rows) - 1 and task != 'PH':
            row = row + ' \n \\midrule'

        rows[i] = row

    print('\n'.join(rows))
