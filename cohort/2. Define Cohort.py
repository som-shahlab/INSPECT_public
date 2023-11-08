#!/usr/bin/env python
# coding: utf-8

# # Format CTPA cohort 2016-2022
# 
# ### Required files
# - https://stanfordmedicine.app.box.com/file/1086939442645?s=dyyvma87ppxzenbeye45z2er7yszwjml
# - aimi-nero-phi-mlungren-msicphi.ctpe_mars_datapull.ctpe_Radreports

# In[ ]:


import pandas as pd
import pickle
import datetime
import os 
import pandas as pd
import numpy as np
import collections
import numpy as np
import math

from datetime import datetime 
from operator import itemgetter
from google.cloud import bigquery
from collections import Counter, defaultdict


# In[ ]:


import os 

USER        = "mschuang@stanford.edu"
CREDENTIALS = os.path.expanduser(f"~/.config/gcloud/legacy_credentials/{USER}/adc.json")
PROJECT     = "som-nero-phi-nigam-starr"
DATASET     = "shahlab_omop_cdm5_subset_2023_03_05"

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = CREDENTIALS
os.environ['GCLOUD_PROJECT'] = PROJECT


# In[ ]:


import os
import ctypes
import hashlib
from google.cloud import bigquery

def load_table_from_dataframe(client, dataframe, table_id):
    
    # infer schema from dataframe
    bq_enums = {"object":"STRING", "int64":"INTEGER"}
    schema = [
        bigquery.SchemaField(name, bq_enums[str(dtype)]) 
        for name, dtype in zip(dataframe.columns, dataframe.dtypes)
    ]

    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition="WRITE_TRUNCATE" #overwrite table if exists
    )

    job = client.load_table_from_dataframe(dataframe, table_id, job_config=job_config)
    job.result()
    
    # validate table loaded
    table = client.get_table(table_id)
    print(
        "Loaded {} rows and {} columns to {}".format(
            table.num_rows, len(table.schema), table_id
        )
    )

def csv_to_bq_table(fpath, project_id, dataset_id, table_id):
    
    bq_table = f"{project_id}.{dataset_id}.{table_id}"

    try:
        client.create_table(bq_table)

        dataset_ref = client.dataset(dataset_id)
        table_ref = dataset_ref.table(table_id)
        job_config = bigquery.LoadJobConfig()
        job_config.source_format = bigquery.SourceFormat.CSV
        job_config.autodetect = True

        # load the csv into bigquery
        with open(fpath, "rb") as source_file:
            job = client.load_table_from_file(source_file, table_ref, job_config=job_config)
        job.result()
    except Exception as e:
        print(f"{e}")
        
    return bq_table

def query_starr_omop(query, 
                     client, 
                     show_progress_bar=True, 
                     use_cache=True, 
                     cache_dir="../data/cache",
                     **kwargs):
    """
    Simple function for processing a query using BigQuery and caching the results.
    """
    hash_value = hashlib.sha256(bytes(query,'utf-8')).hexdigest()
    fpath = f"{cache_dir}/{hash_value}.tsv"
    dtypes=kwargs['dtypes'] if "dtypes" in kwargs else None
    
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    if use_cache and os.path.exists(fpath):
        #df = pd.read_csv(fpath, sep='\t', dtype=dtypes)
        print(f"Loaded from cache {hash_value}")
    else:
        job = client.query(query)
        df = job.to_dataframe(progress_bar_type='tqdm')
        df.to_csv(fpath, sep='\t', index=False)
        
    df = pd.read_csv(fpath, sep='\t', dtype=dtypes)
    return df

client = bigquery.Client()


# ### Get all CTPA studies

# In[ ]:


# IMPORTANT: You must cast MRNs as strings otherwise join will fail to find some patients
FORCE_STR_MRN = True

dtypes = {'MRN':str, 'ORDER_ID':str, 'mrn':str} if FORCE_STR_MRN else {}

query = """\
    SELECT procs.*, JSON_QUERY(procs.trace_id, '$.order_proc_id') AS order_proc_id, CAST(SPLIT(person.person_source_value, " | ")[OFFSET(0)] AS STRING) as mrn
    FROM `som-nero-phi-nigam-starr.shahlab_omop_cdm5_subset_2023_03_05.procedure_occurrence` as procs,
    `som-nero-phi-nigam-starr.shahlab_omop_cdm5_subset_2023_03_05.person` as person
    WHERE \
    procs.person_id=person.person_id
    AND LOWER(procs.procedure_source_value) in (
      SELECT procedure_source_value FROM `som-nero-phi-nigam-starr.pulmonary_embolisms.ctpa_labels`
    )
"""
df_all = query_starr_omop(query, client, dtypes=dtypes)
df_all.head(5)


# In[ ]:


df_ = df_all.groupby('procedure_DATETIME')['person_id'].apply(list).to_frame()
for idx, row in df_.iterrows():
    if len(row.person_id) > 1:
        print(row.person_id)


# In[ ]:


df_all['person_id_date'] = df_all.apply(lambda x: f"{x.person_id}-{x.procedure_DATETIME}", axis=1)
d = df_all.person_id_date.value_counts().reset_index()
print(f"Numver of patient with overlaps: {d[d.person_id_date > 1].shape[0]}")


# In[ ]:


d = df_all.procedure_DATETIME.value_counts().reset_index()
print(f"Numbe of overlaps: {d[d.procedure_DATETIME > 1].shape[0]}")


# In[ ]:


df_all['mrn_date'] = df_all.apply(lambda x: f"{x.mrn}-{x.procedure_DATETIME}", axis=1)
d = df_all.mrn_date.value_counts().reset_index()
print(f"Numver of patient with overlaps: {d[d.mrn_date > 1].shape[0]}")


# # 2016-2022 Cohort
# 
# ### Load all studies

# In[ ]:


df = pd.read_csv('../data/ctpe_Radreports.csv')
df.head(5)

df_mapping = pd.read_csv('../data/crosswalk-2016-2022.csv', dtype={'mrn': object})
acc2mrn = dict(zip(df_mapping.accession, df_mapping.mrn))

df['mrn'] = df['accession_number'].apply(lambda x: acc2mrn[x] if x in acc2mrn else None)
print(f'Number of cases without MRN: {len(df[df.mrn.isna()])}/{len(df)}')

df = df[~df.mrn.isna()]
df = df.rename(columns={'mrn': 'MRN'})
df.head(5)


# ### Link study to STARR by finding closest procedure_datetime
# 
# Using ordering_date due to the smallest days offset: 
# 
# **ordering_date**
# 
# - Total studies:         18116
# 
# - Final Linked Orders (±10 days): 18058 
# 
# - Mean/SD Days Offset:            0.0 (0.0)
# 
# - Final Missing Orders:           58 
# 
# 
# **proc_start_time**
# 
# - Total studies:         18116
# 
# - Final Linked Orders (±10 days): 18057 
# 
# - Mean/SD Days Offset:            0.0 (0.1)
# 
# - Final Missing Orders:           59 
# 
# 
# 
# **proc_end_time**
# 
# - Total studies:         18116
# 
# - Final Linked Orders (±10 days): 18019 
# 
# - Mean/SD Days Offset:            0.1 (0.4)
# 
# - Final Missing Orders:           97 
# 
# 
# **result_time**
# 
# - Total studies:         18116
# 
# - Final Linked Orders (±10 days): 17888 
# 
# - Mean/SD Days Offset:            0.5 (1.0)
# 
# - Final Missing Orders:           228 
# 

# In[ ]:


# bucket our unmapped orders by MRN
order_index = collections.defaultdict(list)
assigned_case = collections.defaultdict(list)

for row in df_all.itertuples():
    order_index[row.mrn].append(row)

    
#for col in ['ordering_date', 'proc_start_time', 'proc_end_time', 'result_time']:
#    print(col)
#    print("="*80)
col = 'ordering_date'

offsets = []
linked = []
THRESHOLD = 10 # days
n_missing = 0
for row in df.itertuples():
    if row.MRN not in order_index:
        n_missing += 1
        continue

    candidates = order_index[row.MRN]
    candidates = [c for c in order_index[row.MRN] if c.procedure_DATETIME not in assigned_case[row.MRN]]
    candidate_dates = [datetime.strptime(c.procedure_DATETIME, '%Y-%m-%d %H:%M:%S').date() for c in candidates]
    if len(candidate_dates) == 0:
        continue

    rad_date = datetime.strptime(row._asdict()[col], '%Y-%m-%d %H:%M:%S').date()
    for ts in candidate_dates:
        if ((rad_date-ts).days <= THRESHOLD) and ((rad_date-ts).days  >= 0):
            ranked.append((rad_date-ts).days)
        else:
            ranked.append(float('inf'))
    
    index, element = min(enumerate(ranked), key=itemgetter(1))
    assigned_case[row.MRN].append(candidates[index].procedure_DATETIME)


    if len(ranked) == 0 or element == float('inf'):
        n_missing += 1
        continue     

    candidate_df = pd.DataFrame.from_dict({k:[v] for k,v in candidates[index]._asdict().items()})
    del candidate_df['Index']

    row_df = pd.DataFrame.from_dict({k:[v] for k,v in row._asdict().items()})
    del row_df['Index']

    result_df = pd.concat([row_df, candidate_df], axis=1)
    del result_df['mrn']
    result_df['delta'] = element

    linked.append(result_df)

    # pick best match
    #delta, ts = ranked[0]
    offsets.append(element)

print(f"Total studies:         {len(df)}")
print(f"Final Linked Orders (±{THRESHOLD} days): {len(offsets)} ")
print(f"Mean/SD Days Offset:            {np.mean(offsets):.1f} ({np.std(offsets):.1f})")
print(f"Final Missing Orders:           {n_missing} ")
print("\n")


# In[ ]:


linked_df = pd.concat(linked)
df_new = linked_df[['MRN', 'person_id', 'accession_number', 'order_proc_id', 'procedure_DATETIME', 'delta']]
df_new = df_new.rename(columns={'accession_number': 'Accession'})
df_new['DICOM'] = True
df_new.head(5)


# In[ ]:


d = df_new.procedure_DATETIME.value_counts().reset_index()
print(f'Number of cases with same datetime : {d[d.procedure_DATETIME > 1].shape[0]}')


# In[ ]:


df_new['mrn_date'] = df_new.apply(lambda x: f"{x.MRN}-{x.procedure_DATETIME}", axis=1)
d = df_new.mrn_date.value_counts().reset_index()
print(f'Number of patients with same datetime : {d[d.mrn_date > 1].shape[0]}')


# # Legacy Cohort

# In[ ]:


df_legacy = pd.read_csv('../data/radfusion_3.0_orderproc_cohort_10d_delta.csv', dtype={'mrn': object})
df_legacy['Accession'] = df_legacy['ACCESSION_NUMBER'].apply(lambda x: str(int(x)) if not math.isnan(x) else x)
del df_legacy['ACCESSION_NUMBER']
df_legacy = df_legacy[['MRN', 'person_id', 'Accession', 'order_proc_id', 'procedure_DATETIME', 'delta']]
df_legacy.head(5)


# ### find overlap cases between legacy and new cohort

# In[ ]:


legacy_acc = df_legacy.Accession.tolist()
new_acc = df_new.Accession.tolist()

print(f'Number of studies in legacy cohort {len(legacy_acc)}')
print(f'Number of studies in New cohort {len(new_acc)}')

set1 = set(legacy_acc)
set2 = set(new_acc)
overlap_acc = set1.intersection(set2)
print(f'Number of overlaps {len(overlap_acc)}')


# ### remove overlap studies from legacy cohort

# In[ ]:


df_legacy = df_legacy[~df_legacy.Accession.isin(overlap_acc)]


# ### label studes with DICOMs

# In[ ]:


import pandas as pd


# In[ ]:


# get mapping
crosswalk = pd.read_csv('../data/crosswalk-2000-2016.csv')
anon2acc = dict(zip(crosswalk.ANON_ACCESSION, crosswalk.ACCESSION))
mrn2anon = dict(zip(crosswalk.MRN, crosswalk.ANON_MRN))

crosswalk2 = pd.read_excel('../data/RSNA_Crosswalk.xlsx')
anon2acc.update(dict(zip(crosswalk2.anon_accession, crosswalk2.accession)))
mrn2anon.update(dict(zip(crosswalk2.mrn, crosswalk2.anon_mrn)))


# In[ ]:


import pickle
pickle.dump(mrn2anon, open('mrn_anon_mapping.pkl', 'wb'))


# In[ ]:


# find accessions with dicoms
df_dicoms = pd.read_csv('../data/stanford_instance_metadata_from_dicom_nigam_partition.csv')
df_dicoms[~df_dicoms.filepath.isna()]

df_dicoms['Acc'] = df_dicoms['AnonAcc'].apply(lambda x: anon2acc[x] if x in anon2acc else None)
df_dicoms = df_dicoms[~df_dicoms.Acc.isna()]
print(f"{df_dicoms['Acc'].nunique()} / {len(df_dicoms)}")
      
acc_w_dicoms = df_dicoms['Acc'].tolist()


# In[ ]:


acc_w_dicoms = set(acc_w_dicoms)
df_legacy['DICOM'] = df_legacy['Accession'].apply(lambda x: True if x in acc_w_dicoms else False)
df_legacy['DICOM'].value_counts()


# # Joined cohort

# In[ ]:


df = pd.concat([df_legacy, df_new])
df['DICOM'].value_counts()


# ### Create CLMBR train/val/test split

# In[ ]:


query = """\
    SELECT *
    FROM `som-nero-phi-nigam-starr.femr.splits_omop_2023_03_05`
"""
df_split = query_starr_omop(query, client)

id2split = dict(zip(df_split.person_id, df_split.split))

df['Split'] = df['person_id'].apply(lambda x: id2split[x] if x in id2split else "missing")
print(f'Full cohort split')
print('='*20)
print(df.Split.value_counts())

print(f'Cohort with DICOM split')
print('='*20)
print(df[df.DICOM == True].Split.value_counts())

df = df[df.Split != 'missing']


# ### Create RSNA Label

# In[ ]:


df_rsna = pd.read_csv('../data/stanford_studies_in_rsna_train_with_labels.csv', dtype={'Mrn': object, 'Acc': object})
rsna_accs = set(df_rsna.Acc.to_list())
df['RSNA'] = df['Accession'].apply(lambda x: True if x in rsna_accs else False)
df['RSNA'].value_counts()


# In[ ]:


df_rsna_all = pd.read_csv('../data/stanford_studies_in_rsna.csv')
df_rsna_test = df_rsna_all[df_rsna_all.Split != 'train']

print(f"test count: {df_rsna_test.shape[0]}/{df_rsna_all.shape[0]}")


# In[ ]:


# Check if all studies from RSNA are in train split
rsna_train = pd.read_csv('/Users/marshuang/Labs/Lungren/PE_CT_data_cleaning/rsna/train.csv')

assert df_rsna[~df_rsna.StudyInstanceUID.isin(rsna_train.StudyInstanceUID)].shape[0] == 0


# ### Create Radfusion Label

# In[ ]:


df_rad  = pd.read_csv('../data/stanford_1815_cohort_w_report.csv', dtype={'Mrn': object, 'Acc': object})
rad_accs = set(df_rad.Acc.tolist())
acc_2_split = {str(row.Acc): row.Split for idx, row in df_rad.iterrows()}
acc_2_subseg = {str(row.Acc): row.Subseg for idx, row in df_rad.iterrows()}
acc_2_pe = {str(row.Acc): row.PE_present for idx, row in df_rad.iterrows()}
acc_w_dicom = df_rad[~df_rad.SeriesPath.isna()].Acc.tolist()

df['Radfusion'] = df['Accession'].apply(lambda x: True if x in rad_accs else False)
df['RadfusionSplit'] = df['Accession'].apply(lambda x: acc_2_split[x] if x in acc_2_split else None)
df['RadfusionPEType'] = df['Accession'].apply(lambda x: acc_2_subseg[x] if x in acc_2_split else None)
df['RadfusionPELabel'] = df['Accession'].apply(lambda x: acc_2_pe[x] if x in acc_2_split else None)
df['DICOM'] = df.apply(lambda x: True if x.Accession in acc_w_dicom else x.DICOM, axis=1)

print(df['Radfusion'].value_counts())
print(df['RadfusionSplit'].value_counts())
print(df['RadfusionPEType'].value_counts())
print(df['RadfusionPELabel'].value_counts())


# ## Only keep studies with DICOM

# In[ ]:


df = df[df.DICOM == True]
print(df['Radfusion'].value_counts())


# ### Remove instances from Radfusion & RSNA val/train from this cohort

# In[ ]:


# Get RSNA test accessions
crosswalk = pd.read_csv('../data/crosswalk-2000-2016.csv')
anon2acc = dict(zip(crosswalk.ANON_ACCESSION, crosswalk.ACCESSION))

crosswalk2 = pd.read_excel('../data/RSNA_Crosswalk.xlsx')
anon2acc.update(dict(zip(crosswalk2.anon_accession, crosswalk2.accession)))

df_rsna_all['Accession'] = df_rsna_all['AnonAcc'].apply(lambda x: anon2acc[x] if x in anon2acc else None)

rsna_test_acc = df_rsna_all[df_rsna_all.Split != 'train'].Accession.unique().tolist()
print(f'rsna test acc: {len(rsna_test_acc)}')

# Get radfusion val/test accession
radfusion_val_test_acc = [a for a,s in acc_2_split.items() if s != 'train']
print(f'radfusion test acc: {len(radfusion_val_test_acc)}')


# remove rsna and radufsion non-train studies
df_train = df[df.Split == 'train']
df_train.loc[
    (df_train.Accession.isin(rsna_test_acc)) | 
    (df_train.Accession.isin(radfusion_val_test_acc))
, 'Split'] = 'remove'


# ## Remove Radfusion & RSNA train from val / test 

# In[ ]:


# Get RSNA train accessions
rsna_train_acc = df_rsna_all[df_rsna_all.Split == 'train'].Accession.unique().tolist()
print(f'rsna train acc: {len(rsna_train_acc)}')

# Get radfusion train accession
radfusion_train_acc = [a for a,s in acc_2_split.items() if s == 'train']
print(f'radfusion train acc: {len(radfusion_train_acc)}')

# remove rsna & radufsion train from this cohort
df_val_test = df[df.Split != 'train']
df_val_test.loc[
    (df_val_test.Accession.isin(rsna_train_acc)) | 
    (df_val_test.Accession.isin(radfusion_train_acc))
, 'Split'] = 'remove'


# ### Combine train, val and test cohort

# In[ ]:


df = pd.concat([df_train, df_val_test])
print(df.Split.value_counts())

df.to_csv('ctpa_cohort_with_outcome_labels_remove_rsna_radfusion.csv', index=False)
#csv_to_bq_table('ctpa_cohort_with_outcome_labels_remove_rsna_radfusion.csv', 'som-nero-phi-nigam-starr', 'pulmonary_embolisms', 'ctpa_cohort_with_outcome_labels_remove_rsna_radfusion')



# In[ ]:


df = pd.read_csv('ctpa_cohort_with_outcome_labels_remove_rsna_radfusion.csv')


# ## Remove cases where DATETIME is repeated 

# In[ ]:


df['mrn_datetime'] = df.apply(lambda x: f"{x.MRN}_{x.procedure_DATETIME}", axis=1)


# In[ ]:


d = df.mrn_datetime.value_counts().reset_index()
overlaptime = d[d.mrn_datetime > 1]['index']

print(f'Before remove by time: {df.shape[0]}')
df = df[~df.mrn_datetime.isin(overlaptime)]

#print(f'Before remove by delta: {df.shape[0]}')
#df = df[df.delta == 0]

print(f'Remaining: {df.shape[0]}')


# In[ ]:


df_cohort = df[
    (df.DICOM == True) & 
    (df.Split != 'remove')
]
df_cohort.to_csv('radfusion3_cohort.csv', index=False)

