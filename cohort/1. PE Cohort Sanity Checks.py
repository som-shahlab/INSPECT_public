#!/usr/bin/env python
# coding: utf-8

# # Pulmonary Embolism (PE) Cohort Validation Checks
# 
# Rember to configure your credentials with 
# 
#     gcloud init
# 
# Notes: There are some frustrating dependencies with google SDK and default conda/pip installs. 
# You need the following
# 
#     google-cloud-bigquery==2.3.1
#     google-cloud-storage==1.32.0
# 
# 
# ## Summary of Goals
# 
# We are interested in the set of humans who 
# - Receive a CTPA (CT scan looking for blood clots in the lungs)
# 
# Data (image, note, ts)
#   - CT Scan
#   - Radiology Note
# 
# Labels (patient, ts)
# - PE Diagnosis
# - Pulmonary Hypertension
# - Mortality
# - Readmission
# 

# In[ ]:


import os 

USER        = "mschuang@stanford.edu"
CREDENTIALS = os.path.expanduser(f"~/.config/gcloud/legacy_credentials/{USER}/adc.json")
PROJECT     = "som-nero-phi-nigam-starr"
DATASET     = "shahlab_omop_cdm5_subset_2023_03_05"

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = CREDENTIALS
os.environ['GCLOUD_PROJECT'] = PROJECT


# In[ ]:


import pandas as pd
from google.cloud import bigquery
from datetime import datetime

print(f"Google BigQuery: {bigquery.__version__}")
print(f"Pandas: {pd.__version__}")


# ## A. Legacy Radiology Note Cohort
# 
# We construct our `person_id` to `mrn` mapping using the table `person_id_to_mrn`
# 
#     SELECT person.person_id, CAST(SPLIT(person.person_source_value, " | ")[OFFSET(0)] AS STRING) as mrn
#     FROM `som-nero-phi-nigam-starr.shahlab_omop_cdm5_subset_2023_03_05.person` as person
# 
# for the 2023-03-05 extract, this contains `3,686,202` `person_id`->`mrn` pairs. 
# 
# ### 1. Matching Patient Populations
# 
# Our first goal is to confirm that the same set of humans (indexed by MRN) exsit in the legacy database and STARR-OMOP

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


# In[ ]:


client = bigquery.Client()


# We need to join this dataset with our most recent STARR-OMOP. We do this by
# 1. Querying our _identified_ legacy radiology dataset (from Matt Lungren and Imon Banerjee) 
# 2. Joining by MRN with our _identified_ STARR-OMOP

# In[ ]:


# IMPORTANT: You must cast MRNs as strings otherwise join will fail to find some patients
FORCE_STR_MRN = True

dtypes = {'MRN':str, 'ORDER_ID':str, 'mrn':str} if FORCE_STR_MRN else {}

# legacy radiology notes
query = """\
SELECT *
FROM `som-nero-phi-nigam-starr.pulmonary_embolisms.legacy_radiology_reports`
"""
legacy_df = query_starr_omop(query, client, dtypes=dtypes) 
legacy_df["RAD_DATETIME"] = pd.to_datetime(legacy_df["RAD_DATETIME"], format="%Y-%m-%d %H:%M:%S%z")

# crosswalk: person_id <> MRN
query = """\
SELECT DISTINCT crosswalk.person_id, crosswalk.mrn
FROM `som-nero-phi-nigam-starr.pulmonary_embolisms.person_id_to_mrn` as crosswalk,
`som-nero-phi-nigam-starr.pulmonary_embolisms.legacy_radiology_reports` as legacy
WHERE crosswalk.mrn = legacy.MRN
"""
joined_df = query_starr_omop(query, client, dtypes=dtypes)


# In[ ]:


# confirm mapping is 1:1
mrn_to_person = {row.mrn:row.person_id for row in joined_df.itertuples()}
person_to_mrn = {row.person_id:row.mrn for row in joined_df.itertuples()}
print("STARR-OMOP Mapping is 1:1", len(mrn_to_person)==len(person_to_mrn), len(person_to_mrn))

# summary stats
print(f"Legacy MRNs:     {len(set(legacy_df.MRN))}")
print(f"STARR-OMOP MRNs: {len(set(joined_df.mrn))}")
print(f"Missing MRNs:    {len(set(legacy_df.MRN).difference(set(joined_df.mrn)))}")

# time span
start = min(legacy_df.RAD_DATETIME)
end = max(legacy_df.RAD_DATETIME)
print(f"Legacy Start: {start.date()}")
print(f"Legacy End:   {end.date()}")


# ### 2. Defining Our CTPA Cohort Inclusion Criteria
# 
# The legacy cohort was defined over patients who received a CTPA to diagnose possible pulmonary embolism. To replicate this cohort in STARR, we pull `procedure_source_value` values from the `procedure_ocurence` table for chest CTs for PE. We use regular expressions to match metions of `(CT scans AND pulmonary embolisms) OR (CT chest angiography)`. This identifies 190 unique source values.
#     
# Unfortunately, this does not identify all patients receiving a CTPA. The value `procedure_source_value` takes seems to depend on whether the original `order_proc_id` is a parent or child in Clarity. Only RIT can provide parent/child relationships so I had them create the project `som-nero-phi-nigam-starr.rit_data_jasonfries.jason_final_list` which contains some (but not all, mysteriously) of the order ids missing from our current linkages. 
# 
# Parent orders contain more general source value names like `CT ANGIO CHEST`. These overlap with our prior values in only two cases. The final labeler set includes 171 normalized values for `procedure_source_value`.
#  

# In[ ]:


# regular expression matching 
query = """SELECT DISTINCT procedure_concept_id, procedure_source_value
FROM `som-nero-phi-nigam-starr.shahlab_omop_cdm5_subset_2023_03_05.procedure_occurrence` as procs,
`som-nero-phi-nigam-starr.shahlab_omop_cdm5_subset_2023_03_05.person` as P
WHERE 
((REGEXP_CONTAINS(LOWER(procedure_source_value), r"pulmonary embolism| pe ") 
AND REGEXP_CONTAINS(LOWER(procedure_source_value), r"cat scan|\s*ct |\s*cta |ct[-]cta"))
OR LOWER(procedure_source_value) LIKE "%ct chest angiography%") 
AND P.person_id = procs.person_id
"""
ctpa_v1_df = query_starr_omop(query, client)

# matching missing parent order_proc_ids
query = """SELECT DISTINCT procedure_concept_id, procedure_source_value
FROM `som-nero-phi-nigam-starr.shahlab_omop_cdm5_subset_2023_03_05.procedure_occurrence` as procs,
(SELECT legacy.ORDER_ID, fix.parent_order_id 
FROM `som-nero-phi-nigam-starr.pulmonary_embolisms.legacy_radiology_reports` AS legacy,
     `som-nero-phi-nigam-starr.rit_data_jasonfries.jason_final_list` as fix
WHERE CAST(legacy.ORDER_ID AS STRING) IN (
  SELECT CAST(ORDER_ID AS STRING) FROM `som-nero-phi-nigam-starr.rit_data_jasonfries.jason_final_list`
) AND legacy.ORDER_ID = fix.ORDER_ID) AS parents
WHERE JSON_QUERY(trace_id, '$.order_proc_id') = CAST(parents.parent_order_id AS STRING)
"""
ctpa_v2_df = query_starr_omop(query, client)

A = set(map(lambda x:x.lower(), ctpa_v1_df.procedure_source_value.to_list()))
B = set(map(lambda x:x.lower(), ctpa_v2_df.procedure_source_value.to_list()))
print("Intersection:", A.intersection(B))
print(f"Total procedure_source_value: {len(A.union(B))}")

# combine results
ctpa_df = pd.concat((ctpa_v1_df, ctpa_v2_df), axis=0)


# In[ ]:


# copy dataframe to BigQuery table
table_id = "som-nero-phi-nigam-starr.pulmonary_embolisms.ctpa_labels"
labels = set([name.lower() for name in ctpa_df.procedure_source_value])
df = pd.DataFrame(data=list(labels), columns=['procedure_source_value'])
load_table_from_dataframe(client, df, table_id)


# ### 3. Replicating & Extending the Legacy Cohort in STARR-OMOP
# 
# To replicate the original datapull (1995/07/18 to 2018/02/16) we use the following linkage with our CTPA labels:
# 
#     SELECT procs.*, JSON_QUERY(procs.trace_id, '$.order_proc_id') AS order_proc_id, CAST(SPLIT(person.person_source_value, " | ")[OFFSET(0)] AS STRING) as mrn
#     FROM `som-nero-phi-nigam-starr.shahlab_omop_cdm5_subset_2023_03_05.procedure_occurrence` as procs,
#     `som-nero-phi-nigam-starr.shahlab_omop_cdm5_subset_2023_03_05.person` as person
#     WHERE 
#     procs.person_id=person.person_id
#     AND LOWER(procs.procedure_source_value) in (
#       SELECT procedure_source_value FROM `som-nero-phi-nigam-starr.pulmonary_embolisms.ctpa_labels`
#     ) 
#     AND procs.person_id IN (SELECT person_id FROM `som-nero-phi-nigam-starr.pulmonary_embolisms.legacy_pe_cohort_starr`)
#     AND procs.procedure_DATETIME BETWEEN '1995-07-18 00:00:00' AND '2018-02-16 23:59:59'
# 
# This finds 
# 
# - Patients: 65,308 vs. 65,325 (i.e., we are missing 187 patients)
# - CT Events: 149,831 vs. 137,834 
# 
# If we don't restrict to our MRN set, we actually find 17,614 more patients
# - Patients: 78,865
# - CT Events 167,445
# We can confirm these 17,614 new patients did have CTPA events by looking at their order procedure_source_values
# 
# We use this query to generate the table `legacy_linked_ctpa_orders`

# ### 4. Matching Procedure `order_proc_id` 
# 
# For all orders in the legacy database `ORDER_ID` corresponds to an internal Epic Clarity. We find that we are missing 
# 22,139 records (matching 115,695 / 137,834).

# In[ ]:


query = """SELECT *
FROM  `som-nero-phi-nigam-starr.pulmonary_embolisms.legacy_radiology_reports` as legacy,
`som-nero-phi-nigam-starr.pulmonary_embolisms.legacy_linked_ctpa_orders` as orders
WHERE
CAST(orders.order_proc_id AS STRING) = CAST(legacy.ORDER_ID AS STRING)
"""
order_procs_df = query_starr_omop(query, client, dtypes= {'MRN':str, 'order_proc_id':str}) 


# In[ ]:


print(f"Legacy Order Proc IDs:     {len(set(legacy_df.ORDER_ID.to_list()))}")
print(f"STARR-OMOP Order Proc IDs: {len(set(order_procs_df.order_proc_id.to_list()))}")

order_ids = set(order_procs_df.order_proc_id)
missing_orders_df = legacy_df[~(legacy_df.ORDER_ID.isin(order_ids))]
print(f"MISSING Order Proc IDs:    {len(missing_orders_df)}")

query = """SELECT *
FROM `som-nero-phi-nigam-starr.pulmonary_embolisms.legacy_linked_ctpa_orders` as orders
WHERE CAST(orders.order_proc_id AS STRING) NOT IN (
SELECT CAST(ORDER_ID AS STRING)
FROM
`som-nero-phi-nigam-starr.pulmonary_embolisms.legacy_radiology_reports`
)
"""
missing_order_procs_df = query_starr_omop(query, 
                                          client, 
                                          dtypes={'mrn':str, 'order_proc_id':str, 'modifier_source_value':str}) 


# #### Missing `order_proc_ids`
# 
# From above, we're finding 149,831 vs. 137,834 CT events in `legacy_linked_ctpa_orders`. 

# In[ ]:


import collections
import numpy as np
from datetime import datetime 
from operator import itemgetter

# bucket our unmapped orders by MRN
order_index = collections.defaultdict(list)
assigned_case = collections.defaultdict(list)

for row in missing_order_procs_df.itertuples():
    order_index[row.mrn].append(row)

# our legacy set w/ missing orders
legacy_missing = legacy_df[~(legacy_df.ORDER_ID.isin(order_procs_df.order_proc_id))]

offsets = []
linked = []
THRESHOLD = 10 # days
n_missing = 0
for row in legacy_missing.itertuples():
    if row.MRN not in order_index:
        n_missing += 1
        continue
        
    candidates = [c for c in order_index[row.MRN] if c.procedure_DATETIME not in assigned_case[row.MRN]]
    candidate_dates = [datetime.strptime(c.procedure_DATETIME, '%Y-%m-%d %H:%M:%S').date() for c in candidates]
    
    if len(candidate_dates) == 0:
        continue
    
    #rad_date = datetime.strptime(row.RAD_DATE, '%d-%b-%y %I.%M.%S.%f000000 %p').date()
    rad_date = datetime.strptime(str(row.RAD_DATETIME), '%Y-%m-%d %H:%M:%S%z').date()
    #ranked = sorted([(abs(rad_date-ts).days, ts) for ts in candidates if abs(rad_date-ts).days <= THRESHOLD])
    
    ranked = []
    for ts in candidate_dates:
        # report date should come after the procedure
        if ((rad_date-ts).days <= THRESHOLD) and ((rad_date-ts).days  >= 0):
            ranked.append((rad_date-ts).days)
        else:
            ranked.append(float('inf'))
    #ranked = sorted([abs(rad_date-ts).days if (abs(rad_date-ts).days <= THRESHOLD) and ((rad_date-ts).days  >= 0) else float('inf') for ts in candidate_dates])
    #ranked = [abs(rad_date-ts).days if abs(rad_date-ts).days <= THRESHOLD else float('inf') for ts in candidate_dates]
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

print(f"MISSING Order Proc IDs:         {len(legacy_missing)}")
print(f"Final Linked Orders (±{THRESHOLD} days): {len(offsets)} ")
print(f"Mean/SD Days Offset:            {np.mean(offsets):.1f} ({np.std(offsets):.1f})")
print(f"Final Missing Orders:           {n_missing} ")


# In[ ]:


linked_df = pd.concat(linked)
order_procs_df['delta'] = 0
#del order_procs_df['mrn_1']

final_df = pd.concat([linked_df, order_procs_df])


# In[ ]:


d = order_procs_df.procedure_DATETIME.value_counts().reset_index()
print(f"Numbe of overlaps: {d[d.procedure_DATETIME > 1].shape[0]}")


# In[ ]:


order_procs_df['mrn_date'] = order_procs_df.apply(lambda x: f"{x.MRN}-{x.procedure_DATETIME}", axis=1)
d = order_procs_df.mrn_date.value_counts().reset_index()
print(f"Numver of patient with overlaps: {d[d.mrn_date > 1].shape[0]}")


# In[ ]:


d = linked_df.procedure_DATETIME.value_counts().reset_index()
print(f"Numbe of overlaps: {d[d.procedure_DATETIME > 1].shape[0]}")


# In[ ]:


linked_df['mrn_date'] = linked_df.apply(lambda x: f"{x.MRN}-{x.procedure_DATETIME}", axis=1)
d = linked_df.mrn_date.value_counts().reset_index()
print(f"Numver of patient with overlaps: {d[d.mrn_date > 1].shape[0]}")


# In[ ]:


d = final_df.procedure_DATETIME.value_counts().reset_index()
print(f"Numbe of overlaps: {d[d.procedure_DATETIME > 1].shape[0]}")


# In[ ]:


final_df['mrn_date'] = final_df.apply(lambda x: f"{x.MRN}-{x.procedure_DATETIME}", axis=1)
d = final_df.mrn_date.value_counts().reset_index()
print(f"Numver of patient with overlaps: {d[d.mrn_date > 1].shape[0]}")


# ### 5. Legacy STARR-OMOP Recreation Summary 
# 
# The above BigQuery SQL enables recreating the legacy PE Cohort (1995-07-18 to 2018-02-16) with a small degree of error
# 
# - MRNs: 65495 / 65325 (99.7%)
# - Orders/Accessions: 136680 / 137834 (99.1%)
# 
# Removing the date range enables recreating the cohort over the most recent data.

# In[ ]:


final_df.to_csv('radfusion_3.0_orderproc_cohort_10d_delta.csv', index=False)

