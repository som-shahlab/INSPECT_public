#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


csv_to_bq_table("./impact_cohort.csv", "som-nero-phi-nigam-starr", "pulmonary_embolisms", "impact_cohort")


# In[ ]:


query = """\
    SELECT * FROM `som-nero-phi-nigam-starr.shahlab_omop_cdm5_subset_2023_03_05.concept` 

"""
concept = query_starr_omop(query, client)


# In[ ]:


"""
    SELECT *
    FROM `som-nero-phi-nigam-starr.pulmonary_embolisms.impact_cohort` as cohort
    INNER JOIN `som-nero-phi-nigam-starr.shahlab_omop_cdm5_subset_2023_03_05.person` as person
    ON person.person_id=cohort.PersonID
"""

query = """
    SELECT *
    FROM `som-nero-phi-nigam-starr.shahlab_omop_cdm5_subset_2023_03_05.person` as person
    WHERE person.person_id IN (SELECT cohort.PersonID FROM `som-nero-phi-nigam-starr.pulmonary_embolisms.impact_cohort` as cohort);
"""
person  = query_starr_omop(query, client)


# In[ ]:


person.ethnicity_source_value.value_counts()


# In[ ]:


person = person[['person_id', 'gender_concept_id', 'race_source_value', 'ethnicity_source_value', 'birth_DATETIME']]


gender_map = {8532: 'Female', 8507: 'Male', 0: 'Unknown'}
person['gender'] = person['gender_concept_id'].apply(lambda x: gender_map[x])
person['race'] = person['race_source_value'].apply(lambda x: x.split('|')[0].strip(' '))
person['race'] = person['race'].apply(lambda x: 'Unknown' if x in ['Unknown', 'Delines to State', '', 'Declines to State'] else x)
person['race'] = person['race'].apply(lambda x: 'Other' if x in ['Other', 'Native Hawaiian or Other Pacific Islander', 'Native Hawaiian or Other Pacific Islander'] else x)


def def_hispanic(text):
    if 'Non-Hispanic' in text: 
        return 'Non-Hispanic'
    elif 'Hispanic/Latino' in text: 
        return 'Hispanic'
    else: 
        return 'Unknown'
    
person['ethnicity'] = person['ethnicity_source_value'].apply(lambda x: x.split('|')[0].strip(' '))
person['ethnicity'] = person['ethnicity'].apply(def_hispanic)


person2gender = dict(zip(person.person_id, person.gender))
person2race = dict(zip(person.person_id, person.race))
person2birth = dict(zip(person.person_id, person.birth_DATETIME))
person2ethnicity = dict(zip(person.person_id, person.ethnicity))


# In[ ]:


df = pd.read_csv('impact_cohort.csv')
df['Gender'] = df['PersonID'].apply(lambda x: person2gender[x])
df['Race'] = df['PersonID'].apply(lambda x: person2race[x])
df['BirthDate'] = df['PersonID'].apply(lambda x: person2birth[x])
df['Ethnicity'] = df['PersonID'].apply(lambda x: person2ethnicity[x])


# In[ ]:


df['Ethnicity'].value_counts()


# In[ ]:


df['BirthDate'] = pd.to_datetime(df['BirthDate'] )
df['ProcedureDatetime'] = pd.to_datetime(df['ProcedureDatetime'])


# In[ ]:


df['Age'] = df['ProcedureDatetime'] - df['BirthDate']
df['Age'] = df['Age'].apply(lambda x: x.days / 365)
df['Age']


# In[ ]:


def map_age(age):
    if age <=20: 
        return '0-20'
    elif (age > 21) & (age <= 40):
        return '21-40'
    elif (age > 41) & (age <= 60):
        return '41-60'
    elif (age > 61) & (age <= 80):
        return '61-80'
    else: 
        return '81+'

df['AnonAge'] = df['Age'].apply(map_age)
df['AnonAge'].value_counts()


# In[ ]:


print(df.shape[0])
print(df.MRN.nunique())
for s in ['train', 'valid', 'test']:
    print(s)
    print(df[df.Split == s].shape[0], f"({(df[df.Split == s].shape[0] / len(df)) * 100 :.2f}\%)")
    print(df[df.Split == s].MRN.nunique(), f"({(df[df.Split == s].MRN.nunique() / df.MRN.nunique()) * 100 :.2f}\%)")


# In[ ]:


df_p = df.groupby("PersonID").head(1).reset_index()

s = df_p.Ethnicity.value_counts()
split_count = len(df_p)
for g in ['Hispanic', 'Non-Hispanic', 'Unknown']:
    print(g, s[g])
    
for v in ['train', 'valid', 'test']:
    print(v)
    s = df_p[df_p.Split == v].Ethnicity.value_counts()
    split_count = len(df_p[df_p.Split == v])
    for g in ['Hispanic', 'Non-Hispanic', 'Unknown']:
        if g in s:
            print(g)
            print(s[g], " & \cellcolor[HTML]{EDF6FF} " + f"{(s[g] / split_count) * 100 :.2f}\%")
    print('\n')


# In[ ]:


df_p = df.groupby("PersonID").head(1).reset_index()

s = df_p.Gender.value_counts()
split_count = len(df_p)
for g in ['Female', 'Male', 'Unknown']:
    print(g, s[g])
    
for v in ['train', 'valid', 'test']:
    print(v)
    s = df_p[df_p.Split == v].Gender.value_counts()
    split_count = len(df_p[df_p.Split == v])
    for g in ['Female', 'Male', 'Unknown']:
        if g in s: 
            print(g, s[g], " & \cellcolor[HTML]{EDF6FF} " + f"{(s[g] / split_count) * 100 :.2f}\%")
    print('\n')
        


# In[ ]:


df_p = df.groupby("PersonID").head(1).reset_index()

s = df_p.Race.value_counts()
split_count = len(df_p)
for g in ['Asian', 'Black or African American', 'American Indian or Alaska Native', 'Other', 'Unknown', 'White']:
    print(g, s[g])
    
for v in ['train', 'valid', 'test']:
    print(v)
    s = df_p[df_p.Split == v].Race.value_counts()
    split_count = len(df_p[df_p.Split == v])
    for g in ['Asian', 'Black or African American', 'American Indian or Alaska Native', 'Other', 'Unknown', 'White']:
        if g in s: 
            print(g, s[g], " & \cellcolor[HTML]{EDF6FF} " + f"({(s[g] / split_count) * 100 :.1f})\%")
    print('\n')
        


# In[ ]:


print(df['1_month_mortality'].value_counts())
    
for v in ['train', 'valid', 'test']:
    print(v)
    s = df[df.Split == v]['1_month_mortality'].value_counts()
    split_count = len(df[df.Split == v])
    for g in ['True', 'False', 'Censored']:
        if g in s: 
            print(g, s[g], " & \cellcolor[HTML]{EDF6FF} " + f"({(s[g] / split_count) * 100 :.1f})\%")
    print('\n')


# In[ ]:


print(df['6_month_mortality'].value_counts())
    
for v in ['train', 'valid', 'test']:
    print(v)
    s = df[df.Split == v]['6_month_mortality'].value_counts()
    split_count = len(df[df.Split == v])
    for g in ['True', 'False', 'Censored']:
        if g in s: 
            print(g, s[g], " & \cellcolor[HTML]{EDF6FF} " + f"({(s[g] / split_count) * 100 :.1f})\%")
    print('\n')


# In[ ]:


print(df['12_month_mortality'].value_counts())
    
for v in ['train', 'valid', 'test']:
    print(v)
    s = df[df.Split == v]['12_month_mortality'].value_counts()
    split_count = len(df[df.Split == v])
    for g in ['True', 'False', 'Censored']:
        if g in s:
            print(g)
            print(s[g], " & \cellcolor[HTML]{EDF6FF} " + f"({(s[g] / split_count) * 100 :.1f})\%")
    print('\n')


# In[ ]:


print(df['1_month_readmission'].value_counts())
    
for v in ['train', 'valid', 'test']:
    print(v)
    s = df[df.Split == v]['1_month_readmission'].value_counts()
    split_count = len(df[df.Split == v])
    for g in ['True', 'False', 'Censored']:
        if g in s: 
            print(g, s[g], " & \cellcolor[HTML]{EDF6FF} " + f"({(s[g] / split_count) * 100 :.1f})\%")
    print('\n')


# In[ ]:


print(df['6_month_readmission'].value_counts())
    
for v in ['train', 'valid', 'test']:
    print(v)
    s = df[df.Split == v]['1_month_readmission'].value_counts()
    split_count = len(df[df.Split == v])
    for g in ['True', 'False', 'Censored']:
        if g in s:
            print(g)
            print(s[g], " & \cellcolor[HTML]{EDF6FF} " + f"({(s[g] / split_count) * 100 :.1f})\%")
    print('\n')


# In[ ]:


print(df['_month_readmission'].value_counts())
    
for v in ['train', 'valid', 'test']:
    print(v)
    s = df[df.Split == v]['1_month_readmission'].value_counts()
    split_count = len(df[df.Split == v])
    for g in ['True', 'False', 'Censored']:
        if g in s:
            print(g)
            print(s[g], " & \cellcolor[HTML]{EDF6FF} " + f"({(s[g] / split_count) * 100 :.1f})\%")
    print('\n')


# In[ ]:




