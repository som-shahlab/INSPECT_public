#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


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

USER = "mschuang@stanford.edu"
CREDENTIALS = os.path.expanduser(f"~/.config/gcloud/legacy_credentials/{USER}/adc.json")
PROJECT = "som-nero-phi-nigam-starr"
DATASET = "shahlab_omop_cdm5_subset_2023_03_05"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS
os.environ["GCLOUD_PROJECT"] = PROJECT


import os
import ctypes
import hashlib
from google.cloud import bigquery


def load_table_from_dataframe(client, dataframe, table_id):
    # infer schema from dataframe
    bq_enums = {"object": "STRING", "int64": "INTEGER"}
    schema = [
        bigquery.SchemaField(name, bq_enums[str(dtype)])
        for name, dtype in zip(dataframe.columns, dataframe.dtypes)
    ]

    job_config = bigquery.LoadJobConfig(
        schema=schema, write_disposition="WRITE_TRUNCATE"  # overwrite table if exists
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
            job = client.load_table_from_file(
                source_file, table_ref, job_config=job_config
            )
        job.result()
    except Exception as e:
        print(f"{e}")

    return bq_table


def query_starr_omop(
    query,
    client,
    show_progress_bar=True,
    use_cache=True,
    cache_dir="../data/cache",
    **kwargs,
):
    """
    Simple function for processing a query using BigQuery and caching the results.
    """
    hash_value = hashlib.sha256(bytes(query, "utf-8")).hexdigest()
    fpath = f"{cache_dir}/{hash_value}.tsv"
    dtypes = kwargs["dtypes"] if "dtypes" in kwargs else None

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    if use_cache and os.path.exists(fpath):
        # df = pd.read_csv(fpath, sep='\t', dtype=dtypes)
        print(f"Loaded from cache {hash_value}")
    else:
        job = client.query(query)
        df = job.to_dataframe(progress_bar_type="tqdm")
        df.to_csv(fpath, sep="\t", index=False)

    df = pd.read_csv(fpath, sep="\t", dtype=dtypes)
    return df


client = bigquery.Client()


# In[ ]:


query = """\
    SELECT * from som-nero-phi-nigam-starr.shahlab_omop_cdm5_subset_2023_03_05.person
"""
# person = query_starr_omop(query, client)
person["MRN"] = person["person_source_value"].apply(
    lambda x: x.split("|")[0].replace(" ", "")
)
mrn2id = dict(zip(person.MRN, person.person_id))


# In[ ]:


df = pd.read_csv("Fullradiologyreport_merged.csv", dtype={"MRN": object})
df.rad_report = df.rad_report.apply(lambda x: x.replace("-***-", " "))
df.shape


# In[ ]:


df["MRN"] = df["MRN"].str.zfill(8)
df["person_id"] = df["MRN"].apply(lambda x: mrn2id[x] if x in mrn2id else None)
df["person_id"].nunique()

df[~df.person_id.isna()]
df.shape


# In[ ]:


query = """\
    SELECT *
    FROM `som-nero-phi-nigam-starr.femr.splits_omop_2023_03_05`
"""
df_split = query_starr_omop(query, client)

id2split = dict(zip(df_split.person_id, df_split.split))

df["Split"] = df["person_id"].apply(
    lambda x: id2split[x] if x in id2split else "missing"
)


# In[ ]:


df["Split"].value_counts()


# In[ ]:


df_train = df[df.Split == "train"]
df_val = df[df.Split == "valid"]
df_test = df[df.Split == "test"]


# In[ ]:


import re


def get_impression_section_start(text):
    """
    Rule-based approach to determine absolute char
    offset of the impression section
    """
    # patterns are in order of precedence
    regexs = [
        re.compile(r"""impression[:]""", re.I),
        re.compile(r"""impression[:;]""", re.I),
        re.compile(r"""IMPRESSSION:""", re.I),
        re.compile(r"""\.IMPRESSION[:]""", re.I),
        re.compile(r"""(\.|)impression[:]""", re.I),
        re.compile(f"""Impression[:;]""", re.I),
        re.compile(r"""conclusion[:;]""", re.I),
        re.compile(r"  1\.", re.I),
        re.compile(r" : 1\.", re.I),
        re.compile(r".\n \n:\n \n1.", re.I),
        re.compile(r".\n \n \n1.", re.I),
        re.compile(r".\n \n1.", re.I),
        # re.compile(r"FINDINGS:", re.I),
    ]
    for rgx in regexs:
        m = rgx.search(text)
        if not m:
            continue
        return m.span()[0]

    return None


def get_impression_section_start_findings(text):
    """
    Rule-based approach to determine absolute char
    offset of the impression section
    """
    # patterns are in order of precedence
    regexs = [
        re.compile(r"FINDINGS:", re.I),
        re.compile(r"FINDING:", re.I),
    ]
    for rgx in regexs:
        m = rgx.search(text)
        if not m:
            continue
        return m.span()[0]

    return None


def get_impression_section_end(text):
    """
    Rule-based approach to determine absolute char
    offset of the impression section.
    """
    # patterns are in order of precedence
    regexs = [
        re.compile(
            r"""(END (OF )*IMPRESSION( SUMMARY)*[:.]*|SUMMARY( CODE)*[:.])""", re.I
        ),
        re.compile(r"""SUMMARY CODE [0-9]|I have personally reviewed""", re.I),
    ]
    for rgx in regexs:
        m = rgx.search(text)
        if not m:
            continue
        return m.span()[0]

    # return end of note string
    return None


def get_impression_section_offsets(text):
    """
    Return start and end section offsets of document text (if they exist)
    """
    start = get_impression_section_start(text)
    start_findings = get_impression_section_start_findings(text)
    if start is None:
        if start_findings is not None:
            start = start_findings
        else:
            start = 0
    # if no match is found, return the end of the note string
    end = get_impression_section_end(text)
    if end is None:
        end = len(text)
    if end < start:
        print(text[end:start])
        end = len(text)
    return start, end


def get_impression(text):
    start, end = get_impression_section_offsets(text)
    if start is None:
        return None
    return text[start:end]


df_train["impression"] = df_train["rad_report"].apply(lambda x: get_impression(x))
df_val["impression"] = df_val["rad_report"].apply(lambda x: get_impression(x))
df_test["impression"] = df_test["rad_report"].apply(lambda x: get_impression(x))


# ### Remove missing impressions

# In[ ]:


def remove_missing(df_cohort):
    print(f"Original cohort size: {df_cohort.shape[0]}")

    print(
        f"Remove cases without impression: {df_cohort[df_cohort['impression'].isna()].shape[0]}"
    )
    df_cohort = df_cohort[~df_cohort["impression"].isna()]

    print(
        f"Remove cases with empty impression: {df_cohort[df_cohort['impression'] == ''].shape[0]}"
    )
    df_cohort = df_cohort[df_cohort["impression"] != ""]

    print(f"Final cohort size: {df_cohort.shape[0]}")

    return df_cohort


df_train = remove_missing(df_train)
df_val = remove_missing(df_val)
df_test = remove_missing(df_test)


# ## De-identify

# #### Anonymize Radiology Text
# Impression sections contain PHI variables that must be removed
#
# - dates / times
# - healthcare worker names
# - pager numbers

# In[ ]:


import re
import datetime
from datetime import timedelta
from collections import defaultdict


meridiems = r"""([APap][Mm]|[APap][.][Mm][.])"""  # AM | PM
clock_time = r"""([0-2]*[0-9][:][0-5][0-9])(:[0-5][0-9])*"""  # 10:00 | 10:10:05

year_range = r"""(19[0-9][0-9]|20[012][0-9])"""  # 1900 - 2029
day_range = r"""(3[01]|[12][0-9]|[0]*[1-9])"""  # 01 - 31
month_range = r"""([1][012]|[0]*[1-9])"""  # 01 - 12

month_full = r"""(january|february|march|april|may|june|july|august|september|october|november|december)"""
month_abbrv = r"""((jan|feb|mar|apr|may|june|jul|aug|sep|sept|oct|nov|dec)[.]*)"""
day_full = r"""(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"""
day_abbrv = r"""(mon|tue|wed|thu|fri|sat|sun)[.]*"""

rgx_time = r"""({clock_time})(\s*{meridiems})*""".format(
    clock_time=clock_time, meridiems=meridiems
)
rgx_month = r"""({month_full}|{month_abbrv})""".format(
    month_full=month_full, month_abbrv=month_abbrv
)
rgx_day = r"""({day_full})""".format(day_full=day_full, day_abbrv=day_abbrv)

# Dates with Numeric Months
d_params = {"M": month_range, "D": day_range, "Y": year_range, "T": rgx_time}

rgx_y = r"""\b({Y})\b""".format(**d_params)  # 2019
rgx_mdy = r"""({M}[-]{D}[-]{Y}|{M}[/]{D}[/]{Y})(\s*{T})*""".format(
    **d_params
)  # 1/11/2000 12:30 PM
rgx_ymd = r"""({Y}[/]{M}[/]{D}|{Y}[-]{M}[-]{D})(\s*{T})*""".format(
    **d_params
)  # 2000-11-12 12:30 PM
rgx_md = r"""\b(([1][0-9]|[0][1-9])[/][0-3][0-9])\b"""  # 11/31 (implied year)
rgx_ys = r"""\b((mid[-])*({Y}|the [2-9]0)[s])\b""".format(
    **d_params
)  # 1920s | mid-1990s | the 80s
rgx_mdy2 = (
    r"""\b({M}[-]{D}[-]([012][0-9])|{M}[/]{D}[/]([012][0-9]))(\s*{T})*\b""".format(
        **d_params
    )
)  # 9/21/09

# Dates with String Months
s_params = {"M": rgx_month, "D": day_range, "Y": year_range, "T": rgx_time}

rgx_dash_mdy = r"""({M}[-]{D}[-]{Y})(\s*{T})*""".format(
    **s_params
)  # Apr-03-2011 13:21:30
rgx_m_of_y = r"""({M} of {Y}|{Y} in {M})""".format(
    **s_params
)  # January of 2018 | 2005 in April
rgx_month_dy = r"""({M}[ ]*{D}(st|nd|rd|th)*[, ]*{Y})""".format(
    **s_params
)  # July 30, 2019
rgx_concat_mdy = r"""({D}{M}{Y})""".format(**s_params)  # 30Jan2019
rgx_d_of_my = r"""({D}(st|nd|rd|th)* of {M}(\s{Y})*)""".format(
    **s_params
)  # 20th of July 2010
rgx_month_d = r"""\b({M} {D}(st|nd|rd|th)*)\b""".format(**s_params)  # September 16
rgx_month_year = r"""\b({M}\s*{Y})\b""".format(**s_params)  # May 2010

# Other TIMEX3 expressions
number_0_10 = r"""(zero|one|two|three|four|five|six|seven|eight|nine|ten)"""
number_11_19 = r"""((thir|four|fif|six|seven|eigh|nine)teen)"""
number_20_90 = (
    r"""((twenty|thirty|fourty|fifty|sixty|seventy|eighty|ninty)([-]*{N})*)""".format(
        N=number_0_10
    )
)

# five years ago
rgx_number_full = r"""{}|{}|{}""".format(number_0_10, number_11_19, number_20_90)
rgx_timex_ago = r"""\b(([1-9][0-9]*([.][5])*|[1-9][0-9]*\s*(to|[-])\s*[1-9][0-9]*|({})|few|a) ((year|month|wk|week|day|hour)[s]*) (ago|back|prior))\b""".format(
    rgx_number_full
)

# relative temporal expressions
rgx_day_parts = r"""\b((this) (morning|afternoon|evening)|(yesterday|today|tomorrow|tonight|tonite)[']*[s]*)\b"""
rgx_day_times = r"""\b(now|currently|presently)\b"""
rgx_day_rela = r"""\b((next|last|this) ({W}|week|month|year))\b""".format(W=day_full)
rgx_recent_now = r"""\b((current|recent)(ly)*|at this (point|time)|now)\b"""
rgx_operative = r"""\b((pre|post|intra)[-]*(operative(ly)*|op))\b"""

timex_regexes = [
    # r'''\b{}\b'''.format(rgx_time),
    r"""\b{}\b""".format(rgx_month),
    r"""\b{}\b""".format(rgx_day),
    rgx_time,
    # rgx_month,
    # rgx_day,
    rgx_month_year,
    rgx_y,
    rgx_mdy,
    rgx_ymd,
    rgx_md,
    rgx_ys,
    rgx_mdy2,
    rgx_dash_mdy,
    rgx_m_of_y,
    rgx_month_dy,
    rgx_concat_mdy,
    rgx_d_of_my,
    rgx_month_d,
    # rgx_timex_ago,
    # rgx_day_parts,
    # rgx_day_rela,
    # rgx_recent_now,
    # rgx_operative
]


# In[ ]:


# from .timex import regexes as timex_regexes


def is_overlapping(a, b):
    return range(max(a[0], b[0]), min(a[-1], b[-1]) + 1)


class RegexTagger:
    def __init__(self, name, patterns, stopwords=None):
        self.name = name
        self.patterns = patterns
        self.stopwords = {} if stopwords is None else stopwords

    def tag(self, text):
        matches = set()
        for rgx in self.patterns:
            for match in re.finditer(rgx, text, re.I):
                start, end = match.span()
                if match[0].lower().strip() in self.stopwords:
                    continue
                matches.add((start, end))

        # sort span matches
        matches = sorted(matches, key=lambda x: x[-1], reverse=True)

        stack = []
        for a in matches:
            if not stack:
                stack.append(a)
            else:
                b = stack.pop()
                if is_overlapping(a, b):
                    start, end = min(a[0], b[0]), max(a[-1], b[-1])
                    stack.append((start, end))
                else:
                    stack += [b, a]

        # strip trailing whitespace
        for i in range(len(stack)):
            start, end = stack[i]
            while text[start:end][-1] == " ":
                end -= 1
            stack[i] = (start, end, self.name)

        # return matches
        return stack


# load dictionary of names (may include PHI)
names = set(pd.read_csv("names.txt", sep="\t").name)
name_dict_rgx = f'({"|".join(sorted(names, key=lambda x:len(x), reverse=True))})'

name_uppercase_rgx = r"""(DR[.]*\s+(([A-Z'-]+\s*){1,2}(?=(UPON|FROM|WITH|WAS|AND|VIA|AS|TO|AT|BY|IN|OF|ON|[@.,;(]))))\b"""
name_titlecase_rgx = r"""(Dr[.]*\s+(?:(?:[A-Z'][a-z-]+)\s*){1})"""

regexes = [name_uppercase_rgx, name_dict_rgx, name_titlecase_rgx]
person = RegexTagger("person", regexes, {"may"})
pager = RegexTagger("pager", [r"""PAGER [0-9-]+\b"""])
timex = RegexTagger("timex", timex_regexes, {"may"})


hours_regexes = [r"""([0-9]{3,4}\s*HOURS)""", r"""(?<=(AT ))[0-9]{3,4}"""]
hours = RegexTagger("hours", hours_regexes)


# In[ ]:


from functools import partial
import tqdm


def anonymize_person(text):
    m = re.search(r"""dr[.]*|PHYSICIAN""", text, re.I)
    if m is not None:
        i, j = m.span()
        anon_toks = [text[i:j]]
        toks = text[j + 1 :].split()
        if len(toks) == 1:
            anon_toks.append("LAST_NAME")
        elif len(toks) > 1:
            anon_toks.append("FIRST_NAME")
            anon_toks.append("LAST_NAME")
        # return f'|{" ".join(anon_toks).strip()}|'
        return f'<PERSON>{" ".join(anon_toks).strip()}</PERSON>'

    # first/last name
    toks = text.split()
    if len(toks) == 2:
        # return "|FIRST_NAME LAST_NAME|"
        return "<PERSON>FIRST_NAME LAST_NAME</PERSON>"
    return None


def anonymize_datetime(text):
    month_full = r"""(january|february|march|april|may|june|july|august|september|october|november|december)"""
    month_abbrv = r"""((jan|feb|mar|apr|may|june|jul|aug|sep|sept|oct|nov|dec)[.]*)"""
    rgx_month = r"""({month_full}|{month_abbrv})""".format(
        month_full=month_full, month_abbrv=month_abbrv
    )

    replace = {
        "AM_PM": r"""([APap][Mm]|[APap][.][Mm][.])""",
        "MONTH": rgx_month,
        "DAY": r"""(monday|tuesday|wednesday|thursday|friday|saturday|sunday)|(mon|tue|wed|thu|fri|sat|sun)[.]*""",
    }

    transforms = [(r"\d", "X")]
    for tok, rgx in replace.items():
        m = re.search(rgx, text, re.I)
        if m:
            transforms.append((rgx, tok))
    # apply transforms
    mention = text
    for rgx, repl in transforms:
        mention = re.sub(rgx, repl, mention, flags=re.I)

    # fix any mentions of Xst Xnd, Xrd
    mention = re.sub("X(st|nd|rd)", "X", mention, flags=re.I)

    # return f'|{mention.strip()}|'
    return f"<TIMEX3>{mention.strip()}</TIMEX3>"


def anonymize_digits(text, tag_name):
    mention = re.sub("\d", "X", text).strip()
    return f"<{tag_name}>{mention}</{tag_name}>"


anonymize = {
    "person": anonymize_person,
    "pager": partial(anonymize_digits, tag_name="PAGER"),
    "timex": anonymize_datetime,
    "hours": partial(anonymize_digits, tag_name="TIMEX3"),
}


def anon_impression(df_cohort):
    acc2anon_text = {}
    for row in tqdm.tqdm(df_cohort.itertuples(), total=len(df_cohort)):
        text = row.impression

        # tag PHI variables
        spans = person.tag(text)
        spans += timex.tag(text)
        spans += pager.tag(text)
        spans += hours.tag(text)

        anon_text = text
        # replace PHI mentions with anonymous tags
        if spans:
            for start, end, stype in spans:
                mention = text[start:end]
                anon_mention = anonymize[stype](mention)
                # normalize whitepace and lettercase
                anon_mention = re.sub(r"\s+", " ", anon_mention).upper()
                anon_text = anon_text.replace(mention, anon_mention)

        acc2anon_text[row.report_id] = anon_text
    return acc2anon_text


def anon_report(df_cohort):
    acc2anon_report = {}
    for row in tqdm.tqdm(df_cohort.itertuples(), total=len(df_cohort)):
        text = row.rad_report

        # tag PHI variables
        spans = person.tag(text)
        spans += timex.tag(text)
        spans += pager.tag(text)
        spans += hours.tag(text)

        anon_text = text
        # replace PHI mentions with anonymous tags
        if spans:
            for start, end, stype in spans:
                mention = text[start:end]
                anon_mention = anonymize[stype](mention)
                # normalize whitepace and lettercase
                anon_mention = re.sub(r"\s+", " ", anon_mention).upper()
                anon_text = anon_text.replace(mention, anon_mention)

        acc2anon_report[row.report_id] = anon_text
    return acc2anon_report


id2anon_impression = anon_impression(df_train)
id2anon_report = anon_report(df_train)
df_train["anon_impression"] = df_train["report_id"].apply(
    lambda x: id2anon_impression[x]
)
df_train["anon_report"] = df_train["report_id"].apply(lambda x: id2anon_report[x])

id2anon_impression = anon_impression(df_val)
id2anon_report = anon_report(df_val)
df_val["anon_impression"] = df_val["report_id"].apply(lambda x: id2anon_impression[x])
df_val["anon_report"] = df_val["report_id"].apply(lambda x: id2anon_report[x])

id2anon_impression = anon_impression(df_test)
id2anon_report = anon_report(df_test)
df_test["anon_impression"] = df_test["report_id"].apply(lambda x: id2anon_impression[x])
df_test["anon_report"] = df_test["report_id"].apply(lambda x: id2anon_report[x])


# In[ ]:


df.columns


# In[ ]:


df_test = df_test[
    [
        "ACCESSION_NUMBER",
        "pe_acute_x",
        "pe_subsegmentalonly_x",
        "pe_positive_x",
        "anon_report",
        "anon_impression",
    ]
]
df_val = df_val[
    [
        "ACCESSION_NUMBER",
        "pe_acute_x",
        "pe_subsegmentalonly_x",
        "pe_positive_x",
        "anon_report",
        "anon_impression",
    ]
]
df_train = df_train[
    [
        "ACCESSION_NUMBER",
        "pe_acute_x",
        "pe_subsegmentalonly_x",
        "pe_positive_x",
        "anon_report",
        "anon_impression",
    ]
]

df_test.columns = [
    "Accession",
    "pe_acute",
    "pe_subsegmentalonly",
    "pe_positive",
    "anon_report",
    "anon_impression",
]
df_train.columns = [
    "Accession",
    "pe_acute",
    "pe_subsegmentalonly",
    "pe_positive",
    "anon_report",
    "anon_impression",
]
df_val.columns = [
    "Accession",
    "pe_acute",
    "pe_subsegmentalonly",
    "pe_positive",
    "anon_report",
    "anon_impression",
]


# In[ ]:


df_train.to_csv("./imon_cohort_anon/train.csv", index=False)
df_val.to_csv("./imon_cohort_anon/val.csv", index=False)
df_test.to_csv("./imon_cohort_anon/test.csv", index=False)


# In[ ]:


pd.concat([df_train, df_val, df_test]).to_csv(
    "./imon_cohort_anon/imon_cohort_all.csv", index=False
)
