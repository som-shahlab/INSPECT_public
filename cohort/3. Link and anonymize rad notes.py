#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


df_cohort = pd.read_csv('./radfusion3_cohort.csv')


# ### Read in RIT provided reports and format column names 

# In[ ]:


df_report_1 = pd.read_csv('/Users/marshuang/Labs/Lungren/PE_CT_data_cleaning/stanford/pe_radreports_00_16.csv', sep='\t', dtype={'ACCESSION_NUMBER': object})
df_report_1 = df_report_1.rename(columns={"ACCESSION_NUMBER": "Accession", 'REPORT': 'Report', 'RAD_DATE': 'Report_Date'})

# Assume your dataframe is df and the column with dates is 'date_column'
df_report_1['Report_Date'] = pd.to_datetime(df_report_1['Report_Date'], format='%d-%b-%y %I.%M.%S.%f %p')

# Now you can format it to the desired string format
df_report_1['Report_Date'] = df_report_1['Report_Date'].dt.strftime('%Y-%m-%d %H:%M:%S').astype(str)


# In[ ]:


df_report_2 = pd.read_csv('/Users/marshuang/Labs/Lungren/PE_CT_data_cleaning/stanford/pe_radreports_16_22.csv')
df_report_2 = df_report_2.rename(columns={"accession_number": "Accession", 'report': 'Report', 'result_time': 'Report_Date'})

df_report_2['Report'] = df_report_2['Report'].apply(lambda x: x.replace('\n', ''))


# #### Check if overlapping accession have same report

# In[ ]:


import difflib


df2_accs = set(df_report_2['Accession'].tolist())

for idx, row in tqdm.tqdm(df_report_1.iterrows(), total=len(df_report_1)):
    if row['Accession'] in df2_accs:
        df2_report = df_report_2[df_report_2['Accession'] == row['Accession']].Report.iloc[0]
        df1_report = row['Report']

        if df1_report not in df2_report: 
            print(row['Accession'])


# ### Create accession to report mapping

# In[ ]:


acc2report = dict(zip(df_report_1.Accession, df_report_1.Report))
acc2report.update(dict(zip(df_report_2.Accession, df_report_2.Report)))

acc2report_date = dict(zip(df_report_1.Accession, df_report_1.Report_Date))
acc2report_date.update(dict(zip(df_report_2.Accession, df_report_2.Report_Date)))

df_cohort['Report'] = df_cohort['Accession'].apply(lambda x: acc2report[x] if x in acc2report else None)
print(f"{len(df_cohort[~df_cohort.Report.isna()])}/{len(df_cohort)}")


df_cohort['ReportDate'] = df_cohort['Accession'].apply(lambda x: acc2report_date[x] if x in acc2report_date else None)
print(f"{len(df_cohort[~df_cohort.ReportDate.isna()])}/{len(df_cohort)}")


# In[ ]:


import re

def get_impression_section_start(text):
    """
    Rule-based approach to determine absolute char 
    offset of the impression section
    """
    # patterns are in order of precedence 
    regexs = [
        re.compile(r'''impression[:]''', re.I),
        re.compile(r'''impression[:;]''', re.I),
        re.compile(r'''IMPRESSSION:''', re.I),
        re.compile(f'''Impression[:;]''', re.I),
        re.compile(r'''conclusion[:;]''', re.I),
        re.compile(r"  1\.", re.I),
        re.compile(r" : 1\.", re.I),

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
        re.compile(r'''(END (OF )*IMPRESSION( SUMMARY)*[:.]*|SUMMARY( CODE)*[:.])''', re.I),
        re.compile(r'''SUMMARY CODE [0-9]|I have personally reviewed''', re.I),
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
    if start is None:
        #return None
        start = None
    # if no match is found, return the end of the note string
    end = get_impression_section_end(text)
    if end is None:
        end = len(text)
    return start, end


def get_impression(text):
    start, end = get_impression_section_offsets(text)
    if start is None: 
        return None
    return text[start: end]

df_cohort['Impression'] = df_cohort['Report'].apply(lambda x: get_impression(x))


# ### Remove missing impressions

# In[ ]:


print(f'Original cohort size: {df_cohort.shape[0]}')

print(f"Remove cases without impression: {df_cohort[df_cohort['Impression'].isna()].shape[0]}")
df_cohort = df_cohort[~df_cohort['Impression'].isna()]

print(f"Remove cases with empty impression: {df_cohort[df_cohort['Impression'] == ''].shape[0]}")
df_cohort = df_cohort[df_cohort['Impression'] != '']

print(f'Final cohort size: {df_cohort.shape[0]}')


# In[ ]:


df_cohort = pd.read_csv('./radfusion3_cohort_final.csv')
csv_to_bq_table('./radfusion3_cohort_final.csv', 'som-nero-phi-nigam-starr', 'pulmonary_embolisms', 'radfusion3_cohort_final')


# In[ ]:


df_rsna = pd.read_csv('../data/stanford_studies_in_rsna_train_with_labels.csv', dtype={'Mrn': object, 'Acc': object})
df_rsna['pe_label'] = df_rsna['negative_exam_for_pe'].apply(lambda x: 1 if x == 0 else 0)
acc2pe = dict(zip(df_rsna.Acc, df_rsna.pe_label))
acc2central = dict(zip(df_rsna.Acc, df_rsna.central_pe))

#df_nlp[['RadfusionPELabel', ]]
df_rsna.columns


# In[ ]:


df_nlp = df_cohort[(df_cohort.RSNA == True) | (df_cohort.Radfusion == True)]
df_nlp['rsna_positive_pe'] = df_nlp['Accession'].apply(lambda x: acc2pe[x] if x in acc2pe else None)
df_nlp['rsna_central_pe'] = df_nlp['Accession'].apply(lambda x: acc2pe[x] if x in acc2central else None)


# In[ ]:


df_nlp = df_nlp[['anon_accession', 'AnonReport', 'AnonImpression', 'RadfusionPELabel', 'RadfusionPEType', 'rsna_positive_pe', 'rsna_central_pe']]
df_nlp['RadfusionCentralPE'] = df_nlp['RadfusionPEType'].apply(lambda x: 1 if x == 'central' else 0)
df_nlp['positive_pe'] = df_nlp.apply(lambda x: x.rsna_positive_pe if x.RadfusionPELabel != x.RadfusionPELabel else x.RadfusionPELabel, axis=1)
df_nlp['central_pe'] = df_nlp.apply(lambda x: x.rsna_central_pe if x.RadfusionCentralPE != x.RadfusionCentralPE else x.RadfusionCentralPE, axis=1)

df_nlp = df_nlp[['anon_accession', 'AnonReport', 'AnonImpression', 'positive_pe', 'central_pe']]
                 


# In[ ]:


df_nlp = df_nlp.sample(frac=1.0)
df_nlp['Split'] = 'train'
split_size = len(df_nlp) // 5
df_nlp.iloc[-2*split_size:-split_size]['Split'] = 'val'
df_nlp.iloc[-split_size:]['Split'] = 'test'
df_nlp.Split.value_counts()


# In[ ]:


df_nlp[df_nlp.Split == 'train'].to_csv('rsna_radfusion/rsna_radfusion_longformer_train.csv')
df_nlp[df_nlp.Split == 'val'].to_csv('rsna_radfusion/rsna_radfusion_longformer_val.csv')
df_nlp[df_nlp.Split == 'test'].to_csv('rsna_radfusion/rsna_radfusion_longformer_test.csv')


# In[ ]:


print('person_id', df_cohort.person_id.nunique())
print('MRN', df_cohort.MRN.nunique())
print('Accession', df_cohort.Accession.nunique())
#print('AnonAccession', df_cohort.anon_accession.nunique())
print('rows', df_cohort.shape[0])


# ## De-identify

# In[ ]:


# to remove
to_remove = "Physician to Physician Radiology Consult Line: (650) 736-1173"

df_cohort['Impression'] = df_cohort['Impression'].apply(lambda x: x.replace(to_remove, ''))


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


meridiems   = r'''([APap][Mm]|[APap][.][Mm][.])'''            # AM | PM
clock_time  = r'''([0-2]*[0-9][:][0-5][0-9])(:[0-5][0-9])*''' # 10:00 | 10:10:05

year_range  = r'''(19[0-9][0-9]|20[012][0-9])'''             # 1900 - 2029
day_range   = r'''(3[01]|[12][0-9]|[0]*[1-9])'''             # 01 - 31
month_range = r'''([1][012]|[0]*[1-9])'''                    # 01 - 12

month_full  = r'''(january|february|march|april|may|june|july|august|september|october|november|december)'''
month_abbrv = r'''((jan|feb|mar|apr|may|june|jul|aug|sep|sept|oct|nov|dec)[.]*)'''
day_full    = r'''(monday|tuesday|wednesday|thursday|friday|saturday|sunday)'''
day_abbrv   = r'''(mon|tue|wed|thu|fri|sat|sun)[.]*'''

rgx_time  = r'''({clock_time})(\s*{meridiems})*'''.format(clock_time=clock_time, meridiems=meridiems)
rgx_month = r'''({month_full}|{month_abbrv})'''.format(month_full=month_full, month_abbrv=month_abbrv)
rgx_day   = r'''({day_full})'''.format(day_full=day_full, day_abbrv=day_abbrv) 

# Dates with Numeric Months
d_params = {'M':month_range, 'D':day_range, 'Y':year_range, 'T':rgx_time}

rgx_y    = r'''\b({Y})\b'''.format(**d_params)                                  # 2019
rgx_mdy  = r'''({M}[-]{D}[-]{Y}|{M}[/]{D}[/]{Y})(\s*{T})*'''.format(**d_params) # 1/11/2000 12:30 PM
rgx_ymd  = r'''({Y}[/]{M}[/]{D}|{Y}[-]{M}[-]{D})(\s*{T})*'''.format(**d_params) # 2000-11-12 12:30 PM
rgx_md   = r'''\b(([1][0-9]|[0][1-9])[/][0-3][0-9])\b'''                        # 11/31 (implied year)
rgx_ys   = r'''\b((mid[-])*({Y}|the [2-9]0)[s])\b'''.format(**d_params)         # 1920s | mid-1990s | the 80s
rgx_mdy2 = r'''\b({M}[-]{D}[-]([012][0-9])|{M}[/]{D}[/]([012][0-9]))(\s*{T})*\b'''.format(**d_params) # 9/21/09

# Dates with String Months
s_params = {'M':rgx_month, 'D':day_range, 'Y':year_range, 'T':rgx_time}

rgx_dash_mdy   = r'''({M}[-]{D}[-]{Y})(\s*{T})*'''.format(**s_params)           # Apr-03-2011 13:21:30
rgx_m_of_y     = r'''({M} of {Y}|{Y} in {M})'''.format(**s_params)              # January of 2018 | 2005 in April
rgx_month_dy   = r'''({M}[ ]*{D}(st|nd|rd|th)*[, ]*{Y})'''.format(**s_params)   # July 30, 2019
rgx_concat_mdy = r'''({D}{M}{Y})'''.format(**s_params)                          # 30Jan2019
rgx_d_of_my    = r'''({D}(st|nd|rd|th)* of {M}(\s{Y})*)'''.format(**s_params)   # 20th of July 2010
rgx_month_d    = r'''\b({M} {D}(st|nd|rd|th)*)\b'''.format(**s_params)          # September 16
rgx_month_year = r'''\b({M}\s*{Y})\b'''.format(**s_params)                      # May 2010

# Other TIMEX3 expressions
number_0_10  = r'''(zero|one|two|three|four|five|six|seven|eight|nine|ten)'''
number_11_19 = r'''((thir|four|fif|six|seven|eigh|nine)teen)'''
number_20_90 = r'''((twenty|thirty|fourty|fifty|sixty|seventy|eighty|ninty)([-]*{N})*)'''.format(N=number_0_10)

# five years ago
rgx_number_full  = r'''{}|{}|{}'''.format(number_0_10, number_11_19, number_20_90)
rgx_timex_ago    = r'''\b(([1-9][0-9]*([.][5])*|[1-9][0-9]*\s*(to|[-])\s*[1-9][0-9]*|({})|few|a) ((year|month|wk|week|day|hour)[s]*) (ago|back|prior))\b'''.format(rgx_number_full)

# relative temporal expressions
rgx_day_parts  = r'''\b((this) (morning|afternoon|evening)|(yesterday|today|tomorrow|tonight|tonite)[']*[s]*)\b'''
rgx_day_times  = r'''\b(now|currently|presently)\b'''
rgx_day_rela   = r'''\b((next|last|this) ({W}|week|month|year))\b'''.format(W=day_full)
rgx_recent_now = r'''\b((current|recent)(ly)*|at this (point|time)|now)\b'''
rgx_operative  = r'''\b((pre|post|intra)[-]*(operative(ly)*|op))\b'''

timex_regexes = [
    # r'''\b{}\b'''.format(rgx_time),
    r'''\b{}\b'''.format(rgx_month),
    r'''\b{}\b'''.format(rgx_day),

    rgx_time,
    #rgx_month,
    #rgx_day,
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
    
    #rgx_timex_ago,
    #rgx_day_parts,
    #rgx_day_rela,
    
    #rgx_recent_now,
    #rgx_operative
]


# In[ ]:


#from .timex import regexes as timex_regexes


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
        matches = sorted(matches, key=lambda x:x[-1], reverse=True)
        
        stack = []
        for a in matches:
            if not stack:
                stack.append(a)
            else:
                b = stack.pop()
                if is_overlapping(a,b):
                    start,end = min(a[0], b[0]), max(a[-1], b[-1])
                    stack.append((start,end))
                else:
                    stack += [b,a]
        
        # strip trailing whitespace
        for i in range(len(stack)):
            start,end = stack[i]
            while text[start:end][-1] == ' ':
                end -= 1
            stack[i] = (start, end, self.name)
        
        #return matches
        return stack


# load dictionary of names (may include PHI)
names = set(pd.read_csv('names.txt', sep='\t').name)
name_dict_rgx =  f'({"|".join(sorted(names, key=lambda x:len(x), reverse=True))})'

name_uppercase_rgx = r'''(DR[.]*\s+(([A-Z'-]+\s*){1,2}(?=(UPON|FROM|WITH|WAS|AND|VIA|AS|TO|AT|BY|IN|OF|ON|[@.,;(]))))\b'''
name_titlecase_rgx = r'''(Dr[.]*\s+(?:(?:[A-Z'][a-z-]+)\s*){1})'''

regexes = [
    name_uppercase_rgx,
    name_dict_rgx,
    name_titlecase_rgx
]
person = RegexTagger('person', regexes, {'may'})
pager = RegexTagger('pager', [r'''PAGER [0-9-]+\b'''])
timex = RegexTagger('timex', timex_regexes, {'may'})


hours_regexes = [
    r'''([0-9]{3,4}\s*HOURS)''',
    r'''(?<=(AT ))[0-9]{3,4}'''
]
hours = RegexTagger('hours', hours_regexes)


# In[ ]:


from functools import partial 

def anonymize_person(text):
    
    m = re.search(r'''dr[.]*|PHYSICIAN''', text, re.I)
    if m is not None:
        i,j = m.span()
        anon_toks = [text[i:j]]
        toks = text[j+1:].split()
        if len(toks) == 1:
            anon_toks.append("LAST_NAME")
        elif len(toks) > 1:
            anon_toks.append("FIRST_NAME")
            anon_toks.append("LAST_NAME")
        #return f'|{" ".join(anon_toks).strip()}|'
        return f'<PERSON>{" ".join(anon_toks).strip()}</PERSON>'
    
    # first/last name
    toks = text.split()
    if len(toks) == 2:
        #return "|FIRST_NAME LAST_NAME|"
        return "<PERSON>FIRST_NAME LAST_NAME</PERSON>"
    return None

def anonymize_datetime(text):
  
    month_full  = r'''(january|february|march|april|may|june|july|august|september|october|november|december)'''
    month_abbrv = r'''((jan|feb|mar|apr|may|june|jul|aug|sep|sept|oct|nov|dec)[.]*)'''
    rgx_month = r'''({month_full}|{month_abbrv})'''.format(month_full=month_full, month_abbrv=month_abbrv)
    
    replace = {
        'AM_PM':r'''([APap][Mm]|[APap][.][Mm][.])''',
        'MONTH': rgx_month,
        'DAY': r'''(monday|tuesday|wednesday|thursday|friday|saturday|sunday)|(mon|tue|wed|thu|fri|sat|sun)[.]*'''
    }
    
    transforms = [(r'\d','X')]
    for tok,rgx in replace.items():
        m = re.search(rgx, text, re.I)
        if m:
            transforms.append((rgx, tok))
    # apply transforms
    mention = text
    for rgx,repl in transforms:
        mention = re.sub(rgx, repl, mention, flags=re.I)
    
    # fix any mentions of Xst Xnd, Xrd
    mention = re.sub('X(st|nd|rd)', 'X', mention, flags=re.I)
    
    #return f'|{mention.strip()}|'
    return f'<TIMEX3>{mention.strip()}</TIMEX3>'


def anonymize_digits(text, tag_name):
    mention = re.sub('\d','X', text).strip()
    return f"<{tag_name}>{mention}</{tag_name}>"



anonymize = {
    'person': anonymize_person,
    'pager': partial(anonymize_digits, tag_name='PAGER'),
    'timex': anonymize_datetime, 
    'hours': partial(anonymize_digits, tag_name='TIMEX3'),
    
}

acc2anon_text = {}
for row in tqdm.tqdm(df_cohort.itertuples(), total=len(df_cohort)):
    text = row.Impression

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
            anon_mention = re.sub(r'\s+', ' ', anon_mention).upper()
            anon_text = anon_text.replace(mention, anon_mention)
            
    acc2anon_text[row.Accession] = anon_text

acc2anon_report = {}
for row in tqdm.tqdm(df_cohort.itertuples(), total=len(df_cohort)):
    text = row.Report

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
            anon_mention = re.sub(r'\s+', ' ', anon_mention).upper()
            anon_text = anon_text.replace(mention, anon_mention)
            
    acc2anon_report[row.Accession] = anon_text    
    
df_cohort['AnonImpression'] = df_cohort['Accession'].apply(lambda x: acc2anon_text[x])
df_cohort['AnonReport'] = df_cohort['Accession'].apply(lambda x: acc2anon_report[x])


# In[ ]:


df_cohort.to_csv('./radfusion3_cohort_w_anon_report.csv', index=False)

