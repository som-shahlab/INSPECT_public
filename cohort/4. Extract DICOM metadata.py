from pathlib import Path 
from collections import defaultdict

import tqdm
import pandas as pd
import pydicom
import argparse


import pandas as pd
import pydicom
import torch
import glob
import os, os.path as osp

from torch.utils.data import Dataset, DataLoader





KEYS = [
    'AccessionNumber',
    'PatientID',
    'SOPInstanceUID', 
    'SeriesInstanceUID', 
    'StudyInstanceUID', 
    'InstanceNumber', 
    'ImagePositionPatient',
    'PixelSpacing', 
    'RescaleIntercept', 
    'RescaleSlope', 
    'WindowCenter', 
    'WindowWidth',
    'Manufacturer',
    'SliceThickness',
]

N_SPLITS = 100

class Metadata(Dataset):
    def __init__(self,
                 dcmfiles):
        self.dcmfiles = dcmfiles
    def __len__(self): return len(self.dcmfiles)
    def __getitem__(self, i):
        dcm = pydicom.dcmread(self.dcmfiles[i], stop_before_pixels=True)
        metadata = {}
        for k in KEYS:
            try:
                att = getattr(dcm, k)
                if k in ['InstanceNumber', 'RescaleSlope', 'RescaleIntercept']:
                    metadata[k] = float(att)
                elif k in ['PixelSpacing', 'ImagePositionPatient']:
                    for ind, coord in enumerate(att):
                        metadata[f'{k}_{ind}'] = float(coord)
                else:
                    metadata[k] = str(att)
            except Exception as e:
                print(e)
        metadata['filepath'] = '/'.join(self.dcmfiles[i].split('/')[-3:])
        return pd.DataFrame(metadata, index=[0])

def group_items(lst, n):
    """Groups items in a list into n sub-lists"""
    # Calculate the length of each sub-list
    sub_length = len(lst) // n
    
    # Initialize an empty list to store the result
    result = []
    
    # Use a loop to slice the list into sub-lists of equal length
    for i in range(n):
        start = i * sub_length
        end = (i + 1) * sub_length
        sub_list = lst[start:end]
        result.append(sub_list)
    
    return result

def main(args):

    # get all files
    files = glob.glob(osp.join(args.dir, '*/*/*.dcm'))
    files = group_items(files, N_SPLITS)

    for i,f in tqdm.tqdm(enumerate(files), total=len(files)):
        # create dataloader
        dset = Metadata(f)
        loader = DataLoader(dset, batch_size=1, shuffle=False, num_workers=8, collate_fn=lambda x: x)

        # get all metadata
        meta = []
        for data in tqdm.tqdm(loader, total=len(loader)): 
            meta += [data[0]]

        # join metadata df  
        meta_df = pd.concat(meta, axis=0, ignore_index=True)

        # count number of slices
        unique_series = pd.DataFrame(meta_df.SeriesInstanceUID.value_counts()).reset_index()
        unique_series.columns = ['SeriesInstanceUID', 'num_slices']
        meta_df = meta_df.merge(unique_series, on='SeriesInstanceUID')

        # save to file
        meta_df.to_csv(f'{i}_{args.outfile}', index=False)
        del meta_df


    dcms = []
    for i in tqdm.tqdm(range(N_SPLITS), total=N_SPLITS):
        df = pd.read_csv(f"{i}_{args.outfile}")
        dcms.append(df)
    df = pd.concat(dcms, ignore_index=True)
    df.to_csv(args.outfile, index=False)
    print(f"Dataframe saved to {args.outfile}")
    ######################################################################


    ### Filter slices by cohort studies ##################################
    #df_cohort = pd.read_csv('radfusion3_cohort.csv')
    df_cohort = pd.read_csv('radfusion3_cohort_w_anon_report.csv')
    print(df_cohort.shape)
    df_crosswalk = pd.read_csv('crosswalk-2016-2022.csv')
    df_crosswalk2 = pd.read_csv('crosswalk-2000-2016.csv')
    df_crosswalk3 = pd.read_excel('RSNA_Crosswalk.xlsx')

    acc2anon = dict(zip(df_crosswalk.accession, df_crosswalk.anon_accession))
    acc2anon.update(dict(zip(df_crosswalk2.ACCESSION, df_crosswalk2.ANON_ACCESSION)))
    acc2anon.update(dict(zip(df_crosswalk3.accession, df_crosswalk3.anon_accession)))

    df_cohort['anon_accession'] = df_cohort['Accession'].apply(
        lambda x: acc2anon[x] if x in acc2anon else None
    )
    print(f"AnonAcc: {df_cohort.anon_accession.nunique()} / Acc: {df_cohort.Accession.nunique()}" )
    df_cohort[~df_cohort.anon_accession.isna()]
    #df_cohort.to_csv("radfusion3_cohort.csv", index=False)
    print(f"AnonAcc: {df_cohort.anon_accession.nunique()} / Acc: {df_cohort.Accession.nunique()}" )

    # remove duplicated order_proc
    d = df_cohort.groupby('Accession').nunique().reset_index()
    acc = d[d.order_proc_id > 1].Accession.unique()
    print(f"AnonAcc: {df_cohort.anon_accession.nunique()} / Acc: {df_cohort.Accession.nunique()}" )
    df_cohort = df_cohort[~df_cohort.Accession.isin(acc)]
    print(f"AnonAcc: {df_cohort.anon_accession.nunique()} / Acc: {df_cohort.Accession.nunique()}" )
    df_cohort.to_csv('radfusion3_cohort_w_filtered.csv', index=False)
    
    ### Select studies ##################################################
    df = pd.read_csv(args.outfile)

    # keep only studies in cohort
    print('='*80)
    print('In Cohort')
    print('='*80)
    print(df.AccessionNumber.nunique())
    cohort_anon_accession = set(df_cohort['anon_accession'].tolist())
    df = df[df.AccessionNumber.isin(cohort_anon_accession)]
    print(df.AccessionNumber.nunique())

    # only keep studyes with slices between 1-3mm
    print('='*80)
    print('Slice Thickness')
    print('='*80)
    print(df.AccessionNumber.nunique())
    df = df[(df.SliceThickness >= 1.0) & (df.SliceThickness <= 3.0)]
    print(df.AccessionNumber.nunique())

    # only keep studyes with slices between 1-3mm
    print('='*80)
    print('Num Slices')
    print('='*80)
    print(df.AccessionNumber.nunique())
    df = df[(df.num_slices >= 50) & (df.num_slices <= 600)]
    print(df.AccessionNumber.nunique())

    # sort studies by resver slice thickness
    df = df.sort_values(by=['SliceThickness'], ascending=False)

    print(f'Before filtering: {df.AccessionNumber.nunique()}')
    df_study = df.groupby('StudyInstanceUID').head(1)
    print(f'After filtering: {df_study.AccessionNumber.nunique()}')


    # remove series with multiple num_slices 
    d = df_study.groupby('SeriesInstanceUID').nunique().reset_index()
    d = d[d.num_slices > 1] 
    remove_series = d.SeriesInstanceUID.unique()
    print(f'Before removing duplicate series: {df_study.AccessionNumber.nunique()}')
    df_study = df_study[~df_study.SeriesInstanceUID.isin(remove_series)]
    print(f'After removing duplicate series: {df_study.AccessionNumber.nunique()}')

    # remove Accession with multiple series
    d = df_study.groupby('AccessionNumber').nunique().reset_index()
    d = d[(d.SeriesInstanceUID > 1) | (d.StudyInstanceUID > 1)] 
    remove_acc = d.AccessionNumber.unique()
    print(f'Before removing duplicate acc: {df_study.AccessionNumber.nunique()}')
    df_study = df_study[~df_study.AccessionNumber.isin(remove_acc)]
    print(f'After removing duplicate acc: {df_study.AccessionNumber.nunique()}')

    df_study.to_csv('pe_dicom_meta_study_series_selected.csv', index=False)
    print(df_study.num_slices.max(), df_study.num_slices.min())
    print('study_series_selected', df_study.AccessionNumber.nunique(), df_study.SeriesInstanceUID.nunique(), df_study.StudyInstanceUID.nunique())

    # only keep selected studies
    selected_series = df_study.SeriesInstanceUID.unique()
    print(df.SeriesInstanceUID.nunique())
    df = df[df.SeriesInstanceUID.isin(selected_series)]
    print(df.SeriesInstanceUID.nunique())
    df.to_csv(f"pe_dicom_meta_cohort.csv", index=False)
    print(df.num_slices.max(), df.num_slices.min())
    print('instance selected', df.AccessionNumber.nunique(), df.SeriesInstanceUID.nunique(), df.StudyInstanceUID.nunique(), df.shape[0])

    # only keep selected studies in cohort
    df_cohort = pd.read_csv('radfusion3_cohort_w_filtered.csv')
    df_cohort = df_cohort[df_cohort['anon_accession'].isin(df_study.AccessionNumber.unique())]
    df_cohort.to_csv('radfusion3_cohort_final.csv', index=False)
    print('cohort', df_cohort['anon_accession'].nunique(), df.shape[0])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='/home/mschuang/data/pe_all')
    parser.add_argument('--outfile', type=str, default='pe_dicom_meta.csv')
    args = parser.parse_args()

    main(args)

