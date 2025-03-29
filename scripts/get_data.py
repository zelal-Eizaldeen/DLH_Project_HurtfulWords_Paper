import pandas as pd
import numpy as np
import psycopg2
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import Constants
import sys
from pathlib import Path

# Define the output folder

output_folder = Path(sys.argv[1])
output_folder.mkdir(parents = True, exist_ok = True)

#Replacing Database Queries with CSVs
# conn = psycopg2.connect('dbname=mimic user=haoran host=mimic password=password')

# pats = pd.read_sql_query('''
# select subject_id, gender, dob, dod from mimiciii.patients
# ''', conn)

# Paths to CSV files (replace with actual paths)
# Load CSV files instead of querying PostgreSQL
PATS_CSV = "/content/drive/MyDrive/MastersDegree/DLH/Project/HurtfulWords/Payel-DLH-related/DataFiles/mimic-iii-clinical-database-1.4/PATIENTS.csv"
ADM_CSV = "/content/drive/MyDrive/MastersDegree/DLH/Project/HurtfulWords/Payel-DLH-related/DataFiles/mimic-iii-clinical-database-1.4/ADMISSIONS.csv"
NOTES_CSV = "/content/drive/MyDrive/MastersDegree/DLH/Project/HurtfulWords/Payel-DLH-related/DataFiles/mimic-iii-clinical-database-1.4/NOTEEVENTS.csv"
ICD_CSV = "/content/drive/MyDrive/MastersDegree/DLH/Project/HurtfulWords/Payel-DLH-related/DataFiles/mimic-iii-clinical-database-1.4/DIAGNOSES_ICD.csv"
ICUSTAYS_CSV = "/content/drive/MyDrive/MastersDegree/DLH/Project/HurtfulWords/Payel-DLH-related/DataFiles/mimic-iii-clinical-database-1.4/ICUSTAYS.csv"
pats = pd.read_csv(PATS_CSV, nrows=1000)

adm = pd.read_csv(ADM_CSV, nrows=1000)


# Load the CSV with minimal memory usage
# df = pd.read_csv(NOTES_CSV, nrows=5)  # Read only a few rows
# print(df.columns)  # Print available column names

# Read CSV again with corrected column names
notes = pd.read_csv(NOTES_CSV, low_memory=False, nrows=1000)

diagnoses_icd= pd.read_csv(ICD_CSV, low_memory=False, nrows=1000)
icustays=pd.read_csv(ICUSTAYS_CSV, nrows=100)
# # Rename columns to match the original query output
pats.rename(columns={"SUBJECT_ID":"subject_id", "GENDER": "gender", "DOB": "dob", "DOD": "dod"}, inplace=True)
adm.rename(columns={"SUBJECT_ID":"subject_id", "HADM_ID": "hadm_id", "INSURANCE": "insurance", "LANGUAGE": "language","RELIGION":"religion", "ETHNICITY":"ethnicity","ADMITTIME":"admittime","DISCHTIME":"dischtime","DIAGNOSIS":"diagnosis","DEATHTIME":"deathtime"}, inplace=True)
diagnoses_icd.rename(columns={"SUBJECT_ID": "subject_id", "ROW_ID":"row_id", "HADM_ID":"hadm_id","HADM_ID":"hadm_id", "SEQ_NUM":"seq_num", "ICD9_CODE":"icd9_code"}, inplace=True)
#NOTEVENTS table
# Rename columns to match expected names
column_mapping = {
    'ROW_ID': 'note_id',
    'HADM_ID': 'hadm_id',
    'CHARTDATE': 'chartdate',
    'CHARTTIME': 'charttime',
    'CATEGORY': 'category',
    'TEXT': 'text',
    'ISERROR': 'iserror'
}
notes.rename(columns=column_mapping, inplace=True)
# Select only expected columns if they exist
expected_cols = ['note_id', 'hadm_id', 'chartdate', 'charttime', 'category', 'text', 'iserror']
available_cols = [col for col in expected_cols if col in notes.columns]
notes = notes[available_cols]
# Drop rows where 'iserror' is NOT NULL (if 'iserror' column exists)
if 'iserror' in notes.columns:
    notes = notes[notes["iserror"].isna()].drop(columns=['iserror'])


#DIAGNOSES_ICD table 
# Group by hadm_id and aggregate ICD9 codes into a list
icds = (diagnoses_icd.groupby('hadm_id')
        .agg({'icd9_code': lambda x: list(x.dropna())})  # Drop NaNs to avoid issues
        .reset_index())


icustays.rename(columns={"SUBJECT_ID": "subject_id", "ICUSTAY_ID":"icustay_id", "HADM_ID":"hadm_id","HADM_ID":"hadm_id", "INTIME":"intime", "OUTTIME":"outtime"}, inplace=True)

#This table is not availabe in the CSV files. 
# acuities = pd.read_sql_query('''
# select * from (
# select a.subject_id, a.hadm_id, a.icustay_id, a.oasis, a.oasis_prob, b.sofa from
# (mimiciii.oasis a
# natural join mimiciii.sofa b )) ab
# natural join
# (select subject_id, hadm_id, icustay_id, sapsii, sapsii_prob from
# mimiciii.sapsii) c
# ''', conn)
print(f"NOTES {notes.columns}")
print(f"PAt {pats.columns}")
print(f"Adm {adm.columns}")
print(f"Diag {diagnoses_icd.columns}")
print(f"icustays {icustays.columns}")
print(f"icds {icds.columns}")







n_splits = 12
pats = pats.sample(frac = 1, random_state = 42).reset_index(drop = True)
kf = KFold(n_splits = n_splits, shuffle = True, random_state = 42)
for c,i in enumerate(kf.split(pats, groups = pats.gender)):
    pats.loc[i[1], 'fold'] = str(c)

# adm = pd.read_sql_query('''
# select subject_id, hadm_id, insurance, language,
# religion, ethnicity,
# admittime, deathtime, dischtime,
# HOSPITAL_EXPIRE_FLAG, DISCHARGE_LOCATION,
# diagnosis as adm_diag
# from mimiciii.admissions
# ''', conn)

df = pd.merge(pats, adm, on='subject_id', how = 'inner')

def merge_death(row):
    if not(pd.isnull(row.deathtime)):
        return row.deathtime
    else:
        return row.dod
df['dod_merged'] = df.apply(merge_death, axis = 1)


# notes = pd.read_sql_query('''
# select category, chartdate, charttime, hadm_id, row_id as note_id, text from mimiciii.noteevents
# where iserror is null
# ''', conn)

# drop all outpatients. They only have a subject_id, so can't link back to insurance or other fields
notes = notes[~(pd.isnull(notes['hadm_id']))]

df = pd.merge(left = notes, right = df, on='hadm_id', how = 'left')

df.ethnicity.fillna(value = 'UNKNOWN/NOT SPECIFIED', inplace = True)

others_set = set()
def cleanField(string):
    mappings = {'HISPANIC OR LATINO': 'HISPANIC/LATINO',
                'BLACK/AFRICAN AMERICAN': 'BLACK',
                'UNABLE TO OBTAIN':'UNKNOWN/NOT SPECIFIED',
               'PATIENT DECLINED TO ANSWER': 'UNKNOWN/NOT SPECIFIED'}
    bases = ['WHITE', 'UNKNOWN/NOT SPECIFIED', 'BLACK', 'HISPANIC/LATINO',
            'OTHER', 'ASIAN']

    if string in bases:
        return string
    elif string in mappings:
        return mappings[string]
    else:
        for i in bases:
            if i in string:
                return i
        others_set.add(string)
        return 'OTHER'

df['ethnicity_to_use'] = df['ethnicity'].apply(cleanField)

# Convert to datetime_Z
df['chartdate'] = pd.to_datetime(df['chartdate'])
df['dob'] = pd.to_datetime(df['dob'])
#end_Z

df = df[df.chartdate >= df.dob]

#Added_Z to_pydatetime()  to solve date problem
ages = []
for i in range(df.shape[0]):
    ages.append((df.chartdate.iloc[i].to_pydatetime() - df.dob.iloc[i].to_pydatetime()).days/365.24)
df['age'] = ages

df.loc[(df.category == 'Discharge summary') |
       (df.category == 'Echo') |
       (df.category == 'ECG'), 'fold'] = 'NA'



# icds = (pd.read_sql_query('select * from mimiciii.diagnoses_icd', conn)
#         .groupby('hadm_id')
#         .agg({'icd9_code': lambda x: list(x.values)})
#         .reset_index())

df = pd.merge(left = df, right = icds, on = 'hadm_id')

def map_lang(x):
    if x == 'ENGL':
        return 'English'
    if pd.isnull(x):
        return 'Missing'
    return 'Other'
df['language_to_use'] = df['language'].apply(map_lang)



for i in Constants.groups:
    assert(i['name'] in df.columns), i['name']



# acuities = pd.read_sql_query('''
# select * from (
# select a.subject_id, a.hadm_id, a.icustay_id, a.oasis, a.oasis_prob, b.sofa from
# (mimiciii.oasis a
# natural join mimiciii.sofa b )) ab
# natural join
# (select subject_id, hadm_id, icustay_id, sapsii, sapsii_prob from
# mimiciii.sapsii) c
# ''', conn)

# icustays = pd.read_sql_query('''
# select subject_id, hadm_id, icustay_id, intime, outtime
# from mimiciii.icustays
# ''', conn).set_index(['subject_id','hadm_id'])

def fill_icustay(row):
    opts = icustays.loc[[row['subject_id'],row['hadm_id']]]
    if pd.isnull(row['charttime']):
        charttime = row['chartdate'] + pd.Timedelta(days = 2)
    else:
        charttime = row['charttime']
    stay = opts[(opts['intime'] <= charttime)].sort_values(by = 'intime', ascending = True)

    if len(stay) == 0:
        return None
        #print(row['subject_id'], row['hadm_id'], row['category'])
    return stay.iloc[-1]['icustay_id']
print(f"HERE {icustays}")

# df['icustay_id'] = df[df.category.isin(['Discharge summary','Physician ','Nursing','Nursing/other'])].apply(fill_icustay, axis = 1)

# df = pd.merge(df, acuities.drop(columns = ['subject_id','hadm_id']), on = 'icustay_id', how = 'left')
# df.loc[df.age >= 90, 'age'] = 91.4

# df.to_pickle(output_folder / "df_raw.pkl")
