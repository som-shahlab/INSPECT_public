# Pulmonary Hypertension

Annotation Guidelines: v0.2.1  Updated: May 20th, 2023

# 1: ICD Labels

SNOMED: 87837008, 88223008, 697915006, 70995007
ICD-9: 416, 416.0, 416.1, 416.2, 416.8, 416.9,
ICD-10: I27.0, I27.1, I27.2, I27.20, I27.21, I27.22, I27.23, I27.24, I27.29, I27.81, I27.82, I27.83, I27.89, I27.9

# 2: NLP Labels

## Current Positive
Explicit mention of “pulmonary hypertension” (or synonym) that refers to a currently ongoing condition. 
Definitive statement of presence of pulmonary hypertension 

## Current Positive Hedged
Explicit mention of a patient having “pulmonary hypertension” but with some hedging.

Examples
Pt may have component of heart failure and/or pulmonary hypertension
is suggestive of Pulm HTN

## Current Negative
Definitive declaration of no pulmonary hypertension 

## Current Negative Hedged
Explicit mention of a patient not having “pulmonary hypertension” but with some hedging.

## Current Hypothetical
Use of “pulmonary hypertension” to refer to a diagnosis that may happen, but hasn’t yet. PH might be suspected (e.g., in the case of ordering a test to rule in or out) but no results are available to draw a conclusion. This differs from hedging in that the clinician has not yet observed evidence that allows them to draw a preliminary conclusion. 

Examples
providing anesthesia to patients who have pulmonary hypertension.
which may be seen in the setting of pulmonary hypertension
We realize that sildenafil (Revatio) for pulmonary hypertension is approved by the FDA at a dose of 20 mg TID
c/f possible heart failure and/or pulmonary hypertension
TTE to rule out pulm HTN /RVH

## Current Unclear
Conflicting inclusion criteria or statements of presence/absence of pulmonary hypertension. For example, “pulmonary hypertension” may occur in a list, but the list header is missing. This makes it difficult to predict if the mention is current or historical.

## N/A
There is no explicit mention of pulmonary hypertension



NOTE: If multiple mentions are included (e.g., historical measurement values) label those as well. This can result in settings where both options (-RV and +RV) are tagged for a patient document.

### PAP <25
Pulmonary Artery Pressure (PAP) <25 mmHg

### PAP ≥25
Pulmonary Artery Pressure (PAP) ≥25 mmHg

### RVSP <=45
Right Ventricular Systolic Pressure (RVSP) <= 45

### RVSP >45
Right Ventricular Systolic Pressure (RVSP) > 45

### -RV Dysfunction
No suggestion of RV Dysfunction

### +RV Dysfunction
Any suggestion of RV Dysfunction


## Historical Positive
Mention of “pulmonary hypertension” refers to a prior diagnosis. This can occur in the summary of past medical history or a statement of a prior diagnosis. Conditions denoted as chronic are by definition both current and historical.

## Historical Negative
Mentions “pulmonary hypertension” being ruled out in the past. 


## OtherMeaning
Use of “pulmonary hypertension” as a modifier, adjective, or proper noun not reflecting a patient’s actual disease state. For example, we can have proper nouns (Stanford Pulmonary Hypertension Clinic) or adjectives for specialities (pulmonary hypertension fellow)

Examples
pulmonary htn fellow

Follow Up Instructions Order Comments: Please call Stanford Pulmonary Hypertension clinic

in the pulmonary hypertension clinic

