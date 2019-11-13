""" Utils functions for general preprocessing tasks """
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import pandas as pd
import scipy.sparse as sparse
import numpy as np
import scipy as sp

AGE_CUTOFFS = [25, 35, 45, 55, 60, 65]
NUM_AGE_BUCKETS = len(AGE_CUTOFFS)
ccs_path = "preprocess/icd10cm_to_ccs.csv"
sdh_table = "preprocess/sdh_variables.csv"
SDH2NORM = {"population_african": ("population_norm", "Population African American, %"),
            "population_asian": ("population_norm", "Population Asian, %"),
            "population_hispanic": ("population_norm", "Population Hispanic or Latino, %"),
            "population_native": ("population_norm", "Population American Indian and Alaska Native, %"),
            "population_white": ("population_norm", "Population White, %"),
            "parents": ("parents_norm", "Families with Single Parent, %"),
            "english": ("english_norm", "Population Speak English Less than \"Very Well\", %"),
            "household": (None, "Median Income in the Past 12 Months, $"),
            "highschool": ("education_norm", "Population Obtained High School Diploma, %"),
            "bachelors": ("education_norm", "Population Obtained Bachelor's Degree, %"),
            "poverty_50": ("poverty_norm", "Families Under 0.5 Ratio of Income to Poverty Level in the Past 12 Months, %"),
            "poverty_75": ("poverty_norm", "Families Between 0.5 and 0.74 Ratio of Income to Poverty Level in the Past 12 Months, %"),
            "poverty_99": ("poverty_norm", "Families Between 0.75 and 0.99 Ratio of Income to Poverty Level in the Past 12 Months, %"),
            "food": ("food_norm", "Families Received Food Stamps/Snap in the Past 12 months, %"),
            "woHealth": ("woHealth_norm", "Population Without Health Insurance Coverage, %"),
            "unemployment": ("unemployment_norm", "Population Unemployed, %"),
            "gini": (None, "Gini Index of Income Inequality")}
OPTUM_ZIPCODE = "Zipcode_5"
OPTUM_ZIP_UNK_KEY = -99999
OPTUM_ZIP_UNK = 'OPTUM_ZIP_UNK'
SDH_TABLE = "./preprocess/sdh_variables.csv"


def load_and_normalize_sdh():
    """Loads the output of acs_query.R (at SDH_TABLE), aggregates from geoid to zip,
    then normalizes using the normalizers defined in constants (SDH2NORM).
    Additionally adds a row for unknown zip codes and a column to indicate this."""

    sdh_table = pd.read_csv(SDH_TABLE, dtype={'zip': str})
    sdh_table.rename(columns={'zip': OPTUM_ZIPCODE},
                     inplace=True)

    for sdh_var, (sdh_norm, _) in SDH2NORM.items():
        if sdh_norm:
            sdh_table[sdh_var] = sdh_table[sdh_var] / sdh_table[sdh_norm]

    sdh_table = sdh_table[list(SDH2NORM.keys()) + [OPTUM_ZIPCODE]]

    median_sdh = sdh_table.median()
    sdh_table.fillna(median_sdh, inplace=True)

    # Convert ZIP to int
    sdh_table[OPTUM_ZIPCODE] = sdh_table[OPTUM_ZIPCODE].astype(float).astype(int)

    # Add column where unknown zip is 1 and known zips are 0
    sdh_table[OPTUM_ZIP_UNK] = 0

    # Add row where unknown zip corresponds to median sdh
    median_df = pd.DataFrame(median_sdh).T
    median_df[OPTUM_ZIPCODE] = OPTUM_ZIP_UNK_KEY
    median_df[OPTUM_ZIP_UNK] = 1
    sdh_table = sdh_table.append(median_df).reset_index(drop=True)

    return sdh_table


def load_age_sex(age, sex):
    """Assign an index of an age-sex bucket to an age and sex.
    F [0, 2) is index 0.
    F [2, 6) is index 1.
    ...
    M [0, 2) is index num_age_buckets
    M [2, 6) is index num_age_buckets + 1
    ...
    age is between 0 and 94, sex is either 0 (female) or 1 (male).
    """
    index = 0
    for age_cutoff in AGE_CUTOFFS:
        if age < age_cutoff:
            return index + sex * NUM_AGE_BUCKETS
        index += 1

    print("Warning: Age outside of [0, 95). Treating as final age bucket.")
    return (index - 1) + sex * NUM_AGE_BUCKETS


def load_icd2ccs(path):

    icd10_to_ccs = pd.read_csv(path)
    icd10_to_ccs = icd10_to_ccs.append({"ICD10CM": "OPTUM_DIAGMAP_KEY_ERROR_UNK",
                                        "CCS": -1},
                                        ignore_index=True)

    ccs_dummies = pd.get_dummies(icd10_to_ccs["CCS"])
    ccs_codes = list(ccs_dummies)

    icd10_to_ccs["CCS"] = list(sparse.csr_matrix(ccs_dummies))
    icd10_to_ccs = icd10_to_ccs.set_index("ICD10CM")

    assert len(set(icd10_to_ccs["CCS"].apply(lambda x: x.shape))) == 1
    return icd10_to_ccs, ccs_codes


def get_diag_features(df):
    # explode out ICD10
    icd = df['ICD10'].str.split(",")
    icd = icd.apply(pd.Series)
    icd['patid'] = icd.index
    icd = icd.melt(id_vars='patid').drop(['variable'], axis=1)
    icd = icd.rename({'value':"ICD10CM"}, axis=1)
    icd = icd[~icd['ICD10CM'].isna()]
    icd['ICD10CM'] = icd['ICD10CM'].str.replace(".", "")

    # convert to CCS
    icd2ccs, diag_names = load_icd2ccs('preprocess/icd10cm_to_ccs.csv')
    icd10codes = icd2ccs.index
    icd['ICD10CM'].where(icd['ICD10CM'].isin(icd10codes), "OPTUM_DIAGMAP_KEY_ERROR_UNK", inplace=True)
    icd['CCS'] = icd2ccs.loc[icd['ICD10CM']].reset_index(drop=True).values.flatten()

    # aggregate within patients
    icd = icd.groupby('patid').sum()
    diag = sp.vstack(icd['CCS'])
    
    return diag
    


def onehot_sparseify(series, get_features=False):
    """
    Transform pandas series into a sparse matrix of onehot encoding.
    Maps NaN to an empty vector (by dropping it).
    :param series: Series with day (Categorical series if necessary)
    :param get_features: Boolean for returning feature names
    :return: A sparse representation of the series in a Series
    """
    one_hot = pd.get_dummies(series, sparse=True).to_sparse()
    one_hot = one_hot.reset_index(drop=True)
    sparse_matrix = sparse.csr_matrix(one_hot.to_coo())
    if get_features:
        return sparse_matrix, one_hot.columns.values
    else:
        return sparse_matrix
