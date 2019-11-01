""" Utils functions for general preprocessing tasks """
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import pandas as pd
import pyarrow as pa
import scipy.sparse as sparse
import dask.dataframe as dd
import numpy as np


AGE_CUTOFFS = [25, 35, 45, 55, 60, 65]
NUM_AGE_BUCKETS = len(AGE_CUTOFFS)
ccs_path = "preprocess/icd10cm_to_ccs.csv"

def sex_age_bucketer(age, sex):
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

    assert len(set(icd10_to_ccs[CCS].apply(lambda x: x.shape))) == 1
    return icd10_to_ccs, ccs_codes


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
