""" Utils functions for general preprocessing tasks """
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import pandas as pd
import pyarrow as pa
import scipy.sparse as sparse
import dask.dataframe as dd
import numpy as np


ccs_path = "preprocess/icd10cm_to_ccs.csv"

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
