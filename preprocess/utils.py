""" Utils functions for general preprocessing tasks """
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import pandas as pd
import pyarrow as pa
import scipy.sparse as sparse
import dask.dataframe as dd
import numpy as np

from constants import *


def filter_quarter_ranges(quarter_range_tuples):
    """Filter range of quarters for 2016 and 2017."""
    valid_years = [YEAR2016, YEAR2017]
    return [(date1, date2) for date1, date2 in quarter_range_tuples
            if str(date1.year) in valid_years and
            str(date2.year) in valid_years]


def stringify_ranges(quarter_ranges):
    """Convert each (quarter_date1, quarter_date2) tuple to a string."""
    return [f"{date1}_{date2}" for date1, date2 in quarter_ranges]


def to_dt(date_str):
    """Convert a string to a datetime object."""
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def get_year(dt):
    """Get string year of a datetime object."""
    return str(dt.year)


def get_quarter(dt):
    """Get string quarter of a datetime object."""
    int_quarter = (dt.month - 1) // 3 + 1
    return f"Q{int_quarter}"


def get_num_days_in_quarter(year, quarter):
    """Compute the number of days in a quarter."""
    start_month = 3 * (int(quarter[-1]) - 1) + 1
    start_date = datetime(int(year), start_month, 1)
    end_date = start_date + relativedelta(months=3)

    return end_date - start_date


def contains_duplicate_headers(path):
    """Return True a file contains duplicate headers."""
    with path.open() as f:
        line1 = next(f)
        line2 = next(f)

    return line1 == line2


def contains_headers(path):
    """Return True is a file contains headers.

    Assumes OPTUM_PAT_ID is in the header for all
    files with headers."""
    with path.open() as f:
        line = next(f)

    return OPTUM_PAT_ID in line


def get_headers(path):
    """Return a list of headers of a csv."""
    headers_df = pd.read_csv(path, nrows=0)

    return list(headers_df)


def read_csv(path, **args):
    """Read a csv using the correct engine."""
    try:
        df = pd.read_csv(path, **args, engine='c', error_bad_lines=False,
                         warn_bad_lines=True)
    except Exception:
        print('Pandas read_csv failed with c engine. Trying python...')
        df = pd.read_csv(path, **args, engine='python', error_bad_lines=False,
                         warn_bad_lines=True)
    return df


def rename_columns(df, dataset_name):
    """Rename columns to prevent duplicates across datasets."""
    renamer = {}
    for col in list(df):
        if col not in [OPTUM_COST, OPTUM_PAT_ID, OPTUM_SEX, OPTUM_YOB,
                       OPTUM_ENROLL_DATE, OPTUM_END_DATE, OPTUM_ZIPCODE]:
            renamer[col] = f"{dataset_name}:{col}"

    return df.rename(columns=renamer)


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
