import argparse
import csv
import pandas as pd
import scipy as sp
from pathlib import Path
import pickle
from lightgbm.sklearn import LGBMRegressor
import lightgbm as lgb

from preprocess import *

def preprocess(csv_path):
    df = pd.read_csv(csv_path)

    # Diagnosis
    # explode out ICD10
    icd = df['ICD10'].str.split(",")
    icd = icd.apply(pd.Series)
    icd['patid'] = icd.index
    icd = icd.melt(id_vars='patid').drop(['variable'], axis=1)
    icd = icd.rename({'value':"ICD10CM"}, axis=1)
    icd = icd[~icd['ICD10CM'].isna()]
    icd['ICD10CM'] = icd['ICD10CM'].str.replace(".", "")

    # convert to CCS
    icd2ccs, diag_names = load_icd2ccs(Path('preprocess/icd10cm_to_ccs.csv'))
    icd10codes = icd2ccs.index
    icd['ICD10CM'].where(icd['ICD10CM'].isin(icd10codes), "OPTUM_DIAGMAP_KEY_ERROR_UNK", inplace=True)
    icd['CCS'] = icd2ccs.loc[icd['ICD10CM']].reset_index(drop=True).values.flatten()

    # aggregate within patients
    icd = icd.groupby('patid').sum()
    diag = sp.vstack(icd['CCS'])

    return df


def main(args):
    model_path = Path(args.model_path)
    csv_path = Path(args.csv_path)

    model = lgb.Booster(model_file=args.model_path)
    import pdb
    pdb.set_trace()

    user_input = preprocess(csv_path)

    prediction = model.predict(user_input)
    print(f'Predicted cost: {prediction}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML for Risk Adjustment')
    parser.add_argument('--csv_path', required=True, type=str, help='Path to csv location',)
    parser.add_argument('--model_path', default='model.txt', type=str, help='Path to csv location')
    parser.add_argument('--save_path', default=None, type=str, help='Path to save location')
    main(parser.parse_args())
