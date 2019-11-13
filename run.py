import argparse
import csv
import pandas as pd
from pathlib import Path
import pickle
from lightgbm.sklearn import LGBMRegressor
import lightgbm as lgb

from preprocess.utils import load_and_normalize_sdh, OPTUM_ZIP_UNK_KEY 

def preprocess(csv_path, sdh):
    df = pd.read_csv(csv_path)

    # Join and merge with SDH
    if sdh:
        sdh_table = load_and_normalize_sdh()
        med_row = sdh_table.median(axis=0)
        nf = ~df['Zipcode'].isin(sdh_table['Zipcode_5'])
        print(f"Warning: {len(df[nf])} patients have unknown zip codes!")
        df.at[nf, 'Zipcode'] = OPTUM_ZIP_UNK_KEY
        df = pd.merge(sdh_table, df, right_on=['Zipcode'], left_on=['Zipcode_5'], how='right')

    return df


def main(args):
    model_path = Path(args.model_path)
    csv_path = Path(args.csv_path)

    # model = lgb.Booster(model_file=args.model_path)
    user_input = preprocess(csv_path, args.sdh)
    # prediction = model.predict(user_input)
    print(f'Predicted cost: {prediction}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML for Risk Adjustment')
    parser.add_argument('--csv_path', required=True, type=str, help='Path to csv location',)
    parser.add_argument('--model_path', default='model.txt', type=str, help='Path to csv location')
    parser.add_argument('--save_path', default=None, type=str, help='Path to save location')
    parser.add_argument('--sdh', action='store_true', help='Predict with SDH')
    main(parser.parse_args())
