import argparse
import csv
import pandas as pd
from pathlib import Path
import pickle
# from lightgbm.sklearn import LGBMRegressor
# import lightgbm as lgb

from preprocess.utils import load_and_normalize_sdh

def preprocess(csv_path, sdh):
    df = pd.read_csv(csv_path)

    # Join and merge with SDH
    if sdh:
        sdh_table = load_and_normalize_sdh()
        sdh_table = sdh_table.rename(columns={"Zipcode_5" : "Zipcode"})
        sdh = sdh_table[sdh_table['Zipcode'].isin(df['Zipcode'])]
        df = pd.merge(sdh, df, on=['Zipcode'])

    return df


def main(args):
    model_path = Path(args.model_path)
    csv_path = Path(args.csv_path)
    
    model = lgb.Booster(model_file=args.model_path)
    user_input = preprocess(csv_path, args.sdh)
    prediction = model.predict(user_input)
    print(f'Predicted cost: {prediction}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML for Risk Adjustment')
    parser.add_argument('--csv_path', required=True, type=str, help='Path to csv location',)
    parser.add_argument('--model_path', default='model.txt', type=str, help='Path to csv location')
    parser.add_argument('--save_path', default=None, type=str, help='Path to save location')
    parser.add_argument('--sdh', action='store_true', help='Predict with SDH')
    main(parser.parse_args())
