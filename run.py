import argparse
import csv
import pandas as pd
from pathlib import Path
import pickle
from lightgbm.sklearn import LGBMRegressor
import lightgbm as lgb

from preprocess import *

def preprocess(csv_path):
    df = pd.read_csv(csv_path)

    # Diagnosis
    diag = get_diag_features(df)
    import pdb
    pdb.set_trace()

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
