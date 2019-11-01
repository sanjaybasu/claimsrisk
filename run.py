import argparse
import csv
import pandas as pd
from pathlib import Path
from lightgbm.sklearn import LGBMRegressor

def preprocess(csv_path):
    df = pd.read_csv(csv_path)

    return df


def main(args):
    model_path = Path(args.model_path)
    csv_path = Path(args.csv_path)

    with model_path.open("rb") as f:
        model = pickle.load(f)

    user_input = preprocess(csv_path)

    prediction = model.predict(user_input)
    print(f'Predicted cost: {prediction}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML for Risk Adjustment')
    parser.add_argument('--csv_path', required=True, type=str, help='Path to csv location',)
    parser.add_argument('--model_path', default='./checkpoints/sdh_model.pkl', type=str, help='Path to csv location')
    parser.add_argument('--save_path', default=None, type=str, help='Path to save location')
    main(parser.parse_args())
