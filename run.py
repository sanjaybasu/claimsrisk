import argparse
import csv
from pathlib import Path
from lightgbm.sklearn import LGBMRegressor

def preprocess(csv_path):
	raise NotImplementedError()

def main(args):
	model_path = Path(args.model_path)
	csv_path = Path(args.csv_path)

	with model_path.open("rb") as f:
        model = pickle.load(f)

    user_input = preprocess(csv_path)

    prediction = model.predict(user_input)
    print(f'Predicted cost: {prediction}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RealNVP on CIFAR-10')
    parser.add_argument('--csv_path', required=True, type=str, help='Path to csv location',)
    parser.add_argument('--model_path', default='./checkpoints/sdh_model.pkl', type=str, help='Path to csv location')
    parser.add_argument('--save_path', default=None, type=str, help='Path to save location')
    main(parser.parse_args())
