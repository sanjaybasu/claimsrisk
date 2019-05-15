## Machine Learning for Risk Adjustment

Based on the paper:

  > Do machine learning and social determinants of health indicators improve healthcare risk adjustment?
  >
  > Jeremy A. Irvin, Andrew A. Kondrich, Michael Ko, Pranav Rajpurkar, Behzad Haghgoo, Bruce E. Landon, Robert Phillips, Stephen Petterson, Andrew Y. Ng, Sanjay Basu 

Includes a script `process.py` to transform CSV files containing demographic and diagnoses information into processed inputs for learning models, and `model.py` which trains a lightgbm model using 3-fold CV on such formatted data.
## Usage

### Environment Setup
    1. Please have [Anaconda or Miniconda](https://conda.io/docs/download.html) installed to create a Python virtual environment.
    2. Clone repo with `https://github.com/stanfordmlgroup/risk-adjustment-ml`.
    3. Go into the cloned repo: `cd risk-adjustment-ml`.
    4. Create the environment: `conda env create -f environment.yml`.
    5. Activate the environment: `source activate ra-ml`.

### Processing
> Demographics: CSV with Patient ID, Age, Sex, [Optional] ZIP (age is an integer, sex is M/F, ZIP is 5-digit)  Diagnoses: CSV with Patient ID, ICD-10 Diagnosis. One row per diagnosis.

### Modeling

> X: covariates that can be input to the ML model. Includes SDH if ZIP is provided, and imputes in the same way as we do if ZIP is not found.