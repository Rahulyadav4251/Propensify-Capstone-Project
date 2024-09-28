"""Create the training and testing dataset."""
import argparse
import logging
import pathlib
import pandas as pd
import numpy as np
import requests
import zipfile
import io
from sklearn.model_selection import train_test_split

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def main():
    logger.info("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True, help="URL of input data")
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)

    logger.info("Downloading data")
    response = requests.get(args.input_data)
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(f"{base_dir}/data")

     # Reading Excel files
    logger.info("Reading data")
    train_data_path = f"{base_dir}/data/train.xlsx"
    test_data_path = f"{base_dir}/data/test.xlsx"
    

    propensify_df = pd.read_excel(train_data_path)
    test_df = pd.read_excel(test_data_path)

    # preprocessing
    logger.info("Handling null value")
    
    ## Keeping only those columns that are required
    columns_to_keep = ['custAge', 'profession', 'marital', 'schooling', 'default', 'housing',
                       'loan', 'contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous',
                       'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
                       'euribor3m', 'nr.employed', 'pmonths', 'pastEmail', 'responded']
    
    propensify_df = propensify_df[columns_to_keep]

    ## Feature engineering for schooling
    schooling_category = {
        'basic.4y' : 'basic',
        'basic.6y' : 'basic',
        'basic.9y' : 'basic',
        'high.school': 'high.school',
        'illiterate':'illiterate',
        'professional.course': 'professional.course',
        'university.degree':'university.degree',
        'unknown':'unknown',
    }

    propensify_df.loc[:,'schooling'] = propensify_df['schooling'].replace(schooling_category)

    ## Imputation of missing values in education based on profession
    imputation_mapping = {
        'blue-collar' : 'basic',
        'self-employed': 'illiterate',
        'technician'   : 'professional.course',
        'admin.'        : 'university.degree',
        'services'      : 'high.school',
        'management'    : 'university.degree',
        'retired'       : 'unknown',
        'entrepreneur'  : 'university.degree'
    }

    propensify_df.loc[:,'schooling'] = propensify_df['schooling'].combine_first(propensify_df['profession'].map(imputation_mapping))

    ## Imputing age values
    ## Calculate median age for each profession
    median_age = propensify_df.groupby('profession')['custAge'].median().rename('mean_age').reset_index()

    ## Create a mapping from profession to mean age
    median_age_dict = median_age.set_index('profession')['mean_age'].to_dict()

    ## Fill missing age values based on profession
    propensify_df['custAge']  = propensify_df['custAge'].fillna(propensify_df['profession'].map(median_age_dict))

    ## Impute random day for missing 'day_of_week' values
    propensify_df.loc[:,'day_of_week'] = propensify_df['day_of_week'].apply(lambda day: np.random.choice(['mon', 'tue', 'wed', 'thu', 'fri']) if pd.isna(day) else day)

    ## Drop remaining missing values
    propensify_df = propensify_df.dropna()

    # Encoding
    logger.info('Encoding variables')

    ## Label encoding for 'profession'
    propensify_df.loc[:,'profession'] = propensify_df['profession'].map({'student': 'Dependents', 'retired': 'Dependents', 'unemployed': 'Unemployed&Unknown', 'unknown': 'Unemployed&Unknown',
                                                 'admin.': 'Working', 'blue-collar': 'Working', 'entrepreneur': 'Working', 'housemaid': 'Working',
                                                 'management': 'Working', 'self-employed': 'Working', 'services': 'Working', 'technician': 'Working'})

    ## Label encoding for 'marital'
    propensify_df.loc[:,'marital'] = propensify_df['marital'].map({'single': 'Single&Divorced', 'divorced': 'Single&Divorced', 'married': 'married', 'unknown': 'Unknown'})

    ## Label encoding for 'schooling'
    propensify_df.loc[:,'schooling'] = propensify_df['schooling'].map({'basic': 'Uneducated&BasicEducation', 'high.school': 'Uneducated&BasicEducation',
                                               'illiterate': 'Uneducated&BasicEducation', 'unknown': 'Unknown',
                                               'professional.course': 'Educated', 'university.degree': 'Educated'})

    ## Transforming month to quarter in a new column
    ## Create a dictionary mapping  month names to quarters
    month_to_quarter = {
        'jan': 'Q1', 'feb': 'Q1', 'mar': 'Q1',
        'apr': 'Q2', 'may': 'Q2', 'jun': 'Q2',
        'jul': 'Q3', 'aug': 'Q3', 'sep': 'Q3',
        'oct': 'Q4', 'nov': 'Q4', 'dec': 'Q4'
    }

    ## Map the  month names to quarters
    propensify_df['quarter'] = propensify_df['month'].map(month_to_quarter)

    ## Dropping month column
    propensify_df = propensify_df.drop(columns='month', axis=1)

    ## Label encoding for 'day_of_week'
    propensify_df.loc[:,'day_of_week'] = propensify_df['day_of_week'].map({'mon': 'WeekBeginning', 'tue': 'WeekBeginning', 'wed': 'WeekBeginning',
                                                   'thu': 'WeekEnding', 'fri': 'WeekEnding'})
    
    ## Label encoding for 'default'
    propensify_df.loc[:,'default'] = propensify_df['default'].map({'no': 'No', 'unknown': 'Yes&Unknown', 'yes': 'Yes&Unknown'})

    ## Feature engineering of other variables
    ## pdays
    conditions = [
        (propensify_df['pdays'] == 999),
        (propensify_df['pdays'] < 5),
        ((propensify_df['pdays'] >= 5) & (propensify_df['pdays'] <= 10)),
        (propensify_df['pdays'] > 10)
    ]

    choices = ['first visit', 'less than 5 days', '5 to 10 days', 'more than 10 days']

    ## Create the 'pdays_bin' column based on conditions
    propensify_df.loc[:,'pdays_bin'] = np.select(conditions, choices, default='unknown')

    ## pmonths
    conditions = [
        (propensify_df['pmonths'] == 999),
        (propensify_df['pmonths'] <= 0.2),
        (propensify_df['pmonths'] > 0.2)
    ]

    choices = ['first visit', 'less than 2 months', 'more than 2 months']

    ## Create the 'pmonths_bin' column based on conditions
    propensify_df['pmonths_bin'] = np.select(conditions, choices, default='unknown')

    ## drop pday and pmonth
    propensify_df = propensify_df.drop(['pdays', 'pmonths'], axis=1)

    # Train Test split 
    logger.info("Split out training and validation datasets")
    train_df, val_df = train_test_split(propensify_df, test_size=0.2, random_state=42)
     
    # save output as csv files
    logger.info(f"Writing out datasets to {base_dir}")
    train_df.to_csv(f"{base_dir}/train/train.csv", index=False)
    val_df.to_csv(f"{base_dir}/validation/validation.csv", index=False)
    test_df.to_csv(f"{base_dir}/test/test.csv", index=False)

if __name__ == "__main__":
    main()