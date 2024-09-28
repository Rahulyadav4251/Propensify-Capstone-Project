
import argparse
import os
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import xgboost as xgb
from sklearn.pipeline import Pipeline

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

model_file_name = "pipeline_model.joblib"

# Main function
def main():
    logger.info("Starting training")
    parser = argparse.ArgumentParser()
    
    # Inbuilt Arguments: https://github.com/aws/sagemaker-containers#id11
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))
    
    args, _ = parser.parse_known_args()
    
    logger.info("Load data")
    train_df = pd.read_csv(os.path.join(args.train, "train.csv"))
    test_df = pd.read_csv(os.path.join(args.validation, "validation.csv"))

    # Define the columns
    cat_cols = ['loan', 'marital', 'schooling', 'default', 'housing', 'day_of_week','poutcome', 'pdays_bin', 'pmonths_bin', 'profession', 'month', 'contact']
    cont_cols = ['custAge', 'campaign', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'pastEmail']

    # Split X(features) and y(response)
    X_train = train_df.drop('responded', axis=1)
    y_train = train_df["responded"]

    X_test = test_df.drop("responded", axis=1)
    y_test = test_df["responded"]

    # One hot encode the categorical columns
    ohe = OneHotEncoder(drop="first")

    # Scale the continuous columns
    sc = StandardScaler()

    # Column transformer to apply transformations on both categorical and continuous columns
    ct = ColumnTransformer([
        ("One Hot Encoding", ohe, cat_cols),
        ("Scaling", sc, cont_cols)
    ])
    
    logger.info("Train the model pipeline")
    xgb_clf = xgb.XGBClassifier(random_state=42)

    # Sklearn pipeline
    pipeline_xgb_clf_model = Pipeline([
        ("Data Transformations", ct),
        ("Random Forest Model", xgb_clf)
    ])

    # Fit the model locally on a smaller subset of data
    pipeline_xgb_clf_model.fit(X_train, y_train)

    # Check the accuracy on training data
    train_accuracy = pipeline_xgb_clf_model.score(X_train, y_train)
    print(f"Training Accuracy: {train_accuracy:.4f}")

    # Check the accuracy on test data
    test_accuracy = pipeline_xgb_clf_model.score(X_test, y_test)
    print(f"Testing Accuracy: {test_accuracy:.4f}")

    logger.info("Save the model")
    model_save_path = os.path.join(args.model_dir, model_file_name)
    joblib.dump(pipeline_xgb_clf_model, model_save_path)
    print(f"Model saved at {model_save_path}")

# Run the main function when the script runs
if __name__ == "__main__":
    main()
