"""
Module for training a random forest model for predicting income > 50k
"""
import pandas as pd
import os
import json
from joblib import dump
from typing import Dict, Tuple
import yaml
from yaml import CLoader as Loader
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score, recall_score, precision_score, accuracy_score, roc_auc_score
)


def train(params: Dict):
    X, y = load_data(params['preproccess_output_path'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    categorical_features = sorted(
        params['random_forest']['features']['categorical']
    )
    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=''), OrdinalEncoder()
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",  # This drops the columns that we do not transform
    )

    clf = RandomForestClassifier(
        **params['random_forest']['params'],
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )
    pipe.fit(X_train, y_train)

    save_model_scores(pipe, X_test, y_test)
    dump(
        pipe,
        os.path.join(
            'model',
            f"{params['random_forest']['export_name']}.joblib"
        )
    )


def calculate_scores(y: pd.Series, y_pred: pd.Series) -> Tuple[
        float, float, float, float, float]:
    f1 = f1_score(y_pred, y)
    precision = precision_score(y_pred, y)
    recall = recall_score(y_pred, y)
    accuracy = accuracy_score(y_pred, y)
    auc = roc_auc_score(y_pred, y)
    return f1, precision, recall, accuracy, auc


def groupby_scores(row: pd.Series, target: str = 'y',
                   prediction: str = 'prediciton') -> pd.Series:
    result = {}
    f1, precision, recall, accuracy, auc = calculate_scores(
        row[target], row[prediction])
    result['f1'] = f1
    result['precision'] = precision
    result['recall'] = recall
    result['accuracy'] = accuracy
    result['auc'] = auc
    return pd.Series(result)


def save_model_scores(model: Pipeline, X: pd.DataFrame, y: pd.Series):
    y_pred = model.predict(X)
    f1, precision, recall, accuracy, auc = calculate_scores(y, y_pred)

    with open("model/scores.json", "w") as f:
        json.dump({
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "auc": auc,
        }, f)


def load_data(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    y = df.pop('target')
    x = df.copy()
    return x, y


if __name__ == "__main__":

    with open("./params.yaml", "rb") as f:
        params = yaml.load(f, Loader=Loader)
    train(params)
