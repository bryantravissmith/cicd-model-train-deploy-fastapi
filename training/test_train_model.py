from joblib import load
import numpy as np
import os
import pandas as pd
import pytest
from train_model import calculate_scores, load_data


@pytest.fixture
def model():
    model = load(os.path.join('model', 'random_forest_census_income.joblib'))
    yield model


@pytest.fixture
def data():
    x, y = load_data(os.path.join('data', 'preprocess_sample.csv'))
    return x, y


@pytest.fixture
def positive_predictions():
    return pd.DataFrame.from_dict({
        'workclass': ['Private', 'Private', 'State-gov', 'Self-emp-not-inc',
                      'Private', 'Private', 'Private', 'Self-emp-inc',
                      'Private'],
        'education': ['Bachelors', 'Masters', 'Doctorate', 'Bachelors',
                      'Bachelors', 'Bachelors', 'HS-grad', 'Bachelors',
                      'Some-college'],
        'marital-status': ['Married-civ-spouse', 'Married-civ-spouse',
                           'Divorced', 'Married-civ-spouse',
                           'Married-civ-spouse', 'Married-civ-spouse',
                           'Married-civ-spouse', 'Married-civ-spouse',
                           'Married-civ-spouse'],
        'occupation': ['Tech-support', 'Exec-managerial', 'Prof-specialty',
                       'Prof-specialty', 'Prof-specialty', 'Exec-managerial',
                       'Exec-managerial', 'Farming-fishing', 'Prof-specialty'],
        'relationship': ['Husband', 'Husband', 'Unmarried', 'Husband',
                         'Husband', 'Husband', 'Husband', 'Husband',
                         'Husband'],
        'race': ['White', 'White', 'White', 'White', 'White', 'White', 'White',
                 'White', 'White'],
        'sex': ['Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male',
                'Male'],
        'native-country': ['United-States', 'United-States', 'United-States',
                           'United-States', 'United-States', 'United-States',
                           'United-States', 'United-States', 'United-States'],
    })


@pytest.fixture
def negative_predictions():
    return pd.DataFrame.from_dict({
        'workclass': ['Private', 'Self-emp-not-inc', 'Private', 'State-gov',
                      'Private', '?', 'Private', 'Private', 'State-gov',
                      'Private'],
        'education': ['Bachelors', 'Bachelors', 'HS-grad', 'Some-college',
                      'HS-grad', 'HS-grad', 'HS-grad', 'Bachelors',
                      'Assoc-voc', '11th'],
        'marital-status': ['Never-married', 'Never-married',
                           'Married-civ-spouse', 'Never-married',
                           'Married-civ-spouse', 'Married-civ-spouse',
                           'Never-married', 'Never-married',
                           'Married-civ-spouse', 'Never-married'],
        'occupation': ['Prof-specialty', 'Farming-fishing', 'Adm-clerical',
                       'Craft-repair', 'Machine-op-inspct', '?',
                       'Handlers-cleaners', 'Other-service', 'Exec-managerial',
                       'Handlers-cleaners'],
        'relationship': ['Not-in-family', 'Unmarried', 'Husband',
                         'Not-in-family', 'Husband', 'Wife', 'Other-relative',
                         'Not-in-family', 'Husband', 'Own-child'],
        'race': ['White', 'White', 'White', 'White', 'White', 'White',
                 'White', 'White', 'White', 'White'],
        'sex': ['Female', 'Male', 'Male', 'Male', 'Male', 'Female', 'Male',
                'Male', 'Male', 'Male'],
        'native-country': ['Mexico', 'United-States', 'United-States',
                           'United-States', 'United-States', 'United-States',
                           'Mexico', 'Mexico', 'United-States',
                           'United-States'],
    })


def test_load_data(data):
    assert isinstance(data[0], pd.DataFrame)
    assert isinstance(data[1], pd.Series)


def test_model_score(model, data):
    y_pred = model.predict(data[0])
    predicted_values = sorted(list(np.unique(y_pred).astype(int)))
    assert predicted_values == [0, 1]


def test_calculate_score():
    f1, precision, recall, accuracy = calculate_scores(
        pd.Series([0, 0, 1, 1]),
        pd.Series([0, 1, 0, 1]),
    )
    assert np.abs(accuracy - 0.5) < 1e-5
    assert np.abs(precision - 0.5) < 1e-5
    assert np.abs(recall - 0.5) < 1e-5
    assert np.abs(f1 - 0.5) < 1e-5


def test_model_predictions(model, positive_predictions, negative_predictions):
    y_pos = model.predict(positive_predictions)
    y_neg = model.predict(negative_predictions)

    y_pos_values = list(np.unique(y_pos).astype(int))
    y_neg_values = list(np.unique(y_neg).astype(int))

    assert y_pos_values == [1]
    assert y_neg_values == [0]
