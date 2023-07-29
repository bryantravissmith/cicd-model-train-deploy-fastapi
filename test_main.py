from fastapi.testclient import TestClient

# Import our app from main.py.
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)


# Write tests using the same syntax as with the requests module.
def test_api_locally_get_root():
    r = client.get("/")
    assert r.status_code == 200


def test_valid_prediciton():

    data = {
        'workclass': 'State-gov',
        'education': 'Bachelors',
        'marital_status': 'Never-married',
        'occupation': 'Adm-clerical',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Male',
        'native_country': 'United-States'
    }
    r = client.post(
        "/",
        json=data,
    )
    assert r.status_code == 200
    response = r.json()
    response_keys = list(response.keys())
    assert len(response_keys) == 1
    assert response_keys[0] == 'income_above_50k'


def test_invalid_prediciton():

    data = {
        'workclass': 'State-gov',
        'education': 'Bachelors',
        'marital_status': 'Never-married',
        'occupation': 'Adm-clerical',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Male',
        'native_country': 'Fake Country'
    }
    r = client.post(
        "/",
        json=data,
    )
    assert r.status_code == 422
