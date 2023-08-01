import requests

URL_ENDPOINT = "https://continuous-deployment-example.onrender.com/"

DATA = {
    'workclass': 'State-gov',
    'education': 'Bachelors',
    'marital_status': 'Never-married',
    'occupation': 'Adm-clerical',
    'relationship': 'Not-in-family',
    'race': 'White',
    'sex': 'Male',
    'native_country': 'United-States'
}


def make_request():
    response = requests.post(URL_ENDPOINT, json=DATA)
    print(response.json())


if __name__ == "__main__":
    make_request()
