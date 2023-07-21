"""
Module for cleaning census csv and output a processed_data.csv
"""
import pandas as pd
import yaml
from yaml import CLoader as Loader


def clean(input_data, output_path):
    df = pd.read_csv(input_data, sep=r",\s+")
    clean_df = transform(df)
    clean_df.to_csv(output_path, index=False)


def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    salary = df.pop('salary')
    df = df.assign(
        target=(salary == '>50K').astype(int)
    )
    return df


if __name__ == '__main__':

    with open("./params.yaml", "rb") as f:
        params = yaml.load(f, Loader=Loader)
    clean(
        params['input_data'],
        params['preproccess_output_path']
    )
