"""
Module for cleaning census csv and output a processed_data.csv
"""
import argparse
import pandas as pd


def clean(args):
    df = pd.read_csv(args.input_data, sep=r",\s+")
    clean_df = transform(df)
    clean_df.to_csv(args.output_path, index=False)


def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    salary = df.pop('salary')
    df = df.assign(
        target=(salary == '>50K').astype(int)
    )
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Preprocess a dataset",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_data",
        type=str,
        help="path to the input data to be cleaned",
        required=True,
    )

    parser.add_argument(
        "--output_path",
        type=str,
        help="putput for the cleaned data",
        required=True
    )

    args = parser.parse_args()
    clean(args)
