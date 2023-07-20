import pandas as pd
import numpy as np
from preprocess import transform


def test_transform():
    df = pd.DataFrame.from_dict({
        'salary': ['>50K', '<=50K', '<=50K', '>50K', '>50K'],
        'feature': ['x', 'y', np.nan, 'x', np.nan]
    })
    new_df = transform(df)
    assert 'salary' not in set(new_df.columns)
    assert 'target' in set(new_df.columns)
    assert new_df.shape == (3, 2)
    assert new_df.target.sum() == 2
    assert (new_df.target == 0).astype(int).sum() == 1
