import sys
#sys.path.append("../src")  # adjust if needed depending on where you run tests from

import pandas as pd
from fraud_triage.data.split import time_based_split


def test_no_time_overlap():
    # A tiny fake dataset, just to test the logic — not the real 590K rows
    df = pd.DataFrame({
        "day": [1, 50, 100, 128, 130, 132, 150, 180],
        "isFraud": [0, 1, 0, 1, 0, 1, 0, 1],
    })

    train_df, val_df = time_based_split(df, train_cutoff=128, val_start=132)

    # The core check: no validation row's day is <= any train row's max day
    assert val_df["day"].min() > train_df["day"].max()