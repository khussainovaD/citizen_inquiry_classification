import pandas as pd
import os

def test_sample_data_exists():
    assert os.path.exists("sample_data/sample_inquiries.csv"), "Sample data file does not exist."

def test_sample_data_structure():
    df = pd.read_csv("sample_data/sample_inquiries.csv")
    required_columns = {"text", "category"}
    assert required_columns.issubset(df.columns), f"Missing columns: {required_columns - set(df.columns)}"
    assert not df.empty, "Sample data is empty."
