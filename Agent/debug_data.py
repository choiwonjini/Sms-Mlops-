import pandas as pd
import sys

try:
    df = pd.read_csv('test_data/validation_data_temp.CSV', encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv('test_data/validation_data_temp.CSV', encoding='cp949')

print("=== Case 1 Guide ===")
print(df.iloc[0]['guide'])
print("====================")
print(df.iloc[0]['order'])
print("=== Case 3 Label ===")
print(df.iloc[2]['label'])
print("====================")
