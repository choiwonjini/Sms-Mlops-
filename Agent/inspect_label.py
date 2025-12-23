import pandas as pd
try:
    df = pd.read_csv('test_data/validation_data_temp.CSV', encoding='utf-8')
except:
    df = pd.read_csv('test_data/validation_data_temp.CSV', encoding='cp949')

label = df.iloc[2]['label']
with open('debug_label_case3.txt', 'w', encoding='utf-8') as f:
    f.write(str(label))
print("Label saved to debug_label_case3.txt")
