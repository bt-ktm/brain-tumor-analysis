import zipfile
import os
import pandas as pd

zip_path = "data/raw/archive.zip"
extract_to = "data/raw/unzipped"

os.makedirs(extract_to, exist_ok=True)


with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)
    
csv_path = os.path.join(extract_to, "Brain Tumor.csv")
df = pd.read_csv(csv_path)

print(df.head())
print(df.shape)
print("Duplicate rows:", df.duplicated().sum())
print(df.dtypes)