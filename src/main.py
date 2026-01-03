import pandas as pd


new_path = '/home/sadegh/Translation/data/Khorasani_final_dataset_filtered (2).csv'

data_path = '/home/sadegh/Translation/datasets/translation_data.csv'

new_data = pd.read_csv(new_path)
# new_data = pd.read_excel(new_path)

data = pd.read_csv(data_path)
data['khorasani_translation'] = new_data['transliteration']
data['khorasani_transliteration'] = new_data['sent1']
data.to_csv(data_path, index=False)
