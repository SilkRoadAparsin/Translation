import pandas as pd


new_path = '/home/sadegh/Translation/data/Hazaragi_translation.csv'

data_path = '/home/sadegh/Translation/datasets/translation_data.csv'

new_data = pd.read_csv(new_path)
# new_data = pd.read_excel(new_path)

data = pd.read_csv(data_path)
data['hazaragi_translation'] = new_data['Hazaragi_translation']
data['hazaragi_transliteration'] = new_data['Hazaragi_transliteration']
data.to_csv(data_path, index=False)
