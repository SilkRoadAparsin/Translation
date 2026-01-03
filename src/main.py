import pandas as pd


new_path = '/home/sadegh/Translation/datasets/esfahani_with_sentiments.xlsx'

data_path = '/home/sadegh/Translation/datasets/translation_data.csv'

# new_data = pd.read_csv(new_path)
new_data = pd.read_excel(new_path)

data = pd.read_csv(data_path)
data['isfahani_translation'] = new_data['isfahani']
data['isfahani_transliteration'] = new_data['transliteration']
data.to_csv(data_path, index=False)
