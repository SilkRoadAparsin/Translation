import pandas as pd


new_path = '/home/sadegh/Translation/data/pashto.xlsx'

data_path = '/home/sadegh/Translation/datasets/translation_data.csv'

# new_data = pd.read_csv(new_path)
new_data = pd.read_excel(new_path)

data = pd.read_csv(data_path)
data['pashto_translation'] = new_data['Pashto Translation']
# data['pashto_transliteration'] = new_data['transliteration']
data.to_csv(data_path, index=False)
