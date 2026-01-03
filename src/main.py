import pandas as pd


new_path = '/home/sadegh/Translation/datasets/translation_data_simplified.csv'

data_path = '/home/sadegh/Translation/datasets/translation_data.csv'

new_data = pd.read_csv(new_path)

data = pd.read_csv(data_path)
data['dari_translation'] = new_data['dari_translation']
# data['zoroastrian_yazdi_transliteration'] = new_data['transliteration']
data.to_csv(data_path, index=False)
