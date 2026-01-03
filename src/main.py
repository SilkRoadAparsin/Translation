import pandas as pd


new_path = '/home/sadegh/Translation/data/shirazi_annotations/100_samples_shirazi_sentiment_ann1.xlsx'

data_path = '/home/sadegh/Translation/datasets/translation_data.csv'

# new_data = pd.read_csv(new_path)
new_data = pd.read_excel(new_path)

data = pd.read_csv(data_path)
data['shirazi_translation'] = new_data['shirazi_translation']
data['shirazi_transliteration'] = new_data['shirazi_transliteration']
data.to_csv(data_path, index=False)
