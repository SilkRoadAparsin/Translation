import pandas as pd
from tqdm import tqdm

from utils import translate_text

dezfuli_path = '/home/sadegh/Translation/data/dezfuli.xlsx'
lori_path = '/home/sadegh/Translation/data/lori.xlsx'
mazandaran_tonekaboni_path = '/home/sadegh/Translation/data/Mazandaran_Tonekaboni.xlsx'
semnani_path = '/home/sadegh/Translation/data/Semnani.xlsx'
southern_kurdish_kalhori_path = '/home/sadegh/Translation/data/Southern_Kurdish_Kalhori.xlsx'
zoroastrian_yazdi_path = '/home/sadegh/Translation/data/Zoroastrian_Yazdi.xlsx'

data_path = '/home/sadegh/Translation/datasets/data.csv'

dezfuli_df = pd.read_excel(dezfuli_path)
lori_df = pd.read_excel(lori_path)
mazandaran_tonekaboni_df = pd.read_excel(mazandaran_tonekaboni_path)
semnani_df = pd.read_excel(semnani_path)
southern_kurdish_kalhori_df = pd.read_excel(southern_kurdish_kalhori_path)
zoroastrian_yazdi_df = pd.read_excel(zoroastrian_yazdi_path)

data = pd.read_csv(data_path)
data['zoroastrian_yazdi_translation'] = zoroastrian_yazdi_df['translation']
data['zoroastrian_yazdi_transliteration'] = zoroastrian_yazdi_df['transliteration']
data.to_csv(data_path, index=False)
