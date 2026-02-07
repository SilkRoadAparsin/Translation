import os
from openai import OpenAI
from tqdm import tqdm

import config


COLUMN_LANGUAGE_META = {
    "standard_persian_translation": {
        "language": "Persian",
        "dialect": "Persian, Iranian (Iran)",
        "accent": "General",
    },
    "dezfuli_translation": {
        "language": "Unclassified",
        "dialect": "Dezfuli (Iran)",
        "accent": "Dezfuli",
    },
    "lori_translation": {
        "language": "Luri",
        "dialect": "Luri, Southern (Iran)",
        "accent": "General",
    },
    "tonekaboni_translation": {
        "language": "Caspian",
        "dialect": "Mazandarani (Iran)",
        "accent": "Tonekaboni",
    },
    "semnani_translation": {
        "language": "Semnani",
        "dialect": "Semnani (Iran)",
        "accent": "Semnani",
    },
    "southern_kurdish_kalhori_translation": {
        "language": "Kurdish",
        "dialect": "Kurdish, Southern (Iran)",
        "accent": "Kurdish, Southern",
    },
    "zoroastrian_yazdi_transliteration": {
        "language": "Central Iran",
        "dialect": "Dari, Zoroastrian (Iran)",
        "accent": "Yazdi",
    },
    "dari_translation": {
        "language": "Persian",
        "dialect": "Dari (Afghanistan)",
        "accent": "General",
    },
    "isfahani_translation": {
        "language": "Persian",
        "dialect": "Persian, Iranian (Iran)",
        "accent": "Isfahani",
    },
    "shirazi_translation": {
        "language": "Persian",
        "dialect": "Persian, Iranian (Iran)",
        "accent": "Shirazi",
    },
    "hazaragi_translation": {
        "language": "Persian",
        "dialect": "Hazaragi (Afghanistan)",
        "accent": "Hazaragi",
    },
    "yazdi_translation": {
        "language": "Persian",
        "dialect": "Persian, Iranian (Iran)",
        "accent": "Yazdi",
    },
    "pashto_translation": {
        "language": "Pashto",
        "dialect": "Pashto, Central (Pakistan)",
        "accent": "General",
    },
}


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=config.OPENROUTER_API_KEY,
)

def translate_text(text: str, language: str, dialect: str, accent: str, target_language: str,  model: str) -> str:
    """
    Translate `text` into `target_language`.
    e.g., target_language = "French", "German", "Dutch", etc.
    """
    prompt = f"""
    Translate the following text with 
    language: {language}, 
    dialect: {dialect}, 
    accent: {accent} 
    
    into {target_language}\n\n\"\"\"\n{text}\n\"\"\"\n
    """
    response = client.responses.create(
        model=model,
        input=[
            {
            "role": "system",
            "content": [
                {
                "type": "input_text",
                "text": "You are a professional translator with expertise in Iranic languages. Just return the translation."
                }
            ]
            },
            {
            "role": "user",
            "content": [
                {
                "type": "input_text",
                "text": prompt
                }
            ]
            }
        ],
        temperature=1,
        max_output_tokens=256,

    )
    try:
        translation = response.output[0].content[0].text.strip()
    except:
        return translate_text(text, language, dialect, accent, target_language,  model)
    return translation


import os
import pandas as pd

data_path = 'datasets/translation_data.csv'
output_path = 'results/translation/data.csv'

df = pd.read_csv(data_path)

# ----------------------------
# 1. Load or create result_df
# ----------------------------
if os.path.exists(output_path):
    result_df = pd.read_csv(output_path)
else:
    result_df = pd.DataFrame({
        'model': [],
        'language': [],
        'dialect': [],
        'accent': [],
        'text': [],
        'english_translation': [],
        'pred_english_translation': [],
    })

KEY_COLS = ['model', 'language', 'dialect', 'accent', 'text', 'english_translation']

# Convert existing rows to a set for fast lookup
existing_keys = set(
    tuple(row[col] for col in KEY_COLS)
    for _, row in result_df.iterrows()
)

# ----------------------------
# 2. Translation loop
# ----------------------------

for model in config.MODELS_8:
    for column, language_info in COLUMN_LANGUAGE_META.items():
        language = language_info['language']
        dialect = language_info['dialect']
        accent = language_info['accent']

        for _, row in tqdm(df.iterrows()):
            text = row[column]
            english_translation = row['english_translation']

            key = (model, language, dialect, accent, text, english_translation)

            # Skip redundant data
            if key in existing_keys:
                continue

            # Translate only if new
            pred_translation = translate_text(
                text,
                language,
                dialect,
                accent,
                'English',
                model
            )
            new_rows = []
            new_rows.append({
                'model': model,
                'language': language,
                'dialect': dialect,
                'accent': accent,
                'text': text,
                'english_translation': english_translation,
                'pred_english_translation': pred_translation,
            })

            existing_keys.add(key)

            # ----------------------------
            # 3. Save results
            # ----------------------------
            if new_rows:
                result_df = pd.concat([result_df, pd.DataFrame(new_rows)], ignore_index=True)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result_df.to_csv(output_path, index=False)
