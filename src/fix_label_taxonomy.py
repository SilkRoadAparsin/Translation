EXTRA_LANGUAGES = {
    "Avestan",
    # Eastern Iranian - Northeastern
    "Ossetic", "Yagnobi", "Yassic",

    # Eastern Iranian - Southeastern
    "Pamir", "Pashto",

    # Western Iranian - Northwestern
    "Balochi", "Caspian", "Central Iran", "Kurdish", "Ormuri-Parachi",
    "Semnani", "Talysh", "Zaza-Gorani", "Unclassified",

    # Western Iranian - Southwestern
    "Fars", "Luri", "Persian", "Tat",
}

EXTRA_DIALECTS_BY_LANGUAGE = {
    "Avestan": {"General"},
    "Ossetic": {"General"},
    "Yagnobi": {"General"},
    "Yassic": {"General"},

    "Pamir": {"Ishkashimi (Afghanistan)", "Munji (Afghanistan)", "Sanglechi (Afghanistan)", "Wakhi (Afghanistan)", "Yidgha (Pakistan)"},
    "Pashto": {"Pashto, Central (Pakistan)", "Pashto, Northern (Pakistan)", "Pashto, Southern (Afghanistan)", "Waneci (Pakistan)"},

    "Balochi": {
        "Balochi, Eastern (Pakistan)",
        "Balochi, Southern (Pakistan)",
        "Balochi, Western (Pakistan)",
        "Bashkardi (Iran)",
        "Koroshi (Iran)",
    },

    "Caspian": {"Gilaki (Iran)", "Mazandarani (Iran)", "Shahmirzadi (Iran)"},

    "Central Iran": {
        "Ashtiani (Iran)", "Dari, Zoroastrian (Iran)", "Gazi (Iran)", "Khunsari (Iran)",
        "Natanzi (Iran)", "Nayini (Iran)", "Parsi-Dari (Iran)", "Sivandi (Iran)",
        "Soi (Iran)", "Vafsi (Iran)",
    },

    "Kurdish": {
        "Kurdish, Central (Iraq)",
        "Kurdish, Northern (Turkey)",
        "Kurdish, Southern (Iran)",
        "Laki (Iran)",
    },

    "Ormuri-Parachi": {"Ormuri (Pakistan)", "Parachi (Afghanistan)"},

    "Semnani": {"Lasgerdi (Iran)", "Sangisari (Iran)", "Semnani (Iran)", "Sorkhei (Iran)"},

    "Talysh": {
        "Alviri-Vidari (Iran)", "Eshtehardi (Iran)", "Gozarkhani (Iran)", "Harzani (Iran)",
        "Kabatei (Iran)", "Kajali (Iran)", "Karingani (Iran)", "Kho'ini (Iran)",
        "Koresh-e Rostam (Iran)", "Maraghei (Iran)", "Razajerdi (Iran)", "Rudbari (Iran)",
        "Shahrudi (Iran)", "Takestani (Iran)", "Talysh (Azerbaijan)", "Taromi, Upper (Iran)",
    },

    "Zaza-Gorani": {
        "Bajelani (Iraq)", "Gurani (Iran)", "Kakayi (Iraq)", "Shabak (Iraq)",
        "Zazaki, Northern (Turkey)", "Zazaki, Southern (Turkey)",
    },

    "Unclassified": {"Dezfuli (Iran)"},

    "Fars": {"Fars, Southwestern (Iran)", "Lari (Iran)"},

    "Luri": {"Bakhtiari (Iran)", "Kumzari (Oman)", "Luri, Northern (Iran)", "Luri, Southern (Iran)"},

    "Persian": {
        "Aimaq (Afghanistan)", "Bukharic (Uzbekistan)", "Dari (Afghanistan)", "Dehwari (Pakistan)",
        "Hazaragi (Afghanistan)", "Judeo-Persian (Iran)", "Pahlavani (Afghanistan)",
        "Persian, Iranian (Iran)", "Tajik (Tajikistan)",
    },

    "Tat": {"Judeo-Tat (Russian Federation)", "Tat, Muslim (Azerbaijan)"},
}

EXTRA_ACCENTS_BY_LANGUAGE_DIALECT = {
    ("Persian", "Persian, Iranian (Iran)"): {"Isfahani", "Shirazi", "Mashhadi", "Yazdi", "Khorasani", "General"},
    ("Persian", "Dari (Afghanistan)"): {"Kaboli", "General"},
    ("Persian", "Hazaragi (Afghanistan)"): {"Hazaragi", "General"},
    ("Kurdish", "Kurdish, Southern (Iran)"): {"Kurdish, Southern", "General"},
    ("Caspian", "Mazandarani (Iran)"): {"Tonekaboni", "General"},
    ("Semnani", "Semnani (Iran)"): {"Semnani", "General"},
    ("Central Iran", "Dari, Zoroastrian (Iran)"): {"Yazdi", "General"},
    ("Unclassified", "Dezfuli (Iran)"): {"Dezfuli", "General"},
    ("Pashto", "Pashto, Southern (Afghanistan)"): {"Isfahani", "Shirazi", "Mashhadi", "Yazdi", "Khorasani", "General"},
    ("Pashto", "Pashto, Central (Pakistan)"): {"Isfahani", "Shirazi", "Mashhadi", "Yazdi", "Khorasani", "General"},
    ("Luri", "Bakhtiari (Iran)"): {"Chaharmahali", "General"},
}

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


data_path = 'datasets/translation_data.csv'
output_path = 'datasets/language_detection_data.csv'

import pandas as pd
df1 = pd.read_csv(data_path)

if pd.io.common.file_exists(output_path):
    output_df = pd.read_csv(output_path)
else:
    output_df = pd.DataFrame(columns=['language', 'dialect', 'accent', 'text'])
    output_df.to_csv(output_path, index=False)

for column, row in COLUMN_LANGUAGE_META.items():
    
    final_df = pd.DataFrame()
    final_df['language'] = [row['language']] * len(df1)
    final_df['dialect'] = [row['dialect']] * len(df1)
    final_df['accent'] = [row['accent']] * len(df1)
    final_df['text'] = df1[column]

    output_df = pd.concat([output_df, final_df], ignore_index=True)

    output_df.to_csv(output_path, index=False)