  
**APARSIN Transliteration Guidelines** 

**1. Consonants**

| Persian | Roman | Persian | Roman | Persian | Roman |
| ----- | ----- | ----- | ----- | ----- | ----- |
| ب | b | س | s | ل | l |
| پ | p | ش | sh | م | m |
| ت | t | ص | s | ن | n |
| ث | s | خ | kh | و | v |
| ج | j | ز | z | ه | h |
| چ | ch | ض | z | ی | y |
| د | d | ق | q | ژ | zh |
| ح | h | ط | t | غ | gh |
| ر | r | ظ | z | ع | ‘ |
| ذ | z | ک | k | گ | g |
| ء | ’ | ف | f |  |  |

**2. Vowels and Diphthongs**

| Category | Roman | Example | Category | Roman | Example |
| :---: | :---: | :---: | :---: | :---: | :---: |
| /a/ | a | dast (دست) | /ā/ | ā | kār (کار) |
| /e/ | e | gereft (گرفت) | /i/ | i | did (دید) |
| /o/ | o | shod (شد) | /u/ | u | bud (بود) |
| /ay/ | ay | hay (حَی) | /āy/ | āy | āy (آی) |
| /ey/ | ey | pey (پِی) | /ow/ | ow | rowshan/rawshan (روشن) |
| /uy/ | uy | guy (گوی) | /oy/ | oy | khoy (خوی) |

**Special Notes:**

Our dataset uses a **simplified Persian transcription system** adapted from Iranian Studies conventions (see [https://associationforiranianstudies.org/journal/transliteration](https://associationforiranianstudies.org/journal/transliteration)). The system is designed for sentence-level transcription of written Persian, with an emphasis on consistency, reversibility at the orthographic level, and suitability for NLP preprocessing rather than narrow phonetic accuracy.

## **1. Consonants**

The consonants **ق (q)** and **غ (gh)** are transcribed distinctly, even though they may not be phonemically contrastive in all dialects of Persian.  
This distinction is preserved to maintain faithfulness to written Persian orthography, which is important for downstream tasks such as alignment or reconstruction.

*Example:*  
`قهوه →` qahveh, `غیب →` gheyb

The following consonant sets are collapsed in transcription due to identical pronunciation in Modern Persian:

**ث / س / ص → s**

**ذ / ز / ض / ظ → z**

## **Hamze and ʿAyn (ء / ع)**

The symbol ’ (corresponding to ء / ئ and ع) is not written in word-initial position before a vowel.

Example:

 Ali `→` علی 

In words of Arabic origin, ’ is retained in non-initial positions where it corresponds to a written hamze or ʿayn in Persian orthography.

*Example:*  
`رئيس → Ra’is`

## 

## **Morphological Affixes**

Plural markers and derivational suffixes are attached directly to the host word in transcription. Do not separate with a dash (-).

Plural suffixes such as **-ān** and **-hā** are always written as part of the same token.

*Example:*  
انبارها → Anbārhā

**Note about W (و):**

Please write **W** if the letter **و** is pronounced with lip rounding in your language.

## **2. Vowels**

The Persian character **آ** is transcribed as **ā**.

*Example:*  
`کار → Kār`

Short vowels **/a/, /e/, /o/** are transcribed as **a, e, o**, even though they are not normally written in Persian orthography.

Long vowels are transcribed as **ā, i, u**, following Iranian Studies conventions.

Because vowel length can vary gradiently across accents, intermediate realizations (e.g., mid-length *i*) are approximated and transcribed using the nearest standard vowel category.

Please note that in some dialects and/or accents, vowel length can distinguish the meaning of words being transliterated. In such cases, you may repeat the vowel (i.e., double it) to represent the contrast in vowel length. For example, in the Esfahani dialect:

خوردم -> khordam 				خورده‌ام -> khordaam

Diphthongs are represented consistently as **ay, āy, ey, ow, uy, oy**, without introducing additional consonants.

*Example:*  
نوروز → Nowruz

## **Word-Final -e/-eh**

Persian words ending in the vowel **e** that are written with final **ه** are transcribed as **eh**.  
This shows the underlying orthography and distinguishes the ending from **ezāfe**.  
*Example:*  
ایذه → Izeh

When a suffix (e.g., **-i**, **-hā**) is added, this non-pronounced **h** **is omitted** in transcription.  
*Example:*  
ایذه‌ای → Izei

## **Ezāfe**

Ezāfe is transcribed as:

*e** after consonants  
  ***ye** after vowels, including **eh**

*Examples:*  
`کتاب علی → ketabe Ali`  
`خانه‌ی بزرگ → Khānehye bozorg`

Ezāfe is transcribed only when it is syntactically present and commonly pronounced.

