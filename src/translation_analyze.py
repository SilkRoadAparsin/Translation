import pandas as pd
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from bert_score import score as bert_score
from tqdm import tqdm

# ---------------------------
# Load data
# ---------------------------
data_path = 'results/translation/data.csv'
df = pd.read_csv(data_path)

# ---------------------------
# Create combined language column
# ---------------------------
df['language_full'] = (
    df['language'].astype(str) + '-' +
    df['dialect'].astype(str) + '-' +
    df['accent'].astype(str)
)

# ---------------------------
# Prepare containers
# ---------------------------
bleu_results = {}
bert_results = {}

smoothie = SmoothingFunction().method4

# ---------------------------
# Compute scores
# ---------------------------
for (model, language), group in tqdm(
        df.groupby(['model', 'language_full']),
        desc="Computing scores"
):
    references = group['english_translation'].astype(str).tolist()
    hypotheses = group['pred_english_translation'].astype(str).tolist()

    # ---- BLEU ----
    references_bleu = [[ref.split()] for ref in references]
    hypotheses_bleu = [hyp.split() for hyp in hypotheses]

    bleu = corpus_bleu(
        references_bleu,
        hypotheses_bleu,
        smoothing_function=smoothie
    )

    bleu_results.setdefault(language, {})[model] = bleu

    # ---- BERTScore ----
    P, R, F1 = bert_score(
        hypotheses,
        references,
        lang="en",
        rescale_with_baseline=True
    )

    bert_f1 = F1.mean().item()
    bert_results.setdefault(language, {})[model] = bert_f1

# ---------------------------
# Convert to tables
# ---------------------------
bleu_table = pd.DataFrame.from_dict(bleu_results, orient='index').sort_index()
bert_table = pd.DataFrame.from_dict(bert_results, orient='index').sort_index()

# ---------------------------
# Beautify tables
# ---------------------------
bleu_table.index.name = "language-dialect-accent"
bert_table.index.name = "language-dialect-accent"

bleu_table = bleu_table.round(4)
bert_table = bert_table.round(4)

# ---------------------------
# Print tables
# ---------------------------
print("\n=== BLEU SCORE TABLE ===")
print(bleu_table)

print("\n=== BERT SCORE (F1) TABLE ===")
print(bert_table)

# ---------------------------
# Optional: save tables
# ---------------------------
bleu_table.to_csv("results/translation/bleu_scores.csv")
bert_table.to_csv("results/translation/bert_scores.csv")
