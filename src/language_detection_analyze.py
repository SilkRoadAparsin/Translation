import pandas as pd
import config
from sklearn.metrics import f1_score
from collections import defaultdict

data_dir = 'results/language_detection'
result_dir = 'results'

tasks = ['language', 'dialect', 'accent']

# Store results per task
task_results = {
    task: defaultdict(dict) for task in tasks
}

for model in config.MODELS_8:
    model_name = model.replace('/', '_')
    data_path = f"{data_dir}/results_{model_name}.csv"
    df = pd.read_csv(data_path)

    # Group by language, dialect, accent
    grouped = df.groupby(['gold_language', 'gold_dialect', 'gold_accent'])

    for (lang, dialect, accent), group in grouped:
        row_id = (lang, dialect, accent)

        for task in tasks:
            gold_col = f'gold_{task}'
            pred_col = f'pred_{task}'

            # Skip if columns are missing
            if gold_col not in group.columns or pred_col not in group.columns:
                continue

            gold = group[gold_col]
            pred = group[pred_col]
            pred = pred.fillna("Unknown")

            score = f1_score(gold, pred, average='macro')

            task_results[task][row_id][model_name] = score


# Save one file per task
for task, rows in task_results.items():
    df_out = pd.DataFrame.from_dict(rows, orient='index')
    df_out.index = pd.MultiIndex.from_tuples(
        df_out.index,
        names=['gold_language', 'gold_dialect', 'gold_accent']
    )

    output_path = f"{result_dir}/f1_{task}.csv"
    df_out.to_csv(output_path)

    print(f"Saved: {output_path}")
