# LoRA Fine-tuning

Fine-tunes Llama 3.2 3B Instruct on Dataset B and evaluates it on Datasets A
and C, against an untuned baseline of the same model.

## Contents

| File | Purpose |
|---|---|
| `lora_training.py` | Trains the adapter and generates Dataset A predictions |
| `score_predictions.py` | Scores Dataset A predictions against the gold spans |
| `evaluate_dataset_c.py` | Runs Dataset C, with a second pass for the model's reasoning |
| `results_dataset_a.csv` | Dataset A summary, one row per condition |
| `results_table_c.csv` | Dataset C summary, one row per condition |

The per-item output files and the trained adapters are not committed. The
detail files contain the evaluation sentences, and we keep Datasets A and C
out of the repository so that models scraping GitHub do not end up trained on
our test set. The commands below regenerate all of it.

## Setup

```bash
pip install "transformers>=4.45" peft datasets accelerate torch
hf auth login
```

Llama 3.2 is gated. Accept the licence at
`huggingface.co/meta-llama/Llama-3.2-3B-Instruct` before logging in. Approval
covers the whole Llama 3.2 collection.

## Data

Dataset B is in `../data/` and is committed. The evaluation sets are not, so
generate them first:

```bash
python ../src/dataset_a_gen.py
python ../src/dataset_c_gen.py
```

`dataset_a_gen.py` writes to the current directory, so move its output to
where the scripts look for it:

```bash
mv sentences.csv spans.csv ~/evaldatasets/dataset_a/
```

Expected layout:

```
llm-redactor/
  data/                 dataset_b_train.jsonl, dataset_b_dev.jsonl
  finetuning/           these scripts
~/evaldatasets/
  dataset_a/            sentences.csv, spans.csv
  dataset_c/            dataset_c.csv, dataset_c_eval.jsonl
```

The generators are deterministic, so a regenerated Dataset A matches the one
these results came from (`md5 e6259a24dd877ff48d312ac10b2c25b5`).

## Running

Start with the smoke test. It runs 8 rows, 1 epoch and 5 eval sentences, and
prints the first training pair so you can check the input and output are the
right way round before committing to a full run.

```bash
python lora_training.py --smoke
```

Dataset A, four conditions:

```bash
python lora_training.py                  # fine-tuned, seed 42
python lora_training.py --base-only      # untuned baseline, no training
python lora_training.py --seed 1
python lora_training.py --seed 2
```

Each condition writes to its own directory (`lora_outputs`,
`baseline_outputs`, `lora_outputs_seed1`, `lora_outputs_seed2`). Training
takes about four minutes on an M4 Max and generating 150 sentences takes about
three.

Scoring:

```bash
python score_predictions.py --dump-errors errors.csv
python score_predictions.py --preds baseline_outputs/dataset_a_predictions.json --label "base model"
```

Dataset C, four conditions:

```bash
python evaluate_dataset_c.py                              # fine-tuned
python evaluate_dataset_c.py --base-only                  # baseline
python evaluate_dataset_c.py --legitimacy-prompt          # exemptions named explicitly
python evaluate_dataset_c.py --base-only --legitimacy-prompt
```

`evaluate_dataset_c.py` makes two calls per sentence, one to redact and one to
ask the model to justify keeping or redacting. Both end up in
`dataset_c_review.csv` next to the dataset's own rationale.

## Checking the scorer

Feeding the dataset's gold redactions back through the scorer should return
100% on every cell. Worth running before trusting any model number.

```bash
python -c "
import csv, json
rows = list(csv.DictReader(open('$HOME/evaldatasets/dataset_a/sentences.csv')))
json.dump([{'id': r['id'], 'input': r['text'], 'prediction': r['redacted_text']} for r in rows],
          open('gold_as_preds.json', 'w'), ensure_ascii=False)
"
python score_predictions.py --preds gold_as_preds.json --label "gold ceiling"
```

## Configuration

Llama 3.2 3B Instruct, LoRA rank 8, alpha 16, dropout 0.05, applied to
`q_proj` and `v_proj` only. Learning rate 1e-4 with a cosine schedule, three
epochs, effective batch size 8 (1 x 8 gradient accumulation), 10 warmup steps.
That gives 2.29M trainable parameters, 0.07% of the model. Decoding is greedy
everywhere.

Training runs in bfloat16 on MPS. Weights load on CPU and move to the device
afterwards, because loading straight onto MPS segfaults with recent
safetensors builds. Those two steps need to stay separate. Loss is computed
only on the assistant's output, with the prompt masked to `-100`.

If bfloat16 gives NaN losses, use `--precision fp32`.

## Output files

Per condition:

```
lora_outputs/
  final_model/                  LoRA adapter weights
  dataset_a_predictions.json    model outputs
  scores_detail.json            per-item scores
  dev_metrics.json              final dev loss
```

`errors.csv` lists the failing items with input, gold and prediction side by
side. `dataset_c_review.csv` does the same for Dataset C, with the model's
reasoning next to the dataset's rationale.
