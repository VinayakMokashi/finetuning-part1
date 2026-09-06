# Fine-Tuning Part 1 — Embeddings and Sequence Classification

Two self-contained, runnable examples of fine-tuning a pretrained transformer on a
small labelled dataset, each written as a single class so that every stage of the
pipeline — model, data, loss, training arguments, trainer, evaluation — is one named
method you can read in isolation.

Both scripts follow the same shape: **measure the pretrained model, train it for one
epoch, measure it again**, and print the difference. That before/after pair is the
whole point — it is the smallest experiment that actually demonstrates whether
fine-tuning did anything.

| Script | Task | Base model | Dataset | What it teaches |
| --- | --- | --- | --- | --- |
| [`finetuning01.py`](finetuning01.py) | Embedding / semantic similarity | [`microsoft/mpnet-base`](https://huggingface.co/microsoft/mpnet-base) | [`sentence-transformers/all-nli`](https://huggingface.co/datasets/sentence-transformers/all-nli) (triplet config) | Contrastive fine-tuning with `sentence-transformers`, triplet evaluation, in-batch negatives |
| [`finetuning02.py`](finetuning02.py) | Binary sentiment classification | [`bert-base-uncased`](https://huggingface.co/bert-base-uncased) | [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb) | The Hugging Face `Trainer` loop, tokenisation, and cross-entropy implemented from scratch in NumPy |

Both are deliberately sized to finish on a laptop CPU in minutes — 1,000 training
triplets and 500 training reviews respectively — not to produce a competitive model.

---

## Quick start

```bash
git clone https://github.com/VinayakMokashi/finetuning-part1.git
cd finetuning-part1

python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

python finetuning01.py
python finetuning02.py
```

Both scripts download their model and dataset from the Hugging Face Hub on first run
(roughly 1 GB in total, cached under `~/.cache/huggingface`). No account, token or API
key is required.

**Requirements:** Python 3.9+ and about 4 GB of free disk space. A GPU is optional —
see [Running on a GPU](#running-on-a-gpu).

---

## `finetuning01.py` — fine-tuning an embedding model

### The task

An embedding model maps a sentence to a vector. A *good* embedding model puts
sentences that mean the same thing close together and sentences that contradict each
other far apart. `microsoft/mpnet-base` is a pretrained language model that has never
been trained to do this, so its raw pooled output is close to useless for similarity —
the run below starts at 53% on a task where guessing at random scores 50%.

The AllNLI triplet dataset supplies exactly the supervision needed: each row is an
`(anchor, positive, negative)` triple, where the positive entails the anchor and the
negative contradicts it.

### How it works

`MultipleNegativesRankingLoss` treats every batch as a miniature retrieval problem.
For each anchor it computes the similarity to its own positive and to *every other
example in the batch*, then applies cross-entropy so that the correct positive scores
highest. This is why the batch sampler matters:

```python
batch_sampler=BatchSamplers.NO_DUPLICATES
```

A duplicated sentence inside a batch would show up as a "negative" that is actually
correct, sending a contradictory gradient. `NO_DUPLICATES` guarantees that never
happens.

### Evaluation

Two independent measurements are taken, and the distinction is worth understanding:

- **`TripletEvaluator`** (reported as `eval_all-nli-dev_cosine_accuracy`) runs *during*
  training on the `dev` split, every 100 steps.
- **`evaluate_cosine_accuracy()`** is a hand-written check on the held-out `test`
  split, run once before training and once after. It embeds the anchors, positives and
  negatives, computes paired cosine distances, and counts the fraction of triplets
  where the positive is closer to the anchor than the negative:

  ```python
  correct = sum(1 for pos, neg in zip(pos_cos_distances, neg_cos_distances) if pos < neg)
  accuracy = correct / len(pos_cos_distances)
  ```

  Writing it out longhand rather than calling a library is the point — it shows that
  "triplet accuracy" is nothing more than a comparison of two distances.

### Class reference

| Method | Responsibility |
| --- | --- |
| `setup_model()` | Loads `microsoft/mpnet-base` as a `SentenceTransformer`, with model-card metadata attached |
| `load_dataset()` | Pulls the `triplet` config of `sentence-transformers/all-nli` |
| `create_train_dataset()` | First 1,000 rows of `train` |
| `create_test_dataset()` | First 200 rows of `test` — the held-out before/after set |
| `create_eval_dataset()` | First 200 rows of `dev` — used for in-training evaluation |
| `setup_dev_evaluator()` | Builds the `TripletEvaluator` over the dev split |
| `setup_loss()` | `MultipleNegativesRankingLoss` |
| `setup_training_args()` | One epoch, batch size 16, 10% warmup, eval/save/log every 100 steps |
| `setup_trainer()` | Assembles the `SentenceTransformerTrainer` |
| `evaluate_cosine_accuracy(dataset)` | The manual triplet-accuracy metric described above |
| `train_model()` | Runs `trainer.train()` |

### Output

Checkpoints are written to `models/mpnet-base-all-nli-triplet/` (git-ignored;
`save_total_limit=2` keeps only the two most recent).

---

## `finetuning02.py` — fine-tuning a sequence classifier

### The task

`bert-base-uncased` is loaded through `AutoModelForSequenceClassification` with
`num_labels=2`. The pretrained encoder weights are reused, but the classification head
on top is **brand new and randomly initialised** — Hugging Face prints a report saying
exactly that:

```
classifier.weight | MISSING
classifier.bias   | MISSING
```

This is not an error. It is the thing being fine-tuned. A random 2-way head gives a
cross-entropy loss of roughly `ln(2) ≈ 0.693`, which is the number to read the
"before" measurement against.

### How it works

Reviews are tokenised to a fixed 512-token width (`padding="max_length"`,
`truncation=True`) and handed to the standard Hugging Face `Trainer`, which owns the
training loop, optimiser and scheduler. The subsets are shuffled with a fixed seed so
the data selection is reproducible:

```python
self.dataset["train"].shuffle(seed=42).select(range(500))
```

### Cross-entropy, computed twice

The most instructive part of this script is that the evaluation loss is computed two
ways that must agree:

- **`evaluate_loss()`** pulls the logits out as NumPy arrays and calls
  `cross_entropy_loss()`, a from-scratch implementation: one-hot encode the labels,
  subtract the row max for numerical stability, softmax, take the log, and average the
  negative log-probability of the true class.
- **`evaluate_loss_torch()`** does the same thing with `torch.nn.CrossEntropyLoss`.

The `- np.max(logits, axis=1, keepdims=True)` shift is the detail worth noticing:
softmax is invariant to a constant offset per row, and subtracting the max keeps
`np.exp` from overflowing on large logits. On the reference machine the two functions agree
to within `2e-8` (`0.69903267` against `0.69903265` on the same batch), which is what
makes the from-scratch version trustworthy.

### Class reference

| Method | Responsibility |
| --- | --- |
| `setup_tokenizer()` / `setup_model()` | Loads the `bert-base-uncased` tokenizer and a 2-label classification head |
| `load_dataset()` | Pulls `stanfordnlp/imdb` |
| `create_train_subset()` | 500 reviews from `train`, shuffled with `seed=42` |
| `create_test_subset()` | 32 reviews from `test`, shuffled with `seed=42` |
| `tokenize_function()` / `tokenize_dataset()` | Pads and truncates to 512 tokens, applied batched |
| `setup_training_args()` | One epoch, lr `2e-5`, batch size 1, weight decay `0.01`, eval each epoch |
| `setup_trainer()` | Assembles the `Trainer` |
| `evaluate_loss()` | Eval loss via the NumPy cross-entropy |
| `evaluate_loss_torch()` | Eval loss via `torch.nn.CrossEntropyLoss`, as a cross-check |
| `cross_entropy_loss()` | Numerically stable softmax + negative log-likelihood, from scratch |
| `train_model()` | Runs `trainer.train()` |

### Output

Checkpoints are written to `./results/` (git-ignored).

---

## Verified results

Measured on the reference machine below, running each script exactly as committed.

### `finetuning01.py`

| Metric | Value |
| --- | --- |
| Triplet accuracy on held-out `test`, **before** fine-tuning | `0.5300` |
| Triplet accuracy on held-out `test`, **after** fine-tuning | `0.8400` |
| Change | **+0.3100** |
| `TripletEvaluator` dev cosine accuracy at end of epoch | `0.8750` |
| Final training loss | `1.016` |
| Training wall-clock (63 steps) | ~3 min 45 s |

The model goes from barely better than chance to correctly ordering 84% of unseen
triplets after a single epoch over 1,000 examples. That jump is the entire lesson: the
pretrained encoder already knew a great deal about language, and one epoch of
contrastive supervision was enough to reshape its output space into something usable
for similarity search.

### `finetuning02.py`

| Metric | Value |
| --- | --- |
| Eval cross-entropy loss, **before** fine-tuning | `0.6809` |
| Eval cross-entropy loss, **after** fine-tuning | `0.3024` |
| Change | **-0.3784** |
| Final training loss | `0.6058` |
| Training wall-clock (500 steps) | ~24 min 50 s |

The "before" figure sits just under `ln(2) ≈ 0.6931`, which is exactly where an
untrained 2-way head should sit — the model starts with no opinion about sentiment.
After 500 single-example steps the loss has more than halved.

Note that the "before" number moves between runs (`0.6809` here, `0.7297` on another
run of the same code) precisely because that head is random at startup. The *direction*
is the reproducible result, not the starting value.

Reference machine: Windows 11, Python 3.12.3, **CPU only** (no CUDA), torch 2.12.1,
transformers 5.12.1, datasets 5.0.0, sentence-transformers 5.6.0, accelerate 1.14.0.

Your numbers will differ. Neither script sets a global training seed, so dropout and
shuffling draw differently on each run, and the randomly initialised classification
head in `finetuning02.py` starts somewhere new every time.

---

## Configuration

The knobs most worth turning are all in the `setup_training_args()` methods.

**`finetuning01.py`**

| Setting | Default | Notes |
| --- | --- | --- |
| Training rows | `range(1000)` in `create_train_dataset()` | The full split has 557,850 rows |
| `num_train_epochs` | `1` | |
| `per_device_train_batch_size` | `16` | Larger batches give `MultipleNegativesRankingLoss` more in-batch negatives, which usually helps |
| `fp16` / `bf16` | `False` / `False` | See below |
| `eval_steps` / `save_steps` | `100` / `100` | |

**`finetuning02.py`**

| Setting | Default | Notes |
| --- | --- | --- |
| Training rows | `range(500)` in `create_train_subset()` | The full split has 25,000 rows |
| Test rows | `range(32)` in `create_test_subset()` | Small enough that the loss is noisy |
| `learning_rate` | `2e-5` | The usual starting point for BERT fine-tuning |
| `per_device_train_batch_size` | `1` | Raise to 8 or 16 on a GPU for a large speedup |
| `num_train_epochs` | `1` | 2–3 is typical for real classification fine-tuning |

### Running on a GPU

Both scripts pick up CUDA automatically when it is available; nothing needs to change
to use one. To go faster, enable mixed precision in `finetuning01.py`:

```python
fp16=True,   # most CUDA GPUs
bf16=True,   # Ampere (RTX 30xx / A100) and newer — prefer this where supported
```

Set only one of the two. In `finetuning02.py`, raise `per_device_train_batch_size` to
8 or 16 first — batch size 1 leaves a GPU almost entirely idle.

---

## Troubleshooting

**`ImportError: Using the Trainer with PyTorch requires accelerate>=1.1.0`**
The Hugging Face `Trainer` cannot even construct its `TrainingArguments` without
`accelerate`, and the base `transformers` install does not pull it in. It is listed in
`requirements.txt`; if you installed packages by hand, run
`pip install "accelerate>=1.1.0"`.

**`HfUriError: Repository id must be namespace/name, got imdb`**
You are on an older revision, or copied a bare `load_dataset("imdb")` call from an
older tutorial. `datasets` 5.x resolves ids through the `huggingface_hub` `hf://` URI
parser, which no longer accepts unqualified aliases. Use `stanfordnlp/imdb`.

**`The warmup_ratio argument is deprecated in Transformers v5+`**
Harmless, and emitted by `finetuning01.py` on transformers 5.x. `warmup_ratio=0.1` is
still honoured; the eventual replacement is `warmup_steps` given as a float. The script
keeps `warmup_ratio` so that it also runs on transformers 4.x.

**`UserWarning: pin_memory is set as true but no accelerator is found`**
Harmless. The `Trainer` enables pinned memory by default to speed up host-to-GPU
transfers; on a CPU-only machine there is nothing to transfer to.

**A `LOAD REPORT` table listing `UNEXPECTED` and `MISSING` keys**
Expected, and printed by both scripts. `UNEXPECTED` keys are the pretraining heads
(`lm_head`, `cls.predictions`) that the downstream architecture discards. `MISSING`
keys are the new task head being initialised from scratch — the parameters you are
about to train.

**`UserWarning: huggingface_hub cache-system uses symlinks by default ...` (Windows)**
Harmless; the cache falls back to copying files, which uses more disk. Silence it by
setting `HF_HUB_DISABLE_SYMLINKS_WARNING=1`, or enable Windows Developer Mode.

**Downloads are slow or rate-limited**
The Hub warns about unauthenticated requests. Anonymous access works fine for these
public models; setting an `HF_TOKEN` environment variable raises the rate limit.

**Out of memory**
Lower `per_device_train_batch_size`, or shrink the training subset in
`create_train_dataset()` / `create_train_subset()`.

---

## Repository layout

```
.
├── finetuning01.py    # Embedding fine-tuning: MPNet + AllNLI triplets
├── finetuning02.py    # Sequence classification: BERT + IMDb
├── requirements.txt   # Python dependencies
├── .gitattributes     # Normalises line endings to LF
├── .gitignore         # Excludes checkpoints, caches and virtual environments
├── LICENSE            # MIT
└── README.md
```

Training checkpoints (`models/`, `results/`) are produced by running the scripts and
are deliberately untracked — they are large and fully reproducible.

## Credits

Coursework from Week 7 (Fine-Tuning) of a GenAI course, following *Hands-On Large
Language Models* by Jay Alammar and Maarten Grootendorst. Models and datasets are the
property of their respective authors and carry their own licenses.

## License

[MIT](LICENSE)
