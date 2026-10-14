# GenoComp

**A Causal Transformer and Arithmetic Coding Framework for Lossless Genomic Sequence Compression**

GenoComp is a reference-free DNA sequence compression research project that combines overlapping 4-mer tokenization, a causal Transformer probability model, and adaptive arithmetic coding. Instead of storing only the model's most likely next-token prediction, the compressor uses the model's probability distribution to encode each observed token. During decoding, the same trained model and the same decoded context regenerate the probability distributions needed to recover the token stream.

The implementation is designed for FASTA-style DNA sequence files and was developed and evaluated in a Kaggle Notebook environment.

> **Important scope note:** The supplied implementation removes all characters other than `A`, `C`, `G`, and `T` before tokenization. It therefore reconstructs the **cleaned DNA sequence**, not the original FASTA file byte-for-byte. FASTA headers, line wrapping, and removed ambiguity characters are not preserved by the shown code. A side channel would be required to restore those details.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [How GenoComp Works](#how-genocomp-works)
- [Model Architecture](#model-architecture)
- [Configuration](#configuration)
- [Kaggle Notebook Environment](#kaggle-notebook-environment)
- [Dataset Layout](#dataset-layout)
- [Installation and Execution](#installation-and-execution)
- [Training Procedure](#training-procedure)
- [Compression and Decompression](#compression-and-decompression)
- [Output Files](#output-files)
- [Evaluation Metrics](#evaluation-metrics)
- [Reported Research Results](#reported-research-results)
- [Reproducibility and Compatibility](#reproducibility-and-compatibility)
- [Current Limitations](#current-limitations)
- [Future Work](#future-work)
- [Research Context](#research-context)

---

## Overview

DNA sequences contain repeated patterns and local dependencies that can be exploited to reduce storage requirements. GenoComp learns these patterns from a training corpus and uses a Transformer to estimate the probability of the next overlapping 4-mer given the preceding context. An arithmetic coder converts those probabilities into a compact bitstream.

The framework is **reference-free**: it does not require a known reference genome to compress a target sequence. However, the encoder and decoder must have access to compatible, identical trained model weights and must produce identical probability distributions in the same sequence of contexts.

### Design goals

- Use a causal Transformer for context-dependent next-token probability estimation.
- Use overlapping 4-mers to represent local DNA sequence patterns.
- Use adaptive arithmetic coding to encode the actual token sequence rather than relying on potentially incorrect greedy predictions.
- Reconstruct the cleaned DNA sequence exactly when encoding and decoding use the same model and configuration.
- Evaluate compression ratio, bits per base, runtime, and reconstruction correctness.

---

## Key Features

- **FASTA input handling:** Reads sequence lines and ignores FASTA header lines beginning with `>`.
- **DNA cleaning:** Converts bases to uppercase and retains only `A`, `C`, `G`, and `T`.
- **Overlapping k-mer tokenization:** Uses a sliding window of `K = 4` with stride 1.
- **Causal Transformer:** Six Transformer encoder layers with causal masking, eight attention heads, and sinusoidal positional encoding.
- **Mixed-precision training:** Uses PyTorch automatic mixed precision and gradient scaling.
- **Adaptive arithmetic coding:** Uses a 32-bit interval arithmetic coder with integer cumulative distribution functions (CDFs).
- **Synchronized decoding:** Recomputes probabilities from the same model and previously decoded tokens.
- **Integrity check:** Compares the reconstructed cleaned DNA sequence with the cleaned input.
- **Training diagnostics:** Saves loss, next-token accuracy, and bits-per-base plots.

---

## How GenoComp Works

The end-to-end workflow is:

1. **Read FASTA:** Ignore header lines and concatenate sequence lines.
2. **Clean DNA:** Convert to uppercase and discard every character outside `ACGT`.
3. **Tokenize:** Convert each overlapping 4-mer to an integer ID from `0` to `255`.
4. **Train the Transformer:** Learn next-token distributions using shifted token sequences.
5. **Predict probabilities:** The `ProbEngine` uses the most recent context, up to 320 tokens, to produce a probability distribution for the next token.
6. **Build an integer CDF:** Convert probabilities into integer symbol frequencies suitable for arithmetic coding.
7. **Encode:** Arithmetic-code each true token and write a metadata header followed by the coded bytes.
8. **Decode:** Read the metadata and bitstream, recover tokens one at a time, and update the context after every decoded token.
9. **Reconstruct:** Convert the token sequence back into DNA by taking the first full 4-mer and appending the final base of every subsequent 4-mer.
10. **Verify:** Compare the reconstructed sequence with the cleaned DNA sequence.

### Why use overlapping 4-mers?

For a sequence such as `AGCCTAACTG`, the stride-1 4-mer windows are:

```text
AGCC, GCCT, CCTA, CTAA, TAAC, AACT, ACTG
```

For a DNA sequence of length \(N\), the number of overlapping k-mers is:

\[
N_{\text{tokens}} = \max(0, N-K+1)
\]

With `K = 4`, adjacent tokens share three bases, and every token after the first contributes one new base during reconstruction. This overlap preserves the cleaned sequence while giving the Transformer a token vocabulary of only \(4^4 = 256\) symbols.

### K-mer ID mapping

The code maps bases to digits in base 4:

| Base | Digit |
|---|---:|
| A | 0 |
| C | 1 |
| G | 2 |
| T | 3 |

For a k-mer \(b_1b_2b_3b_4\), its integer ID is calculated as:

\[
\operatorname{ID}(b_1b_2b_3b_4)
=\sum_{j=1}^{4}s(b_j)\,4^{4-j}
\]

The mapping is deterministic and reversible.

---

## Model Architecture

The model estimates the probability of the next k-mer conditioned on the preceding tokens:

\[
P(x_t \mid x_{t-c}, \ldots, x_{t-1})
\]

where the configured context length is 320 tokens.

### Architecture components

1. **Token embedding:** Maps each token ID to a 256-dimensional vector.
2. **Sinusoidal positional encoding:** Adds fixed positional information to the token embeddings. The implementation sets the maximum positional-encoding length to `BLOCK_SIZE + 1`.
3. **Causal Transformer encoder:** Six layers; each layer uses multi-head self-attention and a feed-forward network. The upper-triangular mask prevents a position from attending to future positions.
4. **Output projection:** Maps the final hidden representation to 256 logits.
5. **Softmax:** Converts logits into a probability distribution over all possible 4-mers.

The implementation uses `batch_first=True` and `norm_first=True` in `nn.TransformerEncoderLayer`.

### Training objective

The model is trained using cross-entropy loss on shifted token sequences. For each block of 321 tokens, the model learns to predict the next token from the preceding token positions. The training code also tracks next-token accuracy and a cross-entropy-derived bits-per-base estimate.

---

## Configuration

The values below are taken from the supplied code. Change them only if you intend to run a different experiment.

| Parameter | Code value | Meaning |
|---|---:|---|
| `DEVICE` | `"cuda"` if available, otherwise `"cpu"` | Computation device selected by PyTorch |
| `K` | `4` | k-mer length |
| `KMER_VOCAB` | `4**K = 256` | Number of possible k-mer tokens |
| `BLOCK_SIZE` | `320` | Maximum context/block input length |
| `EMBED_DIM` | `256` | Token embedding dimension |
| `N_HEADS` | `8` | Attention heads per Transformer layer |
| `N_LAYERS` | `6` | Number of Transformer encoder layers |
| `FF_DIM` | `1024` | Feed-forward hidden dimension |
| `DROPOUT` | `0.1` | Transformer dropout |
| `EPOCHS` | `30` | Training epochs in the supplied code |
| `BATCH_SIZE` | `32` | Training and validation batch size |
| `LR` | `2e-4` | Initial learning rate |
| AdamW weight decay | `1e-4` | Weight decay passed to AdamW |
| Gradient clipping | `1.0` | Maximum gradient norm |
| Validation split | `10%` | Last 10% of shuffled blocks |
| Block shuffle seed | `42` | Seed used before splitting blocks |
| Scheduler | Cosine annealing | `T_max = EPOCHS`, `eta_min = LR/20` |
| Positional encoding maximum length | `BLOCK_SIZE + 1 = 321` | Maximum positions represented by the encoding |
| Arithmetic coder precision | `32` bits | Precision of the arithmetic-coding interval |
| Output directory | `/kaggle/working` | Location for generated artifacts |

### Data paths configured in the code

```python
TRAIN_DIR = "/kaggle/input/datasets/arkadipmaths/transformer-train-small"
TEST_DIR  = "/kaggle/input/datasets/arkadipmaths/transformer-test-small"
OUT_DIR   = "/kaggle/working"
```

These are Kaggle-specific paths. If you run GenoComp elsewhere, update `TRAIN_DIR`, `TEST_DIR`, and `OUT_DIR` to match your local directories.

### Training details

- Optimizer: `torch.optim.AdamW`
- Initial learning rate: `2e-4`
- Weight decay: `1e-4`
- Learning-rate scheduler: `CosineAnnealingLR`
- Minimum learning rate: `LR / 20`
- Loss: cross-entropy
- Mixed precision: `autocast()` and `GradScaler()`
- Gradient clipping: norm `1.0`
- Model checkpoint: lowest validation-loss checkpoint saved as `best_model.pt`

**Epoch-count distinction:** The supplied code sets `EPOCHS = 30`. The research paper reports that its main base-model results were obtained from a 10-epoch training run. The README keeps the code's configuration value rather than silently substituting the paper's experimental setting. Results may differ if the code is run with 30 epochs or with a different training corpus.

---

## Kaggle Notebook Environment

The reported experiments used a Kaggle Notebook configured with **2 × NVIDIA T4 GPU units** and mixed-precision training.

### Suggested notebook setup

1. Create or open a Kaggle Notebook.
2. Attach the training dataset and test dataset.
3. In the notebook's accelerator settings, select the GPU option that provides **two T4 GPU units**, if available for your account and session.
4. Confirm that PyTorch detects CUDA:
   ```python
   import torch

   print("PyTorch version:", torch.__version__)
   print("CUDA available:", torch.cuda.is_available())
   print("CUDA device count:", torch.cuda.device_count())

   if torch.cuda.is_available():
       for i in range(torch.cuda.device_count()):
           print(i, torch.cuda.get_device_name(i))
   ```
5. Run the code after confirming that the dataset paths match the attached Kaggle datasets.

**GPU usage clarification:** The supplied script selects `cuda` when available and moves the model and batches to `DEVICE`. It does **not** explicitly implement `DataParallel`, `DistributedDataParallel`, or another multi-GPU training strategy. Therefore, although the reported notebook environment used two T4 GPU units, the shown code alone does not guarantee that both GPUs are actively used. For strict reproducibility, record the actual device count and the notebook accelerator configuration for each run.

---

## Dataset Layout

The code expects the training and testing directories to contain files that can be read as FASTA-like sequence text. Each file is read independently; directory entries that are not regular files are skipped.

A typical layout is:

```text
input/
├── transformer-train-small/
│   ├── training_genome_1.fasta
│   ├── training_genome_2.fasta
│   └── ...
└── transformer-test-small/
    ├── test_genome_1.fasta
    ├── test_genome_2.fasta
    └── ...
```

The filenames above are illustrative. Use the actual directory and file names from your dataset.

### Training data processing

- All files in `TRAIN_DIR` are read and tokenized.
- Tokens from the files are appended to a single token list.
- The combined list is divided into blocks of `BLOCK_SIZE + 1` tokens.
- The blocks are shuffled with `random.seed(42)`.
- The blocks are split into 90% training and 10% validation data.

Because tokens from files are concatenated before blocks are formed, a block can potentially cross a file boundary. If strict per-genome separation is required, the block-generation procedure should be changed accordingly.

### Test data processing

Each regular file in `TEST_DIR` is cleaned, tokenized, compressed, decompressed, and checked against its cleaned sequence.

---

## Installation and Execution

### Dependencies

The supplied code imports:

- Python standard library: `os`, `json`, `struct`, `random`, `math`, `time`
- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)

The implementation uses PyTorch CUDA automatic mixed-precision APIs. The exact CUDA-enabled PyTorch build should be compatible with the GPU environment in which the notebook runs.

For a local environment, install compatible versions of PyTorch, NumPy, and Matplotlib. For example:

```bash
pip install numpy matplotlib
```

Install the appropriate PyTorch build by following the official PyTorch installation selector: https://pytorch.org/get-started/locally/

### Run the project

The supplied script is a combined training-and-evaluation pipeline. After configuring the data paths:

1. Run the notebook cell or Python script containing the implementation.
2. Wait for the training loop to finish.
3. The best validation-loss checkpoint is saved in the output directory.
4. Training curves are saved as `training_plots.png`.
5. The script processes every regular file in `TEST_DIR`.
6. Compression and decompression metrics are printed for each test file.
7. The compressed stream and reconstructed cleaned sequence are written to the output directory.

The current code does not expose separate command-line commands for training, compression, and decompression. It performs these steps sequentially in one run.

---

## Training Procedure

The training pipeline follows these stages:

1. Read all training files and remove FASTA headers.
2. Uppercase the sequence and keep only valid `ACGT` bases.
3. Convert the cleaned sequences into overlapping 4-mer token IDs.
4. Create blocks of 321 tokens: 320 context positions plus the shifted next-token target.
5. Shuffle the blocks and split them into training and validation subsets.
6. Train the causal Transformer with AdamW and mixed precision.
7. Evaluate validation loss, next-token accuracy, and the bits-per-base estimate after each epoch.
8. Save the model state whenever validation loss improves.
9. Reload the best checkpoint before testing.
10. Save plots of training and validation metrics.

The saved checkpoint contains the model's `state_dict`. The shown script does not package the trained model weights into each compressed output file.

---

## Compression and Decompression

### Probability engine (`ProbEngine`)

The `ProbEngine` maintains a list of previously observed or decoded token IDs.

- It starts with an empty context for each file.
- With an empty context, it returns a uniform probability distribution over the 256-token vocabulary.
- Otherwise, it runs the Transformer on the most recent `BLOCK_SIZE` tokens.
- It applies softmax to the final position's logits to obtain the next-token probabilities.
- After a token is encoded or decoded, `advance(token)` appends it to the context.

The encoder advances using the true token. The decoder advances using the token it has just recovered. With identical weights, settings, and token history, both sides can regenerate the same distributions.

### Integer CDF

The probability distribution is converted into integer frequencies, and a cumulative distribution function is built for arithmetic coding. The implementation uses a 32-bit arithmetic-coding interval and handles interval renormalization and pending bits.

### Compressed file structure

The output extension used by the code is `.dnacomp`. Its binary layout is:

```text
+------------------------------+
| 4-byte big-endian JSON length |
+------------------------------+
| JSON metadata bytes           |
+------------------------------+
| Arithmetic-coded bitstream    |
+------------------------------+
```

The JSON metadata includes:

| Field | Meaning |
|---|---|
| `n_tokens` | Number of encoded k-mer tokens |
| `dna_len` | Length of the cleaned DNA sequence |
| `k` | k-mer size (`4` in the supplied configuration) |
| `version` | Format version (`1`) |

The decoder reads the metadata length, parses the JSON, extracts the arithmetic-coded bytes, and uses `n_tokens` to determine how many tokens to decode.

### Model compatibility requirement

The `.dnacomp` file does **not** contain the Transformer weights. Decompression requires the same trained model weights and compatible configuration as compression. Using different weights, a different token mapping, or different arithmetic-CDF behavior can cause decoding to fail or produce an incorrect sequence.

### Reconstruction guarantee and scope

The overlap reconstruction works by taking the complete first 4-mer and then appending the last base from each subsequent token. The script truncates the reconstructed sequence to the recorded cleaned DNA length and checks it against the cleaned input.

This validates the cleaned sequence. Since the current cleaner removes non-ACGT characters and the metadata does not store FASTA headers or original formatting, it does not provide byte-for-byte restoration of the original FASTA file.

---

## Output Files

For a test file named `sample.fasta`, the code writes the following artifacts under `/kaggle/working`:

| Output | Description |
|---|---|
| `best_model.pt` | Model state with the lowest observed validation loss |
| `training_plots.png` | Training/validation loss, next-token accuracy, and BPB plots |
| `sample.dnacomp` | Metadata header and arithmetic-coded token stream |
| `sample_reconstructed.txt` | Reconstructed cleaned DNA sequence |

The output prefix is generated from the input filename without its extension. If two different input files have the same stem, their output names may collide.

---

## Evaluation Metrics

### Compression ratio

The code calculates compression ratio against the original input file size in bytes:

\[
CR = \frac{S_{\text{original file}}}{S_{\text{compressed file}}}
\]

A higher ratio indicates a smaller compressed file relative to the original file. Since the denominator includes the `.dnacomp` metadata header and the numerator is the original FASTA file size, the metric compares the compressed artifact against the original text file, even though the compressor only encodes cleaned DNA content.

### Compressed bits per base

The code calculates:

\[
BPB_{\text{compressed}} =
\frac{8 \times S_{\text{compressed file in bytes}}}
{N_{\text{cleaned bases}}}
\]

Lower compressed BPB indicates fewer compressed bits per cleaned DNA base.

### Model BPB

The script also reports a cross-entropy-derived model BPB estimate. This measures the model's predictive uncertainty and is distinct from the actual compressed BPB, which includes the complete compressed artifact and metadata.

### Runtime

The script reports compression time, decompression time, and total time. These measurements depend on the input size, model, hardware, and runtime environment.

### Reconstruction accuracy

The script compares the reconstructed cleaned sequence with the cleaned input. A perfect result is reported as `✓ PERFECT`, with reconstruction accuracy of `100.0000%`. This is a sequence-level check on the cleaned ACGT string, not a byte-level comparison of the original FASTA file.

---

## Reported Research Results

The accompanying GenoComp paper reports the following base-model results on six held-out genomes. These are **reported research results**, not a guarantee that every run of the supplied code will reproduce them; results depend on the exact checkpoint, corpus, epoch count, hardware, and implementation.

| Test genome | Original size (bytes) | Compressed size (bytes) | Compression ratio | Compressed BPB |
|---|---:|---:|---:|---:|
| BuEb | 18,940 | 4,712 | 4.020× | 1.9903 |
| AgPh | 43,970 | 10,849 | 4.053× | 1.9739 |
| YeMi | 73,689 | 17,122 | 4.304× | 1.8588 |
| AeCa | 1,554,000 | 393,941 | 4.039× | 1.9808 |
| HePy | 1,629,000 | 404,417 | 4.124× | 1.9399 |
| ThTh | 2,104,000 | 490,926 | 4.388× | 1.8460 |

The paper reports compression ratios from **4.020× to 4.388×**, with lossless reconstruction of the cleaned DNA sequences. It also identifies token-by-token Transformer inference as the main runtime bottleneck, especially for larger genomes.

---

## Reproducibility and Compatibility

For comparable results, keep the following consistent:

- Training and test datasets and their file contents.
- `K`, token mapping, context length, model dimensions, and number of layers.
- Model checkpoint used for both compression and decompression.
- Arithmetic coder precision, integer-frequency/CDF construction, and implementation.
- Cleaning rules and reconstruction length.
- PyTorch and NumPy versions, device configuration, and mixed-precision behavior.
- Random seeds and block-generation order.

The code sets `random.seed(42)` before shuffling the blocks, but it does not set all possible Python, NumPy, and PyTorch random seeds. Full training reproducibility is therefore not guaranteed by that seed alone.

For long-term storage or distribution, preserve the exact model checkpoint and the code/configuration version used to produce each `.dnacomp` file. Consider recording a checkpoint hash and implementation version alongside the artifact.

---

## Current Limitations

1. **Cleaning is not byte-preserving.** Characters outside `ACGT`, FASTA headers, and line formatting are not retained in the compressed stream.
2. **Model weights must be available separately.** The compressed file does not include the model checkpoint.
3. **Slow token-by-token inference.** The Transformer is run for every token during encoding and decoding, which can make processing large genomes time-consuming.
4. **Multi-GPU use is not implemented explicitly.** The code selects CUDA but does not set up data parallelism or distributed training.
5. **Kaggle-specific paths.** The default data paths must be updated for other environments.
6. **Single combined workflow.** Training, testing, compression, and decompression are not yet separated into reusable command-line tools.
7. **Limited metadata and format versioning.** The metadata includes token count, cleaned DNA length, k-mer size, and a version integer, but does not identify the model checkpoint or preserve original FASTA metadata.
8. **Dataset boundary handling.** Training tokens from all files are concatenated before block generation, so blocks can cross file boundaries.
9. **Hardware-dependent performance.** Runtime and GPU availability vary by environment.

---

## Future Work

Potential improvements consistent with the project's direction include:

- Improving inference efficiency while preserving the Transformer-based probability model.
- Explicitly using multiple GPUs for training where supported.
- Expanding and balancing the training corpus.
- Improving arithmetic-coder efficiency and reducing probability-quantization overhead.
- Adding a reversible side channel for ambiguity codes, headers, and original FASTA formatting.
- Storing model/checkpoint identifiers in the compressed-file metadata.
- Separating training, encoding, and decoding into dedicated entry points.
- Adding automated tests for tokenization round trips, coder round trips, corrupted inputs, and end-to-end reconstruction.
- Benchmarking on a larger and more diverse set of genomes.

---

## Research Context

**Project title:** *GenoComp: A Causal Transformer and Arithmetic Coding Framework for Lossless Genomic Sequence Compression*

The accompanying paper describes GenoComp as a reference-free compressor that combines overlapping k-mer tokenization, a causal Transformer, and adaptive arithmetic coding. Its central design is to encode the actual token sequence using model probabilities rather than storing correction data for incorrect greedy predictions.

The reported work uses an eight-genome training corpus of approximately 335.5 MB and evaluates six held-out genomes. Consult the accompanying research paper for the full methodology, dataset references, ablation studies, comparisons, and discussion of limitations.

---

## Citation

If you use GenoComp in academic work, please cite the accompanying research paper and include the appropriate bibliographic details once the paper's publication status and final citation format are established.

**Authors:** Ankit Kumar Dam, Aratrik Sarkar, Arkadip Ray, Arnab Chakraborty, Aryesh Das, Subhankar Roy, Partha Ghosh, Arup Kumar Bhattacharjee, Debarjeeta Biswas, and Anirban Mukhopadhyay.

**Title:** *GenoComp: A Causal Transformer and Arithmetic Coding Framework for Lossless Genomic Sequence Compression.*

---

## Acknowledgements

The experiments were conducted in a Kaggle Notebook environment using PyTorch and GPU acceleration. The research evaluates reference-free DNA compression on held-out genomic sequences and builds on established work in genomic compression, sequence modeling, and arithmetic coding.
