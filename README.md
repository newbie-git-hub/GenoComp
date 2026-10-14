# GenoComp

GenoComp is a reference-free DNA sequence compression project that combines overlapping 4-mer tokenization, a causal Transformer, and adaptive arithmetic coding. The Transformer predicts probability distributions for the next DNA token, and the arithmetic coder uses these probabilities to compress the sequence. During decompression, the same trained model and decoded context are used to reconstruct the cleaned DNA sequence.

## Programming Language and Technologies

- **Programming language:** Python
- **Deep learning framework:** PyTorch
- **Numerical processing:** NumPy
- **Training plots:** Matplotlib
- **Hardware environment:** Kaggle Notebook with **2 × NVIDIA T4 GPU units** in the reported experiments

The code selects CUDA when available. It does not explicitly configure multi-GPU training, so both T4 GPUs are not necessarily used simultaneously by the supplied implementation.

## Setup

### 1. Prepare the environment

Run the project in a Kaggle Notebook with GPU acceleration enabled. The reported research setup used two NVIDIA T4 GPU units.

The implementation uses Python and the following libraries:

- PyTorch
- NumPy
- Matplotlib

Install NumPy and Matplotlib if they are not already available:

```bash
pip install numpy matplotlib
```

Install a compatible PyTorch build by following the instructions at https://pytorch.org/get-started/locally/.

### 2. Configure the datasets

The supplied code uses these Kaggle input paths:

```python
TRAIN_DIR = "/kaggle/input/datasets/arkadipmaths/transformer-train-small"
TEST_DIR  = "/kaggle/input/datasets/arkadipmaths/transformer-test-small"
OUT_DIR   = "/kaggle/working"
```

Attach the corresponding training and testing datasets to the Kaggle Notebook. If your datasets are stored elsewhere, update `TRAIN_DIR` and `TEST_DIR`. The output directory is created automatically.

The input directories should contain FASTA-like sequence files. The code ignores header lines beginning with `>` and retains only uppercase `A`, `C`, `G`, and `T` bases.

### 3. Check GPU availability

```python
import torch

print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(i))
```

## How to Use

The supplied code runs training and testing as one pipeline.

1. Configure `TRAIN_DIR`, `TEST_DIR`, and `OUT_DIR`.
2. Run the code in the Kaggle Notebook.
3. The program cleans and tokenizes the training sequences, creates training and validation blocks, and trains the causal Transformer.
4. The checkpoint with the lowest validation loss is saved as `best_model.pt`.
5. After training, the code processes each test file: it cleans and tokenizes the DNA, compresses the tokens into a `.dnacomp` file, decompresses that file, and checks the reconstructed cleaned sequence.
6. Outputs are written to `/kaggle/working`. The training metrics plot is saved as `training_plots.png`, compressed files use the `.dnacomp` extension, and reconstructed cleaned sequences use `_reconstructed.txt`.

**Important:** Decompression requires the same trained model weights and compatible configuration used during compression. The supplied implementation reconstructs the cleaned `ACGT` sequence; it does not preserve FASTA headers, original line formatting, or characters removed during cleaning.

## Training Parameters

The following values are taken from the supplied code.

| Parameter | Value |
|---|---:|
| k-mer size (`K`) | `4` |
| Vocabulary size (`KMER_VOCAB`) | `256` (`4^4`) |
| Context/block size (`BLOCK_SIZE`) | `320` tokens |
| Embedding dimension (`EMBED_DIM`) | `256` |
| Attention heads (`N_HEADS`) | `8` |
| Transformer layers (`N_LAYERS`) | `6` |
| Feed-forward dimension (`FF_DIM`) | `1024` |
| Dropout (`DROPOUT`) | `0.1` |
| Epochs (`EPOCHS`) | `30` |
| Batch size (`BATCH_SIZE`) | `32` |
| Initial learning rate (`LR`) | `2e-4` |
| Optimizer | AdamW |
| Weight decay | `1e-4` |
| Learning-rate scheduler | Cosine annealing |
| Minimum learning rate | `LR / 20` (`1e-5`) |
| Gradient clipping norm | `1.0` |
| Training/validation split | `90% / 10%` |
| Block shuffle seed | `42` |
| Mixed-precision training | Enabled using `autocast()` and `GradScaler()` |
| Positional encoding | Sinusoidal |
| Arithmetic coder precision | `32` bits |
| Device selection | CUDA if available; otherwise CPU |

The code sets `EPOCHS = 30`. The accompanying paper reports that its main base-model results used a 10-epoch training run, so results from the supplied code may differ depending on the training configuration and dataset.
