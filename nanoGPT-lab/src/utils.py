# Original source: https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
# Utility functions for data loading and model evaluation

import torch

# ==============================================================================
# HYPERPARAMETERS (Global Configuration)
# ==============================================================================
batch_size = 64  # Number of independent sequences in one batch (B)
block_size = 256  # Maximum context length for predictions (T)
max_iters = 5000  # Total training iterations
eval_interval = 500  # How often to measure loss
learning_rate = 3e-4
# Auto-select device: MPS (Mac), CUDA (Nvidia), or CPU
device = (
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
eval_iters = 200  # Number of batches for loss evaluation
n_embd = 384  # Embedding dimension (C)
n_head = 6  # Number of attention heads (h)
n_layer = 6  # Number of Transformer layers (L)
dropout = 0.2  # Probability of zeroing activations for regularization

print(f"Using device: {device}")

# ==============================================================================
# DATA PREPARATION (Tokenizer & Data Loading)
# ==============================================================================
torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
import os
input_txt_path = os.path.join(os.path.dirname(__file__), "input.txt")
with open(input_txt_path, "r", encoding="utf-8") as f:
    text = f.read()

# Character-level tokenizer (Vocabulary size V)
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Set vocab_size in model module (it's a model hyperparameter)
import src.model
src.model.vocab_size = vocab_size

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string

# Train/Val split (90/10)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
