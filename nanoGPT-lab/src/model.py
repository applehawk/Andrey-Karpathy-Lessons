# Original source: https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
# Local reference: external/ng-video-lecture/gpt.py

import torch
import torch.nn as nn
from torch.nn import functional as F

# ==============================================================================
# 1. HYPERPARAMETERS (Global Configuration)
# ==============================================================================
# These parameters define the "width" and "depth" of the network, which directly
# affects memory usage and compression potential.
# $C = n\_embd$, $L = n\_layer$, $H = n\_head$

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

# Vocab size will be set after loading data in utils.py
# We declare it here, but the value comes from utils
vocab_size = None

print(f"Using device: {device}")

# ==============================================================================
# 2. MODEL COMPONENTS (Model Architecture)
# ==============================================================================


class Head(nn.Module):
    r"""
    One head of Self-Attention.
    Computes Query (Q), Key (K), and Value (V).
    Formula: $Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$
    """

    def __init__(self, head_size):
        super().__init__()
        # Linear projections without bias (bias=False) — ideal candidates for quantization
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Mask for Causal Attention (to prevent looking into the future)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # Compute affinity (token similarity)
        # Scaled Dot-Product: $score = \frac{Q \cdot K^T}{\sqrt{C}}$
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """
    Multiple heads of self-attention running in parallel.
    Allows the model to simultaneously focus on different aspects of context.
    """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate outputs of all heads along the channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Final projection back to n_embd
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """
    Fully-connected network (MLP).
    "Individual thinking" of each token after information exchange.
    Formula: $FFN(x) = ReLU(xW_1 + b_1)W_2 + b_2$
    """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            # Expand by 4x (GPT standard)
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            # Compress back
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    Transformer Block: Communication (Attention) + Computation (FFN).
    Uses Residual Connections and LayerNorm before layers (Pre-norm).
    $x = x + SA(LN(x))$
    $x = x + FFN(LN(x))$
    """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# ==============================================================================
# 3. MAIN MODEL (The GPT Model)
# ==============================================================================


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # Token Embedding Table (V -> C)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # Position Embedding Table (T -> C)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Stack of L Transformer blocks
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        # Final normalization
        self.ln_f = nn.LayerNorm(n_embd)
        # Output linear layer to vocabulary size
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Custom weight initialization (Mean=0, Std=0.02)
        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        # 1. Embeddings (Meaning + Position)
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)

        # 2. Processing through attention blocks
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)

        # 3. Computing logits (character probabilities)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            # Flatten for Cross Entropy function
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Iterative text generation token by token"""
        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond = idx[:, -block_size:]
            # Get predictions
            logits, loss = self.forward(idx_cond)
            # Take only the last time step
            logits = logits[:, -1, :]  # (B, C)
            # Softmax converts logits to probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append to current sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx