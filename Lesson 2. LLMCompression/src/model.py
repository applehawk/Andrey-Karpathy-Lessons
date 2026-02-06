# Original source: https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
# Local reference: external/ng-video-lecture/gpt.py

import torch
import torch.nn as nn
from torch.nn import functional as F

# ==============================================================================
# 1. ГИПЕРПАРАМЕТРЫ (Global Configuration)
# ==============================================================================
# Эти параметры определяют "ширину" и "глубину" сети, что напрямую влияет на
# объем памяти и потенциал сжатия.
# $C = n\_embd$, $L = n\_layer$, $H = n\_head$

batch_size = 64     # Количество независимых последовательностей в одном батче (B)
block_size = 256    # Максимальный контекст предсказания (T)
max_iters = 5000    # Всего итераций обучения
eval_interval = 500 # Как часто замерять лосс
learning_rate = 3e-4
# Авто-выбор устройства: MPS (Mac), CUDA (Nvidia) или CPU
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
eval_iters = 200    # Количество батчей для оценки лосса
n_embd = 384        # Размерность эмбеддинга (C)
n_head = 6          # Количество голов внимания (h)
n_layer = 6         # Количество слоев Transformer (L)
dropout = 0.2       # Вероятность зануления активаций для регуляризации

print(f"Using device: {device}")

# ==============================================================================
# 2. ПОДГОТОВКА ДАННЫХ (Tokenizer & Data Loading)
# ==============================================================================
torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Символьный токенизатор (Vocabulary size V)
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Разделение на Train/Val (90/10)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ==============================================================================
# 3. КОМПОНЕНТЫ МОДЕЛИ (Model Architecture)
# ==============================================================================

class Head(nn.Module):
    """ 
    Одна голова Self-Attention.
    Здесь вычисляются Query (Q), Key (K) и Value (V).
    Формула: $Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$
    """

    def __init__(self, head_size):
        super().__init__()
        # Линейные проекции без смещения (bias=False) — идеальные кандидаты для квантования
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Маска для Causal Attention (чтобы не смотреть в будущее)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # Вычисление аффинити (схожести токенов)
        # Scaled Dot-Product: $score = \frac{Q \cdot K^T}{\sqrt{C}}$
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ 
    Параллельный запуск нескольких голов внимания.
    Позволяет модели одновременно фокусироваться на разных аспектax контекста.
    """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Склеиваем выходы всех голов по размерности каналов
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Итоговая проекция обратно в n_embd
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ 
    Полносвязная сеть (MLP).
    "Индивидуальное размышление" каждого токена после обмена информацией.
    Формула: $FFN(x) = ReLU(xW_1 + b_1)W_2 + b_2$
    """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            # Расширение в 4 раза (стандарт GPT)
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            # Сжатие обратно
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ 
    Transformer Block: Communication (Attention) + Computation (FFN).
    Использует Residual Connections и LayerNorm перед слоями (Pre-norm).
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
# 4. ГЛАВНАЯ МОДЕЛЬ (The GPT Model)
# ==============================================================================

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # Token Embedding Table (V -> C)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # Position Embedding Table (T -> C)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Стек из L блоков Transformer
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        # Финальная нормировка
        self.ln_f = nn.LayerNorm(n_embd)
        # Выходной линейный слой до размера словаря
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Кастомная инициализация весов (Mean=0, Std=0.02)
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
        # 1. Эмбеддинги (Смысл + Позиция)
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        
        # 2. Обработка блоками внимания
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        
        # 3. Вычисление логитов (вероятностей символов)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            # Превращаем в плоский список для функции Cross Entropy
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """ Итеративная генерация текста токен за токеном """
        for _ in range(max_new_tokens):
            # Обрезаем контекст до block_size
            idx_cond = idx[:, -block_size:]
            # Получаем предсказания
            logits, loss = self(idx_cond)
            # Берем только последний временной шаг
            logits = logits[:, -1, :] # (B, C)
            # Softmax превращает логиты в вероятности
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Сэмплируем из распределения
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Добавляем к текущей последовательности
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
