# Inference utilities for GPT model
# Extended generation functions beyond the basic model

import torch
from torch.nn import functional as F

from src.utils import block_size, device, decode


@torch.no_grad()
def generate_stream(model, idx, max_new_tokens):
    """
    Генерация текста в виде потока (yield) для streaming inference.
    
    Args:
        model: GPTLanguageModel instance
        idx: Starting context tensor (B, T)
        max_new_tokens: Number of tokens to generate
        
    Yields:
        int: Next generated token ID
    """
    for _ in range(max_new_tokens):
        # Обрезаем контекст
        idx_cond = idx[:, -block_size:]
        # Получаем логиты
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        # Сэмплируем следующий токен
        idx_next = torch.multinomial(probs, num_samples=1)
        # Обновляем контекст
        idx = torch.cat((idx, idx_next), dim=1)
        # Возвращаем токен (предполагаем batch_size=1)
        yield idx_next.item()


@torch.no_grad()
def generate_text(model, prompt="", max_new_tokens=100, temperature=1.0):
    """
    Генерация текста с заданным промптом.
    
    Args:
        model: GPTLanguageModel instance
        prompt: Starting text prompt (will be encoded)
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        
    Returns:
        str: Generated text
    """
    from src.utils import encode
    
    # Encode prompt or start with empty context
    if prompt:
        context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    # Generate
    generated = model.generate(context, max_new_tokens)
    
    # Decode and return
    return decode(generated[0].tolist())


@torch.no_grad()
def generate_text_stream(model, prompt="", max_new_tokens=100):
    """
    Генерация текста с промптом в режиме streaming.
    
    Args:
        model: GPTLanguageModel instance
        prompt: Starting text prompt
        max_new_tokens: Number of tokens to generate
        
    Yields:
        str: Next generated character
    """
    from src.utils import encode
    
    # Encode prompt or start with empty context
    if prompt:
        context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    # Stream generation
    for token_id in generate_stream(model, context, max_new_tokens):
        yield decode([token_id])
