import torch
from src import model


def get_model_size(mdl):
    param_size = 0
    for param in mdl.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in mdl.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


# 1. Загружаем базовую модель (FP32)
gpt = model.GPTLanguageModel().to(model.device)
size_fp32 = get_model_size(gpt)
print(f"Model size (FP32): {size_fp32:.2f} MB")

# 2. Анализ распределения весов
# Посмотрим на веса первого слоя Attention
first_block_key = gpt.blocks[0].sa.heads[0].key.weight.data.cpu().flatten()
print(
    f"Weight stats: mean={first_block_key.mean():.4f}, std={first_block_key.std():.4f}"
)

# 3. Naive Quantization (FP16)
gpt_fp16 = model.GPTLanguageModel().to(model.device).half()
size_fp16 = get_model_size(gpt_fp16)
print(f"Model size (FP16): {size_fp16:.2f} MB")
print(f"Compression ratio: {size_fp32 / size_fp16:.2f}x")

# 4. Проверка на выбросы (Outliers)
# В настоящем сжатии мы ищем значения, которые сильно выходят за пределы std
outliers = torch.sum(torch.abs(first_block_key) > 3 * first_block_key.std())
print(
    f"Potential outliers in first key layer (>3std): {outliers.item()} ({outliers.item() / len(first_block_key) * 100:.2f}%)"
)
