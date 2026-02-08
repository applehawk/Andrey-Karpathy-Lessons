# nanoGPT-lab

Shared resources for LLM Compression lessons based on Andrey Karpathy's nanoGPT.

## Structure

```
nanoGPT-lab/
├── src/
│   ├── __init__.py      # Package exports
│   ├── model.py         # GPT model architecture (pure Karpathy style)
│   ├── utils.py         # Data loading and training utilities
│   └── inference.py     # Extended inference functions (streaming, etc.)
├── input.txt            # Training data (Shakespeare)
├── model_ckpt.pt        # Pre-trained checkpoint
├── train.py             # Training script
├── generate.py          # Generation script
├── chat.py              # Interactive chat
└── analyze.py           # Model analysis tools
```

## Module Organization

### `src/model.py`
- **Pure model architecture** - classes only, following Karpathy's original structure
- Hyperparameters defined at module level
- Classes: `GPTLanguageModel`, `Block`, `MultiHeadAttention`, `Head`, `FeedFoward`
- Basic `generate()` method

### `src/utils.py`
- Data loading: `get_batch()`, tokenizer (`encode`/`decode`)
- Training utilities: `estimate_loss()`
- Exports: `vocab_size`, `train_data`, `val_data`

### `src/inference.py`
- Extended generation functions
- `generate_stream()` - streaming token generation
- `generate_text()` - text generation with prompts
- `generate_text_stream()` - streaming text generation

## Usage in Notebooks

```python
# Import model and utilities
from src.model import GPTLanguageModel, Block, MultiHeadAttention, Head, FeedFoward
from src.utils import device, encode, decode, get_batch, estimate_loss
from src.inference import generate_text, generate_text_stream

# Or import everything
from src import *

# Create and use model
model = GPTLanguageModel().to(device)
```

## Design Philosophy

- **`model.py`**: Keep it close to Karpathy's original - architecture only
- **`utils.py`**: Data and training helpers
- **`inference.py`**: Extended features for production use
- **Separation of concerns**: Model definition vs. utilities vs. inference
