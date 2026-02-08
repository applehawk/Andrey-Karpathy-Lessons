import torch
from src.model import GPTLanguageModel, device, decode


def generate_text(max_tokens=500):
    # Initialize model
    model = GPTLanguageModel().to(device)

    # Load weights
    try:
        model.load_state_dict(torch.load("model_ckpt.pt", map_location=device))
        print("✅ Loaded weights from model_ckpt.pt")
    except FileNotFoundError:
        print("❌ Error: model_ckpt.pt not found. Please run 'make train' first.")
        return

    model.eval()

    # Start generation from a single zero token (usually newline or space in Shakespere)
    context = torch.zeros((1, 1), dtype=torch.long, device=device)

    print("\n" + "=" * 50)
    print("GPT GENERATION OUTPUT:")
    print("=" * 50 + "\n")

    # Generate and decode
    generated_indices = model.generate(context, max_new_tokens=max_tokens)[0].tolist()
    print(decode(generated_indices))
    print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    import sys

    tokens = 500
    if len(sys.argv) > 1:
        tokens = int(sys.argv[1])
    generate_text(tokens)
