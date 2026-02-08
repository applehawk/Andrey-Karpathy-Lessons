import torch
from src.model import GPTLanguageModel, device
from src.utils import decode, encode


def chat():
    # Initialize model
    model = GPTLanguageModel().to(device)

    # Load weights
    try:
        model.load_state_dict(torch.load("model_ckpt.pt", map_location=device))
        print("✅ Model loaded successfully.")
    except FileNotFoundError:
        print("❌ Error: model_ckpt.pt not found. Run 'make train' first.")
        return

    model.eval()

    print("\n" + "=" * 50)
    print("WELCOME TO SHAKESPEARE-GPT CHAT")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 50 + "\n")

    while True:
        user_input = input("Enter your prompt (English): ")
        if user_input.lower() in ["exit", "quit"]:
            break

        if not user_input:
            continue

        try:
            # Encode user input
            context = torch.tensor(
                encode(user_input), dtype=torch.long, device=device
            ).unsqueeze(0)

            print("\nGenerating...")
            # Generate 200 new tokens
            generated_indices = model.generate(context, max_new_tokens=200)[0].tolist()

            print("\n" + "-" * 30)
            print(decode(generated_indices))
            print("-" * 30 + "\n")

        except KeyError as e:
            print(
                f"❌ Error: Your prompt contains a character not in the vocabulary: {e}"
            )
            print("Please use standard English characters and punctuation.")


if __name__ == "__main__":
    chat()
