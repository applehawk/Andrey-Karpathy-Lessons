# Modified based on: https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
# Local reference: external/ng-video-lecture/gpt.py

import torch
from src.model import GPTLanguageModel, device, get_batch, estimate_loss
import time

# Hyperparameters for quick training
max_iters = 2000 # Enough to see some learning
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200

# Initialize model
model = GPTLanguageModel().to(device)
print(f"Training on: {device}")
print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("Starting training...")
start_time = time.time()

for iter in range(max_iters):

    # Every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample a batch of data
    xb, yb = get_batch('train')

    # Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

end_time = time.time()
print(f"Training finished in {end_time - start_time:.2f} seconds")

# Save the model
checkpoint_path = "model_ckpt.pt"
torch.save(model.state_dict(), checkpoint_path)
print(f"Model saved to {checkpoint_path}")

# Quick generation test
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("--- Sample Output ---")
print(model.generate(context, max_new_tokens=100)[0].tolist())
