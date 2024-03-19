import requests
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM

# Fetch text from the API
key = 'Python:418'
url = f'http://127.0.0.1:8000/api/v1/load/{key}'

response = requests.get(url)
if response.status_code == 200:
    data = response.json()
    text = data.get('text', 'No text available for the given key.')
else:
    print(f'Failed to fetch the text for key "{key}". HTTP Status Code: {response.status_code}. Response: {response.text}')
    text = ''  # Use an empty string if the fetch fails

# Assuming successful fetch, use the fetched text as the prompt
prompts = [text] if text else ["No text available for the given key."]

# Initialize tokenizer and model
model_name = "petals-team/StableBeluga2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
INITIAL_PEERS = ['/ip4/45.79.153.218/tcp/31337/p2p/QmXfANcrDYnt5LTXKwtBP5nsTMLQdgxJHbK3L1hZdFN8km']
model = AutoDistributedModelForCausalLM.from_pretrained(model_name, initial_peers=INITIAL_PEERS).cuda()

# Custom adapter class for fine-tuning
class CustomAdapter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.base_model = model
        vocab_size = model.config.vocab_size
        self.adapter = nn.Sequential(
            nn.Linear(vocab_size, 32),
            nn.ReLU(),
            nn.Linear(32, vocab_size)
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        adapted_logits = self.adapter(logits)
        return adapted_logits

custom_model = CustomAdapter(model).cuda()

# Initialize optimizer and scheduler
optimizer = AdamW(custom_model.parameters(), lr=1e-3)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

# Number of epochs (iterations) for fine-tuning
num_epochs = 20

# Optimization loop for fine-tuning
for epoch in range(num_epochs):
    for prompt in prompts:
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
        
        # Forward pass and compute the loss (this example uses a dummy loss)
        outputs = custom_model(input_ids=prompt_ids)
        loss = torch.tensor(0.0).cuda()  # Placeholder for actual loss computation
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        print(f"Epoch {epoch}, Loss: {loss.item():.3f}")

# Save the fine-tuned model
torch.save(custom_model.state_dict(), "your_custom_model_path_here.pth")
