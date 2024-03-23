import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer
import requests
from petals import AutoDistributedModelForCausalLM  # Assuming this is a valid import based on your script

FETCH_URL = "http://localhost:8000/api/v1/load"

class FastAPIDataset(Dataset):
    def __init__(self, fetch_url, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = self.fetch_data(fetch_url)
        self.max_length = max_length

    def fetch_data(self, url):
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()['data']
            texts = [item['value'] for item in data]
            return texts
        else:
            raise Exception(f"Failed to fetch data: HTTP status code {response.status_code}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, truncation=True, padding="max_length")
        input_ids = encoding['input_ids'].squeeze()  # Remove the batch dimension
        attention_mask = encoding['attention_mask'].squeeze()
        return input_ids, attention_mask

model_name = "deepseek-ai/deepseek-coder-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
INITIAL_PEERS = ['/ip4/45.79.153.218/tcp/31337/p2p/QmXfANcrDYnt5LTXKwtBP5nsTMLQdgxJHbK3L1hZdFN8km']
model = AutoDistributedModelForCausalLM.from_pretrained(model_name, initial_peers=INITIAL_PEERS).cuda()

# Ensure all model parameters require gradients
for param in model.parameters():
    param.requires_grad = True

dataset = FastAPIDataset(FETCH_URL, tokenizer)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 2
for epoch in range(num_epochs):
    model.train()  # Ensure the model is in training mode
    for input_ids, attention_mask in dataloader:
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass: Compute predicted y by passing x to the model
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss

        # Backward pass: Compute gradient of the loss with respect to model parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Save the model
model.save_pretrained("./")
