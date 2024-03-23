import requests
import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM

# Fetch text from the API
key = 'Python:418'
url = f'http://127.0.0.1:8000/api/v1/load/{key}'


# NB: Set up to process one example at a time - less efficient than batch processing - 
# could create issues with stability - less regularization than with batches, etc.

# NB: need to receive both the prompt and the labels:
response = requests.get(url)
if response.status_code == 200:
    data = response.json()
    text = data.get('text', 'No text available for the given key.')
    labels_text = data.get('labels', 'No labels available for the given key.')  # Assuming API now also returns labels
else:
    print(f'Failed to fetch the data for key "{key}". HTTP Status Code: {response.status_code}. Response: {response.text}')
    text, labels_text = '', ''  # Use empty strings if the fetch fails


# Assuming successful fetch, use the fetched text as the prompt (along with the labels)
training_data = [(text, labels_text)] if text and labels_text else [("No text available for the given key.", "")]

#petals-team/StableBeluga2
# Initialize tokenizer and model
model_name = "deepseek-ai/deepseek-coder-7b-instruct"
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
    for text, labels_text in training_data:
        # Tokenize input text and labels
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to('cuda')
        labels = tokenizer(labels_text, return_tensors="pt", padding=True, truncation=True)["input_ids"].to('cuda')
        
        # Perform a forward pass to get adapted logits from the custom model
        adapted_logits = custom_model(**inputs) # Since outputs are the adapted logits directly from the custom adapter
        
        # Compute the loss using the labels. Assuming a shift in labels for language modeling tasks -
        # i.e. remove the last logit because there is no next token to predict after the last token in the sequence.
        # The contiguous() call - ensures the tensor is stored in a continuous block of memory, sometimes necessary for .view().
        # may need to adjust according to  model's output structure...
        shift_logits = adapted_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        print(f"Epoch {epoch}, Loss: {loss.item():.3f}")

# Save the fine-tuned model
torch.save(custom_model.state_dict(), "your_custom_model_path_here.pth")
