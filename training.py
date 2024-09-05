import os
from torch.utils.data import Dataset
from transformers import GPT2LMHeadModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from dataset import TextSummaryDataset
def train():
    print("Start")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    texts_dir = 'data/text'
    summaries_dir = 'data/summary'
    dataset = TextSummaryDataset(texts_dir, summaries_dir, tokenizer)

    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)  
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    model.train()
    for epoch in range(3):  # Number of epochs
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_ids = batch['target_ids'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            print(f'Epoch: {epoch}, Loss: {loss.item()}')
    
    model.save_pretrained('./fine_tuned_model')
    tokenizer.save_pretrained('./fine_tuned_model')

if __name__ == "__main__":
    train()
