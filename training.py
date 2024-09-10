import torch
import torch.nn as nn
import torch.optim as optim

def train(train_loader, tokenizer, model, device):
    print("Start")
    
    # Move model to GPU
    model.to(device)
    
    # Ensure the tokenizer has the pad token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))  

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    model.train()

    for epoch in range(10):  
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device).long()
            attention_mask = batch['attention_mask'].to(device).long()
            target_ids = batch['target_ids'].to(device).long()

            try:
                outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()

            except Exception as e:
                print(f'Error during forward pass or backward pass: {e}')
                print(f'input_ids shape: {input_ids.shape}, target_ids shape: {target_ids.shape}')
                print(f'input_ids max: {input_ids.max().item()}, target_ids max: {target_ids.max().item()}')

        print(f'Epoch: {epoch}, Loss: {epoch_loss / len(train_loader)}')
        model.save_pretrained('./checkpoint')
        tokenizer.save_pretrained('./checkpoint')


