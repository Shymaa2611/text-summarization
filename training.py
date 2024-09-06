import torch.nn as nn
import torch.optim as optim

def train(train_loader, tokenizer, model):
    print("Start")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))  
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    model.train()

    for epoch in range(1):  
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].long()
            attention_mask = batch['attention_mask'].long()
            target_ids = batch['target_ids'].long()

            try:
                outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                print(f'Epoch: {epoch}, Loss: {loss.item()}')
            except Exception as e:
                print(f'Error during forward pass or backward pass: {e}')
                print(f'input_ids shape: {input_ids.shape}, target_ids shape: {target_ids.shape}')
                print(f'input_ids max: {input_ids.max().item()}, target_ids max: {target_ids.max().item()}')
    
            model.save_pretrained('./checkpoint')
            tokenizer.save_pretrained('./checkpoint')
