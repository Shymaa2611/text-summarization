import torch.nn as nn
import torch.optim as optim

def train(train_loader,tokenizer,model):
    print("Start")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss= nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    model.train()
    for epoch in range(100):  
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].long()
            attention_mask = batch['attention_mask'].long()
            target_ids = batch['target_ids'].long()
            #print(f'input_ids shape: {input_ids.shape}')
            #print(f'target_ids shape: {target_ids.shape}')
            #if input_ids.shape != target_ids.shape:
            #    raise ValueError(f'Mismatch in shapes: input_ids {input_ids.shape}, target_ids {target_ids.shape}')
            try:
                outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                print(f'Epoch: {epoch}, Loss: {loss.item()}')
            except Exception as e:
                print(f'Error during forward pass or backward pass: {e}')
    
    model.save_pretrained('./checkpoint')
    tokenizer.save_pretrained('./checkpoint')

if __name__ == "__main__":
    train()
