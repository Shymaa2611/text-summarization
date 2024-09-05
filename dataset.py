import os
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

class TextSummaryDataset(Dataset):
    def __init__(self, texts_dir, summaries_dir, tokenizer, max_input_length=1024, max_output_length=150):
        self.texts_dir = texts_dir
        self.summaries_dir = summaries_dir
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.file_names = sorted(os.listdir(texts_dir))
    
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        text_file = os.path.join(self.texts_dir, self.file_names[idx])
        summary_file = os.path.join(self.summaries_dir, self.file_names[idx])
        
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary = f.read()
        
        inputs = self.tokenizer.encode_plus(
            text, 
            max_length=self.max_input_length, 
            padding="max_length", 
            truncation=True,
            return_tensors="pt"
        )
        targets = self.tokenizer.encode_plus(
            summary, 
            max_length=self.max_output_length, 
            padding="max_length", 
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        target_ids = targets['input_ids'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_ids': target_ids
        }
