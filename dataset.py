from torch.utils.data import Dataset
import os


class TextSummaryDataset(Dataset):
    def __init__(self, texts_dir, summaries_dir, tokenizer, max_length=512):
        self.texts_dir = texts_dir
        self.summaries_dir = summaries_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_files = sorted(os.listdir(texts_dir))
        self.summary_files = sorted(os.listdir(summaries_dir))
    
    def __len__(self):
        return len(self.text_files)
    
    def __getitem__(self, idx):
        text_path = os.path.join(self.texts_dir, self.text_files[idx])
        summary_path = os.path.join(self.summaries_dir, self.summary_files[idx])

        with open(text_path, 'r') as file:
            text = file.read()
        with open(summary_path, 'r') as file:
            summary = file.read()

        inputs = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        targets = self.tokenizer.encode_plus(
            summary,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].squeeze(0)  
        attention_mask = inputs['attention_mask'].squeeze(0)  
        target_ids = targets['input_ids'].squeeze(0)  
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_ids': target_ids
        }
