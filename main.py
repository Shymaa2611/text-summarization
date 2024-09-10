from torch.utils.data import Dataset
from transformers import GPT2LMHeadModel
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from dataset import TextSummaryDataset
from training import train
import torch

def main():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    texts_dir = 'data/text'
    summaries_dir = 'data/summary'
    dataset = TextSummaryDataset(texts_dir, summaries_dir, tokenizer)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(train_loader,tokenizer,model,device)


if __name__=="__main__":
    main()