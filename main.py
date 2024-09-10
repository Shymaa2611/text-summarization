from torch.utils.data import Dataset
from transformers import GPT2LMHeadModel
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from dataset import TextSummaryDataset
from training import train
from model import load_model_in_4bit_Quantization

def main():
    model_name='gpt2'
    tokenizer,model=load_model_in_4bit_Quantization(model_name)
    texts_dir = 'data/text'
    summaries_dir = 'data/summary'
    dataset = TextSummaryDataset(texts_dir, summaries_dir, tokenizer)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    train(train_loader,tokenizer,model)


if __name__=="__main__":
    main()