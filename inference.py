from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def summarize_text(text, tokenizer, model, max_length=512, summary_length=200):
    inputs = tokenizer.encode(text, return_tensors='pt', max_length=max_length, truncation=True)
    with torch.no_grad():
        summary_ids = model.generate(inputs, max_length=summary_length, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def main():
    model = GPT2LMHeadModel.from_pretrained('./checkpoint')
    tokenizer = GPT2Tokenizer.from_pretrained('./checkpoint')
    model.eval()  

    text = """
    It can be a long piece of text that you want to summarize.
    """
    
    summary = summarize_text(text, tokenizer, model)
    print(f"Summary: {summary}")

if __name__ == "__main__":
    main()
