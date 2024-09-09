# GPT-2 Fine-Tuning for Text Summarization
- This project demonstrates how to fine-tune GPT-2 for the task of text summarization. 
  Fine-tuning GPT-2 allows you to adapt a pre-trained model to generate concise and coherent summaries from text.

## Overview

1. **Fine-Tuning GPT-2**: Train a GPT-2 model on a summarization dataset to customize it for generating summaries.
2. **Evaluation**: Evaluate the performance of the fine-tuned model on a validation dataset.

## Requirements

- Python 3.7 or higher
- Transformers library
- PyTorch
## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Shymaa2611/text-summarization.git
   cd https://github.com/Shymaa2611/text-summarization.git
   ```
2. Install the required packages:
   pip install -r requirements.txt

## Dataset

### Overview

The dataset used for fine-tuning GPT-2 consists of two separate folders: one for text documents and one for summaries. This structure allows you to train the model on text-summary pairs where each text file corresponds to a summary file.

### Folder Structure

- **`text/`**: Contains text documents to be summarized.
- **`summary/`**: Contains summary documents corresponding to each text document.

Each text file in the `text/` folder has a corresponding summary file in the `summary/` folder. The filenames in both folders should match to ensure proper pairing. For example, a text file named `document1.txt` should have a corresponding summary file named `document1_summary.txt`.

### Preprocessing 
  - run Python dataset.py file 

## Inference
 ``Python
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

   
   ```
## Checkpoint
