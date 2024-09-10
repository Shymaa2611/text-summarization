from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

def Quantization_Config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    return bnb_config

def load_model_in_4bit_Quantization(model_name):
    bnb_config = Quantization_Config()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        quantization_config=bnb_config,
        load_in_4bit=True  
    )
    return tokenizer, model
