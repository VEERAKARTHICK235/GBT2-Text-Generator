!pip install transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
# Load pre-trained model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# Prompt
user_prompt = input("Enter a topic or prompt: ")
input_ids = tokenizer.encode(user_prompt, return_tensors="pt")
# Generate
output = model.generate(
    input_ids,
    max_length=100,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    top_k=50,
    top_p=0.95,
    temperature=0.7,
    do_sample=True,
    early_stopping=True
)
# Decode and print
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("\nGenerated Text:\n")
print(generated_text)