from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import re

# Load pre-trained model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Function to clean output
def clean_output(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return ' '.join(sentences[:-1]) if len(sentences) > 1 else text

# Function to generate text
def generate_text(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=150,                     # Increased for longer output
        num_return_sequences=1,
        no_repeat_ngram_size=3,             # Improved to reduce repeated phrases
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        do_sample=True,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id  # Prevents warning
    )

    # Decode and return the cleaned text
    raw_output = tokenizer.decode(output[0], skip_special_tokens=True)
    return clean_output(raw_output)

# Main
if __name__ == "__main__":
    user_input = input("Enter a topic or prompt: ")
    result = generate_text(user_input)
    print("\nGenerated Text:\n")
    print(result)
