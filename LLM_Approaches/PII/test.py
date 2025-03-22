import transformers
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

chunks = []
with open("forum_one.txt", "r") as file:
    
    while True:
        count = 0
        chunk = ""
        line = file.readline()
        while line:
            chunk += line + "\n"
            line = file.readline()
            count += 1
            if count == 20:
                break
        chunks.append(chunk)
        if not line:
            break
        

with open("instructions.txt", "r") as file:
    instruction = file.read()


prompt = "The United States of America is a country located in North America. It was founded in 1776 by George Washington. My name is Cooper"

messages = [
    {"role": "system", "content": instruction},
    {"role": "user", "content": prompt}
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)

print(outputs[0]["generated_text"][-1])
