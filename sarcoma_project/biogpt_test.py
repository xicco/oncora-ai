from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "microsoft/BioGPT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 更具体、清晰的问题提示，后面加上空格，模型才能继续生成内容
prompt = "Question: What gene mutations are related to sarcoma? Answer:"

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_length=100,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.7,
    num_return_sequences=1
)

# 解码模型的回答，跳过特殊符号
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n🧠 BioGPT answer:")
print(generated_text.replace(prompt, ""))