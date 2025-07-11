from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "stanford-crfm/BioMedLM"

device = "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float32, 
    offload_folder='./offload'
    ).to(device)

raw_case = input("Enter patient case description:\n")

prompt = (
    "You are a trusted oncology AI assistant with up-to-date knowledge on rare cancers, "
    "especially sarcomas.\n"
    "Your job is to analyze clinical case descriptions and provide helpful, medically accurate "
    "suggestions based on prior research and treatment guidelines.\n\n"
    f"Patient Case: {raw_case.strip()}\n\n"
    "What would be your clinical insight or recommended next step?\n"
)

model_inputs = tokenizer(prompt, return_tensors="pt").to(device)

print("Generating response...")

with torch.no_grad():
    outputs = model.generate(
        **model_inputs,
        max_new_tokens=100,
        temperature=0.3,
        top_p=0.8,
        num_beams=3
        #do_sample=True
    )

print(outputs)

generated_ids = outputs[0][len(model_inputs["input_ids"][0]):]  

generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

print("\n🧠 BioMedLM's Response:\n")
print(generated_text)

