from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "stanford-crfm/BioMedLM"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

raw_case = input("Enter patient case description:\n")

prompt = (
    "You are a trusted oncology AI assistant with up-to-date knowledge on rare cancers, "
    "especially sarcomas.\n"
    "Your job is to analyze clinical case descriptions and provide helpful, medically accurate "
    "suggestions based on prior research and treatment guidelines.\n\n"
    f"Patient Case: {raw_case.strip()}\n\n"
    "What would be your clinical insight or recommended next step?\n"
)

model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **model_inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
generated_text = response[len(prompt):].strip()

print("\n🧠 BioMedLM's Response:\n")
print(generated_text)

