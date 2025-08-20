from transformers import AutoTokenizer, AutoModelForCausalLM # type: ignore

model_id = "mistralai/Mistral-7B-Instruct-v0.3"
local_dir = "./mistral-7b-instruct-local"

# Télécharger et sauvegarder le tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(local_dir)

# Télécharger et sauvegarder le modèle
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
model.save_pretrained(local_dir)

