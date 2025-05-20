from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import traceback

model_name_or_path = "devam-sheth-bits/enhanced-sleep-ai"

print(f"Loading model/tokenizer from Hub: {model_name_or_path}")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
print("Model/tokenizer loaded.")
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
print(f"Using device: {device}")

# --- Inference Function ---
def query_sleep_ai_multi_turn(conversation_history, max_new_tokens=150):
    prompt = ""
    for turn in conversation_history:
        if isinstance(turn, dict) and 'role' in turn and 'content' in turn: prompt += f"{turn['role']}: {turn['content']}{tokenizer.eos_token}"
        else: print(f"Warn: Skipping malformed turn: {turn}")
    prompt += "assistant:"
    inputs = tokenizer(prompt, return_tensors="pt"); input_ids = inputs["input_ids"].to(device); attention_mask = inputs["attention_mask"].to(device)
    if input_ids.shape[-1] == 0: print("Error: Encoded empty."); return None
    if input_ids.shape[-1] > tokenizer.model_max_length: print(f"Warn: Input length {input_ids.shape[-1]} > max {tokenizer.model_max_length}.")
    with torch.no_grad():
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, do_sample=True, temperature=0.7, top_p=0.9, num_return_sequences=1)
    generated_ids = output[0][input_ids.shape[-1]:]; reply = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if not reply: print("Warn: Empty reply."); return "I'm unsure how to reply." if output[0].shape[-1] == input_ids.shape[-1] else None
    return reply

if __name__ == "__main__":
    print("\n--- Running Inference Test ---")
    test_history = [{'role': 'user', 'content': 'Hey'}, {'role': 'assistant', 'content': 'Hello, I am your Sleep AI.'}, {'role': 'user', 'content': 'I use my phone before bed.'}]
    print(f"Test History: {test_history}"); assistant_response = query_sleep_ai_multi_turn(test_history)
    if assistant_response: print(f"\nModel Response: {assistant_response}")
    else: print("\nModel failed response.")
    print("\n--- Inference Test Complete ---")