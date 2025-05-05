from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pyttsx3

def generate_response(prompt):
    print("💬 Generating response with Pythia...")

    model_id = "EleutherAI/pythia-70m"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    response = generator(prompt, max_new_tokens=100, do_sample=True)[0]["generated_text"]

    print("\n🤖 Generated Response:\n")
    print(response)

    return response


def text_to_speech(text):
    print("\n🔊 Speaking response...")
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


if __name__ == "__main__":
    user_prompt = input("📝 Enter your prompt for the assistant: ")
    generated_text = generate_response(user_prompt)
    text_to_speech(generated_text)
