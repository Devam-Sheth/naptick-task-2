import pandas as pd
import json
import random

# Load the dataset
sleep_df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

# Shuffle and sample 1000 rows (or fewer if dataset is smaller)
sampled_df = sleep_df.sample(n=min(1000, len(sleep_df)), random_state=42).reset_index(drop=True)

# Define assistant tips
tips = [
    "To improve your sleep, maintain a consistent schedule and reduce screen time before bed.",
    "Consider avoiding caffeine late in the day and creating a relaxing bedtime routine.",
    "Make sure your bedroom is quiet, dark, and at a comfortable temperature.",
    "Engage in regular physical activity, but avoid vigorous workouts close to bedtime.",
    "Try meditation or breathing exercises to wind down before sleep."
]

# Function to create a structured 7-turn conversation
def generate_conversation(row):
    sleep_quality = row['Quality of Sleep']
    sleep_duration = row['Sleep Duration']
    
    conversation = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello, I am your Sleep AI."},
        {"role": "user", "content": "Can you help me improve my sleep cycle?"},
        {"role": "assistant", "content": "For sure. Tell me about your sleep quality."},
        {"role": "user", "content": f"My sleep quality is {sleep_quality}."},
        {"role": "assistant", "content": "Ohh, tell me about your sleep duration."},
        {"role": "user", "content": f"I sleep for {sleep_duration} hours."}
    ]

    if sleep_quality >= 7 and sleep_duration >= 7:
        final_response = "Great job! Your sleep habits are already on the right track. Keep it up!"
    else:
        final_response = random.choice(tips)

    conversation.append({"role": "assistant", "content": final_response})
    
    return {"conversation": conversation}

# Generate all conversations
final_dataset = [generate_conversation(row) for _, row in sampled_df.iterrows()]

# Save to JSONL format
output_path = "sleep_coach_multichat_1000.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for convo in final_dataset:
        json.dump(convo, f)
        f.write("\n")

output_path
