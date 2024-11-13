import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import numpy as np
from bias_modification.sentiment_analysis import run_roberta_sentiment_analysis
from langdetect import detect

# Define a reward function to encourage neutrality
def calculate_reward(generated_text):
    sentiment_score = run_roberta_sentiment_analysis(generated_text)
    # Reward is maximized when the sentiment score is near zero (neutral)
    reward = -abs(sentiment_score)  # More negative scores if further from neutral (0)
    
    # Language check
    language = detect(generated_text)
    if language != 'en':
        reward *= 2  # Penalty if the text is not in English

    return reward

# Fine-tuning function with reinforcement learning
def reinforce_neutralization(input_text, epochs=5, lr=1e-5):
    
    # Load a pretrained T5 model and tokenizer
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        # Prepare input prompt
        prompt = f"neutralize the sentiment: {input_text}"
        inputs = tokenizer(prompt, return_tensors="pt")

        # Generate output with sampling for variability
        outputs = model.generate(**inputs, max_length=50, do_sample=True, top_k=50, top_p=0.95)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Calculate reward
        reward = calculate_reward(generated_text)
        print(f"Epoch {epoch+1}: '{generated_text}' with reward {reward}")

        # Convert reward to a loss (negative reward for loss minimization)
        loss = -reward

        # Backpropagate the loss
        loss_tensor = torch.tensor(loss, requires_grad=True)
        loss_tensor.backward()
        optimizer.step()
        optimizer.zero_grad()

    return generated_text
