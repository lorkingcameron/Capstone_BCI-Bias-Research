import torch.nn as nn
from sentence_transformers import util

from bias_modification.neutral_modification.neutral_synonym_retrieval import find_lower_sentiment_synonym, embedder


# Check if sentence meaning is preserved using sentence embeddings
def meaning_preserved(original_text, modified_text, threshold=0.85):
    original_embedding = embedder.encode(original_text, convert_to_tensor=True)
    modified_embedding = embedder.encode(modified_text, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(original_embedding, modified_embedding).item()
    print(f"Sentence similarity score with new word: {similarity_score}")
    return similarity_score >= threshold


# Neural network structure
class SentimentModifierNN(nn.Module):
    def __init__(self):
        super(SentimentModifierNN, self).__init__()

    def forward(self, original_phrase: list[str], tokenised_phrase: list[str], tokens_to_modify: dict, word_sentiments: dict):
        modified_phrase = tokenised_phrase[:]  # Copy tokens to modify iteratively
        replacements = tokens_to_modify.copy()  # Output dict for replacements by index
        
        while True:
            # Sort words by sentiment score in descending order
            sorted_words = sorted(word_sentiments.items(), key=lambda x: abs(x[1]), reverse=True)
            if not sorted_words:
                break

            # Pick the word with the highest sentiment to modify
            word, sentiment = sorted_words[0]
            word_index = modified_phrase.index(word)
            print(f"\nModifying '{word}' with sentiment {sentiment}")

            # Attempt to find a synonym with lower sentiment
            replacement, new_sentiment = find_lower_sentiment_synonym(word, sentiment, original_phrase, tokenised_phrase)
            print(f"Found replacement '{replacement}' with sentiment {new_sentiment}")
            modified_phrase[word_index] = replacement  # Replace in token list

            # Re-assess sentence meaning
            if meaning_preserved(" ".join(tokenised_phrase), " ".join(modified_phrase)):
                # Update replacements list and word sentiments dictionary 
                replacements[word_index] = replacement
                word_sentiments[replacement] = new_sentiment
                word_sentiments.pop(word)  # Remove old word
                print("Meaning preserved, continuing")
            else:
                print("Meaning lost, stopping")
                break  # Stop if meaning is not preserved

        return replacements


# Function to modify sentiment in the phrase
def modify_sentiment_in_phrase(original_phrase: str, tokenised_phrase: list[str], tokens_to_modify: dict, token_sentiments: dict, initial_sentiment: float):

    # Initialize and run the neural network
    model = SentimentModifierNN()
    replacements = model.forward(original_phrase, tokenised_phrase, tokens_to_modify, token_sentiments)
    
    return replacements
