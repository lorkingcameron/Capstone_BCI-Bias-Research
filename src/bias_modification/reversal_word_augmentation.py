import os
import spacy
import numpy as np
from nltk.corpus import wordnet as wn
from bias_modification.sentiment_analysis import run_roberta_sentiment_analysis


PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
nlp = spacy.load("en_core_web_md")


def get_antonym(word):
    """Find an antonym for the given word using WordNet."""
    antonyms = []
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            if lemma.antonyms():
                antonyms.append(lemma.antonyms()[0].name())
    return antonyms[0] if antonyms else None


def apply_negation(doc, sentence, target_word):
    """Apply negation to the target word in the sentence, ensuring grammatical and contextual integrity."""
    
    modified_tokens = []
    for i, token in enumerate(doc):
        # Check if the token matches the target word
        if token.text.lower() == target_word.lower():
            # Check if "not" is directly before the target word
            if i > 0 and doc[i - 1].text.lower() == "not":
                # Remove "not" by skipping the previous token
                modified_tokens.pop()  # Remove the last added "not"
            else:
                # Add "not" before the target word
                modified_tokens.append("not")
            # Add the target word itself
            modified_tokens.append(token.text)
        else:
            # Keep all other tokens as they are
            modified_tokens.append(token.text)

    # Form the modified sentence
    modified_sentence = " ".join(modified_tokens)

    # Verify the modified sentence retains similar context to the original
    similarity_score = doc.similarity(nlp(modified_sentence))
    print("Similarity Score:", similarity_score)
    if similarity_score >= 0.8:
        return modified_sentence
    else:
        return sentence  # Return the original if the context changes too much


def reverse_sentiment(tokens_to_modify, sentence):
    """Reverse the sentiment of the sentence."""
    doc = nlp(sentence)

    modified_sentence = str(sentence)
    
    for index, word in tokens_to_modify.items():
        antonym = get_antonym(word)
        antonym_applied = False
        
        if antonym:
            # Replace word with antonym
            temp_sentence = modified_sentence.replace(word, antonym)
            
            # Check context similarity to ensure the object and context are preserved
            similarity_score = doc.similarity(nlp(modified_sentence))
            print("Similarity Score:", similarity_score)
            if similarity_score >= 0.8 :
                modified_sentence = str(temp_sentence)  # Accept the replacement
                antonym_applied = True
                print(f"Replaced '{word}' with '{antonym}'")
            else:
                print(f"Skipping replacement for '{word}' as it altered the context too much.")
        else:
            print(f"No antonym found for '{word}'")
            
        if not antonym_applied:
            # Apply negation if antonym is unsuccessful
            modified_sentence = apply_negation(doc, modified_sentence, word)
    
    return modified_sentence
