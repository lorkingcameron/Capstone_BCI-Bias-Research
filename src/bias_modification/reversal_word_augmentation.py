import os
import spacy
from nltk.tag.stanford import StanfordPOSTagger
from nltk.corpus import wordnet as wn


PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
nlp = spacy.load("en_core_web_md")


def _get_word_pos(word):
    """Returns the pos tag for a word"""    
    jar_path = f"{PATH}/models/stanford_postagger/stanford-postagger.jar"
    model_path = f"{PATH}/models/stanford_postagger/english-left3words-distsim.tagger"
    
    tagger = StanfordPOSTagger(model_path, jar_path)
    # return the pos tag only for the word
    return tagger.tag(word.split())[0][1]


def _check_antonym_possible(word):
    """Check if an antonym is possible for the given word."""    
    return word.lower() not in ["not", "only", "is", "was", "are", "were", "am"]


def _check_not_required_after(word, word_pos):
    """Check if the word requires a 'not' to be added after it."""
    return word.lower() in ["is", "was", "are", "were", "am"] or word_pos == 'MD'


def _join_with_sentence_formatting(tokens: list[str]):
    """Join the tokens to form a sentence."""   
    updated_tokens = [] 
    for i, token in enumerate(tokens):
        if i>0 and token in [".", ",", "!", "?"]:
            updated_tokens[-1] += token
            continue

        updated_tokens.append(token)

    return " ".join(updated_tokens).replace(" - ", "-").capitalize()


def get_antonym(word, word_pos):
    """Find an antonym for the given word using WordNet."""
    antonyms = []
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            if lemma.antonyms():
                for antonym in lemma.antonyms():
                    if _get_word_pos(antonym.name()) == word_pos:
                        antonyms.append(antonym.name())
                        break
    return antonyms[0] if antonyms else None


def apply_negation(doc, sentence, target_word, target_word_pos):
    """Apply negation to the target word in the sentence, ensuring grammatical and contextual integrity."""
    
    skip_next = False
    modified_tokens = []
    for i, token in enumerate(doc):
        if skip_next:
            skip_next = False
            continue
        
        # Check if the token matches the target word
        if token.text.lower() == target_word.lower():
            # Check if the word itself is "not", and remove
            if target_word.lower() == "not":
                continue
            
            # "not" required after the target word
            elif _check_not_required_after(target_word, target_word_pos):
                # Add the target word itself
                modified_tokens.append(token.text)
                # Check if not already after the target word
                if i < len(doc) - 1 and doc[i + 1].text.lower() == "not":
                    skip_next = True # Skip the next token which is the not to remove it
                else:
                    modified_tokens.append("not") # Add "not" after the target word
            # "not" required before the target word
            else:
                # Check if "not" is already before the target word
                if i > 0 and doc[i - 1].text.lower() == "not":
                    modified_tokens.pop()  # Remove the last added token "not"
                else:
                    # Add "do" before not if the word is only and its not starting the sentence
                    if target_word == "only" and i != 0:
                        modified_tokens.append("do")
                    
                    modified_tokens.append("not") # Add "not" before the target word
                # Add the target word itself
                modified_tokens.append(token.text)
        else:
            # Keep all other tokens as they are
            modified_tokens.append(token.text)

    # Form the modified sentence
    modified_sentence = _join_with_sentence_formatting(modified_tokens)

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
    
    for index, (word, word_pos) in tokens_to_modify.items():
        antonym = None
        if _check_antonym_possible(word):
            antonym = get_antonym(word, word_pos)
        
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
            modified_sentence = apply_negation(doc, modified_sentence, word, word_pos)
    
    return modified_sentence
