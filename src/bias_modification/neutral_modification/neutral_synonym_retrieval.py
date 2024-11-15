import re
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import wordnet
from nltk.wsd import lesk

from bias_modification.sentiment_analysis import run_roberta_sentiment_analysis


# Load embedding model for similarity check
embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")


def get_definition_in_context(word, sentence):
    """Retrieve the definition of a word in the specific context of a sentence."""
    # Apply the lesk algorithm to find the best matching sense of the word in context
    best_synset = lesk(sentence, word)
    
    if best_synset:
        return best_synset.definition()
    else:
        return "No definition found for the context."


def clean_definition(definition):
    """Helper function to clean up WordNet definitions for comparison."""
    return set(re.findall(r'\w+', definition.lower()))


def get_antonyms(word):
    """Retrieve antonyms for a word from WordNet."""
    antonyms = set()
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            for antonym in lemma.antonyms():
                antonyms.add(antonym.name().replace('_', ' '))
    return antonyms


def get_synonyms(word, original_antonyms):
    """Retrieve synonyms from WordNet for a word with a specific POS tag, excluding original word's antonyms."""
    synonyms = set()
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            new_word = lemma.name().replace('_', ' ')
            # Exclude if this synonym is an antonym of the original word
            if new_word not in original_antonyms:
                synonyms.add(new_word)
    return list(synonyms)


def find_lower_sentiment_synonym(original_word, original_sentiment, context_sentence, tokenised_context):
    # Get original word embedding in context
    context_with_word = context_sentence.replace(original_word, f"<{original_word}>")
    original_embedding = embedder.encode(context_with_word, convert_to_tensor=True)
    
    # Get original definition in context
    original_definition = get_definition_in_context(original_word, tokenised_context)
    print("original_definition", original_definition)

    # Gather antonyms
    original_antonyms = get_antonyms(original_word)
    print("original_antonyms", original_antonyms)
    
    # Gather synonyms
    synonyms = get_synonyms(original_word, original_antonyms)
    print("word", original_word)
    print("synonyms", synonyms)
    
    valid_synonyms = []    
    for synonym in synonyms:
        new_sentiment = run_roberta_sentiment_analysis(synonym)

        # Check if sentiment is lower than the current word's sentiment
        if abs(new_sentiment) < abs(original_sentiment):
            print("valid synonym", synonym, new_sentiment)
            valid_synonyms.append((synonym, new_sentiment))

    print("valid_synonyms", valid_synonyms)
    # Rank valid synonyms by similarity to the original word in context
    ranked_synonyms = []
    for synonym, new_sentiment in valid_synonyms:
        context_with_synonym = context_sentence.replace(original_word, f"<{synonym}>")
        synonym_embedding = embedder.encode(context_with_synonym, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(original_embedding, synonym_embedding).item()
        ranked_synonyms.append((synonym, new_sentiment, similarity))

    # Sort by similarity, then by sentiment if multiple have the same similarity
    ranked_synonyms = sorted(ranked_synonyms, key=lambda x: (-x[2], abs(x[1])))

    # Return the best synonym with closest meaning and lowest sentiment
    if ranked_synonyms:
        best_synonym, best_sentiment, _ = ranked_synonyms[0]
        return best_synonym, best_sentiment
    else:
        return original_word, original_sentiment  # Return original if no suitable synonym is found