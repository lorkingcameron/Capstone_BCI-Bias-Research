import numpy as np
import nltk
from nltk.tag.stanford import StanfordPOSTagger
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from bias_modification.sentiment_analysis import *


# Ensure NLTK resources are available
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')


def is_replaceable(pos_tag):
    """Returns (bool) if the POS tag corresponds to an adjective or not."""
    return pos_tag in ['JJ', 'JJR', 'JJS', 'VBP', 'VBZ', 'VBD', 'VBN', 'VBG', 'VB', 'RB', 'MD']  # Adjective tags in NLTK


def tokenise_text(text):
    """Returns (list[str]) tokenised list of words from input text."""
    text_tokenised = word_tokenize(text)
    
    return text_tokenised


def get_words_with_pos(abs_path, tokenised_input):
    """Returns a list of tuples with words and their POS tags."""    
    jar_path = f"{abs_path}/models/stanford_postagger/stanford-postagger.jar"
    model_path = f"{abs_path}/models/stanford_postagger/english-left3words-distsim.tagger"
    
    tagger = StanfordPOSTagger(model_path, jar_path)
    return tagger.tag(tokenised_input)


def identify_words_to_modify(overall_sentiment, words_with_pos):
    '''
    Take a tokenised phrase and return the descriptuve words with high sentiment impact for modification.

        Parameters:
            overall_sentiment (float): overall sentiment score for the input text between -1 and 1
            words_with_pos (list[tuple[str, str]]): list of words with POS tags by (token, POS) pair

        Returns:
            (list[str]): list of words worth modifying
    '''
    potential_words_to_modify = {}
    word_sentiments = {}
    most_sentiment = None
    most_sentiment_index = None
    for index, (word, pos) in enumerate(words_with_pos):
        # Only consider certain word types
        if not is_replaceable(pos):
            continue
        
        # Only consider words that align with the overall phrase sentiment
        sentiment_score = run_roberta_sentiment_analysis(word)
        
        # Change just the strongest word contributing to the sentiment
        if pos == 'MD':
            most_sentiment = sentiment_score
            most_sentiment_index = index
            break
        elif overall_sentiment < 0:
            if most_sentiment is None or sentiment_score < most_sentiment:
                most_sentiment = sentiment_score
                most_sentiment_index = index
        else:
            if most_sentiment is None or sentiment_score > most_sentiment:
                most_sentiment = sentiment_score
                most_sentiment_index = index
            
        # # ! Old code, considers all words of same sentiment direction at sentence
        # if np.sign(sentiment_score) == np.sign(overall_sentiment):
        #     potential_words_to_modify[index] = word
        #     word_sentiments[word] = sentiment_score
        
    potential_words_to_modify[most_sentiment_index] = words_with_pos[most_sentiment_index][0]
    word_sentiments[words_with_pos[most_sentiment_index][0]] = most_sentiment
    return potential_words_to_modify, word_sentiments


def find_synonyms(word):
    '''
    Find synonyms for a given word using WordNet.

        Parameters:
            word (str): word to find synonyms for

        Returns:
            (list[str]): list of synonyms for the input word
    '''
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())  # get synonyms
    return list(synonyms)