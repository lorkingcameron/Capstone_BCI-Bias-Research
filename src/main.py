
import os

from plot_generator import *

from eeg_classification.preprocessing import *
from eeg_classification.cnn import *

from bias_modification.nlp_phrase_augmentation import *
from bias_modification.sentiment_analysis import *
from bias_modification.word_tokenisation import *
from bias_modification.neutral_word_augmentation import *
from bias_modification.reversal_word_augmentation import *


# os.system("cls")
# os.system("pip install -r requirements.txt")


PATH = os.path.dirname(os.path.dirname(__file__))


def perform_eeg_classification():
    # all_data, data_x, data_y, max_epochs, max_channels, time_points = data_preprocessing_2_classes()
    all_data, data_x, data_y, max_epochs, max_channels, time_points = data_preprocessing_5_classes()
    # TODO modify 5 class to interpret as two without reduction in accuracy to improve data size

    print(data_x[0].shape)
    print(data_y)
    print(max_epochs, max_channels, time_points)

    cnn_hyperparameters = {
        'num_epochs': max_epochs,
        'num_channels': max_channels,
        'num_time_points': time_points,
        'num_classes': 4
    }
    
    run_cnn(data_x, data_y, cnn_hyperparameters)


def perform_phrase_modification():
    negative_phrases, positive_phrases = extract_phrases_from_files()
    
     # Test specific index
    negative_index_to_test = None
    positive_index_to_test = None
    
    for index, phrase in enumerate(negative_phrases):
        if negative_index_to_test and index != negative_index_to_test:
            continue
        
        modify_phrase(phrase)
        input("Press Enter to continue...")
        
    for index, phrase in enumerate(positive_phrases):
        if positive_index_to_test and index != positive_index_to_test:
            continue
        
        modify_phrase(phrase)
        input("Press Enter to continue...")


def extract_phrases_from_files():
    with open(f'{PATH}/Statements/negative.txt', encoding="utf-8") as f:
        negative_content = f.read().split("\n")
    with open(f'{PATH}/Statements/positive.txt', encoding="utf-8") as f:
        positive_content = f.read().split("\n")
        
    return negative_content, positive_content
    

def modify_phrase(text):
    print("\nOriginal phrase:", text)
    
    # Get overall sentiment score for the text
    overall_sentiment = run_roberta_sentiment_analysis(text)
    print("Overall Sentiment", overall_sentiment)
    
    # Get word tokens
    tokenised_text = tokenise_text(text)
    print("\nTokens", tokenised_text)
    
    # Get POS tags for each word
    tokenised_text_with_pos = get_words_with_pos(tokenised_text)
    print("\nPOS", tokenised_text_with_pos)
    
    # Identify words to modify
    tokens_to_modify, tokens_to_modify_sentiments = identify_words_to_modify(overall_sentiment, tokenised_text_with_pos)
    print("\nWords to Replace:", tokens_to_modify, "\n\twith sentiments", tokens_to_modify_sentiments)
    
    # ! not working NEUTRALIZATION cause of sentence context
    # # Find replacements for these words
    # replacements = modify_sentiment_in_phrase(text, tokenised_text, tokens_to_modify, tokens_to_modify_sentiments, overall_sentiment)
    # print("\nReplacement words:", replacements)
    
    # # Final modified phrase and sentiment
    # updated_text = tokenised_text[:]
    # for index, word in replacements.items():
    #     updated_text[index] = word
    # updated_text = " ".join(updated_text)
    
    # Reverse the sentiment of the phrase
    updated_text = reverse_sentiment(tokens_to_modify, text)
    print("\nReversed sentiment phrase:", updated_text)
    
    # print final sentiment
    updated_sentiment = run_roberta_sentiment_analysis(updated_text)
    print("Updated sentiment:", updated_sentiment)


def main():
    os.system("cls")
    
    perform_eeg_classification()
    
    # perform_phrase_modification()
    
    # generate_plot()


if __name__ == "__main__":
    main()
