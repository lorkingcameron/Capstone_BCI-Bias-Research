
import numpy as np
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from scipy.special import softmax


def run_roberta_sentiment_analysis(phrase):
    # Load the model
    MODEL = f"./models/roberta"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    # model.save_pretrained(MODEL)

    # Assuming data is a list
    encoded_input = tokenizer(phrase, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    
    # Print labels and scores
    consolidated_sentiment = scores[2] - scores[0]
    
    print("\n" + phrase)
    
    for i in range(scores.shape[0]):
        l = config.id2label[i]
        s = scores[i]
        print(f"{i+1}) {l} {np.round(float(s), 4)}")
    
    print(f"Consolidated sentiment: {consolidated_sentiment}")
    
    return consolidated_sentiment

