**This is the programatic implementation for BCI driven sentiment modification as a proof of concept.**

To download this, make sure to install the pip packages from the **requirements.txt**, and to install the models needed into the models folder.

This repo uses:\
    * **Stanford POS tagger** english left3words, installed under "models/stanford_postagger"\
        * needed files:\
            * english-left3words-distsim.tagger\
            * english-left3words-distim.tagger.props\
            * stanford-postagger.jar\
        * Important steps:\
            * Make sure to have an updated jdk, and to add JAVAHOME to path.\
        * **RoBERTa sentiment analyser**, installed under "models/roberta"\
            * Can be installed using the following code in the sentiment analysis on first run:\
            
                ```
                MODEL = "cardiffnlp/twitter-roberta-base-emotion"
                tokenizer = AutoTokenizer.from_pretrained(MODEL)
                config = AutoConfig.from_pretrained(MODEL)
                model = AutoModelForSequenceClassification.from_pretrained(MODEL)
                model.save_pretrained('YOUR_PATH_TO_ROOT/models/roberta')
                config.save_pretrained('YOUR_PATH_TO_ROOT/models/roberta')
                tokenizer.save_pretrained('YOUR_PATH_TO_ROOT/models/roberta')
                ```
