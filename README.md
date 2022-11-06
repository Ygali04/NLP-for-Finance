# NLP-for-Finance

Food for Thought: 

If a stock broker that uses an NLP ML model to trade stocks loses a lot of money for the client, who is responsible? 

The broker, the model, or the programmer?

Natural Language Processing model designed using one-hot encoding, LSTM, sentence segmentation, word tokenization, parts of speech prediction per token, text lemmatization, and stop word identification to detect booms and busts in the stock market via tweets.

_BERT Model in Development! (2023 Spring)_

# Challenge

NLP transforms text input into computational data utilizing various methods and algorithms (such as Bag-of-Words).

Unfortunately, techniques like one-hot encoding and BOW do not effectively capture sentence structure and word order. Due to to no recording of the order of words, the meaning and sentiment (positive, negative, or neutral) of a sentence can completely change.

# Solution: Long Short-Term Memory Model

LSTM solves this problem by learning the long-term dependencies in the vectorization process through sequence-level interpretation to aid in the effective storage of short term memory (words). Hence the name.

To effectively utilize LSTM, the data must be preprocessed via techniques like stopword & puncuation removal, vector padding, tokenization, stemming, 

_Word Embeddings to improve accuracy currently being implemented_
