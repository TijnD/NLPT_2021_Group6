from collections import Counter
import spacy

def input_to_tweets(file_name):
    text_input = open(file, "r", encoding='utf8')
    text_lines = text_input.readlines()
    all_tweets = []
    for line in text_lines[1:]:
        all_tweets.append(line.split('\t')[2])  # only take data from 3th column (Tweets)
    text_input.close()
    all_tweets = [line[:-1] for line in all_tweets]
    return all_tweets

def frequencies(tweets):
    token_count, len_word = 0, 0
    word_frequencies = Counter()
    POS_tagger = Counter()
    words = []
    POS_tags = []
    nouns = []
    verbs = []
    propn = []
    det = []
    adp = []
    pron = []
    adj = []
    adv = []
    aux = []
    part = []
    for tweet in tweets:
        doc = nlp(tweet)
        for sentence in doc.sents:

            for token in sentence:
                # Let's filter out punctuation
                if not token.is_punct:
                    words.append(token.text)
                    len_word += len(token)
                    POS_tags.append(token.tag_)
                    nouns, verbs, propn, det, adp, pron, adj, adv, aux, part = pos_tagger(token, nouns, verbs, propn, det, adp, pron, adj, adv, aux, part)
            POS_tagger.update(POS_tags) # used to find top 10 pos-tags
            word_frequencies.update(words)
        token_count += len(doc)
    word_count = sum(word_frequencies.values())
    type_count = len(word_frequencies.keys())
    return token_count, word_count, type_count, len_word, word_frequencies, nouns, verbs, propn, det, adp, pron, adj, adv, aux, part

def pos_tagger(token, nouns, verbs, propn, det, adp, pron, adj, adv, aux, part):
    if token.tag_ == 'NN':
        nouns.append(token)
    elif token.tag_ == 'NNP':
        verbs.append(token)
    elif token.tag_ == 'IN':
        propn.append(token)
    elif token.tag_ == 'DT':
        det.append(token)
    elif token.tag_ == 'PRP':
        adp.append(token)
    elif token.tag_ == 'RB':
        pron.append(token)
    elif token.tag_ == 'JJ':
        adj.append(token)
    elif token.tag_ == 'VB':
        adv.append(token)
    elif token.tag_ == 'NNS':
        aux.append(token)
    elif token.tag_ == 'VBP':
        part.append(token)
    return nouns, verbs, propn, det, adp, pron, adj, adv, aux, part

# open txt file and prepare data
file = "SemEval2018-T3-train-taskB.txt"
tweets = input_to_tweets(file)

# Let's run the NLP pipeline on our test input
nlp = spacy.load('en_core_web_sm')
tokens, words, types, word_length, word_frequencies, nouns, verbs, propn, det, adp, pron, adj, adv, aux, part = frequencies(tweets)
average_words_per_tweet = round(words / len(tweets), 2)
average_word_length = round(word_length / tokens, 2)
print("number of tokens: {}\n"
      "number of words: {}\n"
      "number of types: {}\n"
      "average words per tweet: {}\n"
      "average word length: {}".format(tokens, words, types, average_words_per_tweet, average_word_length))

print(nouns, verbs, propn, det, adp, pron, adj, adv, aux, part)