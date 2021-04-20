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


file = "SemEval2018-T3-train-taskB.txt"
tweets = input_to_tweets(file)

nlp = spacy.load('en_core_web_sm')
token_count, len_word = 0, 0
word_frequencies = Counter()

for doc in nlp.pipe(tweets):
    words = []
    for token in doc:
            # Let's filter out punctuation
        if not token.is_punct:
            words.append(token.text)
            len_word += len(token)
    word_frequencies.update(words)
    token_count += len(doc)

word_count = sum(word_frequencies.values())
print(word_count)
type_count = len(word_frequencies.keys())
average_words_per_tweet = round(word_count / len(tweets), 2)
average_word_length = round(len_word / token_count, 2)
print("number of tokens: {}\n"
      "number of words: {}\n"
      "number of types: {}\n"
      "average words per tweet: {}\n"
      "average word length: {}".format(token_count, word_count, type_count, average_words_per_tweet, average_word_length))