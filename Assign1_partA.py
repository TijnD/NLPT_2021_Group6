from collections import Counter
import spacy
# open txt file and prepare data
file = "SemEval2018-T3-train-taskB.txt"
text_input = open(file, "r", encoding='utf8')
text_lines = text_input.readlines()
tweets = []
for line in text_lines[1:]:
    tweets.append(line.split('\t')[2])  # only take data from 3th column (Tweets)
text_input.close()
tweets = [line[:-1] for line in tweets]


# Let's run the NLP pipeline on our test input
nlp = spacy.load('en_core_web_sm')
token_count, word_count, type_count, number_of_tweets, len_word = 0, 0, 0, 0, 0

for tweet in tweets:
    doc = nlp(tweet)
    word_frequencies = Counter()
    for sentence in doc.sents:
        words = []
        for token in sentence:
            # Let's filter out punctuation
            if not token.is_punct:
                words.append(token.text)
                len_word += len(token)
        word_frequencies.update(words)

        # print(token.text, token.pos_, token.tag_)
    # print(word_frequencies)
    # num_tokens = len(doc)
    # num_words = sum(word_frequencies.values())
    # num_types = len(word_frequencies.keys())
    token_count += len(doc)
    word_count += sum(word_frequencies.values())
    type_count += len(word_frequencies.keys())
    number_of_tweets += 1

average_words_per_tweet = round(word_count / number_of_tweets, 2)
average_word_length = round(len_word / token_count, 2)
print(token_count, word_count, type_count, average_words_per_tweet, average_word_length)