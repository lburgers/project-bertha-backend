from flask import Flask, request, jsonify
from nltk.corpus import twitter_samples, stopwords, opinion_lexicon
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
from nltk import classify
from nltk import NaiveBayesClassifier
from collections import defaultdict
import logging
import tweepy 
import os
import sys
import re
import string
from random import shuffle 

tweet_tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
stemmer = PorterStemmer()

stopwords_english = stopwords.words('english')
pos_tweets = twitter_samples.strings('positive_tweets.json')
neg_tweets = twitter_samples.strings('negative_tweets.json')

pos_words = opinion_lexicon.positive()
neg_words = opinion_lexicon.negative()

# print pos_words
words = defaultdict(str)
for word in pos_words:
    words[word] = 'pos'

for word in neg_words:
    words[word] = 'neg'

 
# Happy Emoticons
emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
    ])
 
# Sad Emoticons
emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ])
 
# all emoticons (happy + sad)
emoticons = emoticons_happy.union(emoticons_sad)
 
def clean_tweets(tweet):
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
 
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
 
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
 
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)
 
    tweets_clean = []    
    for word in tweet_tokens:
        if (word not in stopwords_english and # remove stopwords
              word not in emoticons and # remove emoticons
                word not in string.punctuation): # remove punctuation
            stem_word = stemmer.stem(word) # stemming word
            score = words[stem_word]

            tweets_clean.append((stem_word, score))
 
    return tweets_clean

def bag_of_words(tweet):
    words = clean_tweets(tweet)
    words_dictionary = dict([word, True] for word in words)    
    return words_dictionary

pos_tweets_set = []
for tweet in pos_tweets:
    pos_tweets_set.append((bag_of_words(tweet), 'pos'))    
 
# negative tweets feature set
neg_tweets_set = []
for tweet in neg_tweets:
    neg_tweets_set.append((bag_of_words(tweet), 'neg'))

# print pos_tweets_set

shuffle(pos_tweets_set)
shuffle(neg_tweets_set)
 
train_set = pos_tweets_set + neg_tweets_set

# shuffle(pos_words)
# shuffle(neg_words)
 
# train_set = pos_words + neg_words

# classifier = NaiveBayesClassifier.train(train_set)



test_set = pos_tweets_set[:1000] + neg_tweets_set[:1000]
train_set = pos_tweets_set[1000:] + neg_tweets_set[1000:]

 
classifier = NaiveBayesClassifier.train(train_set)
 
accuracy = classify.accuracy(classifier, test_set)

print(accuracy) # Output: 0.765
 
print (classifier.show_most_informative_features(10))
 
def add_user():
    # username = request.args['username']
    # tweets = get_all_tweets(username)

    test_tweet = "@hayskeys7 But RWD with electric is really good in snow (assuming all weather or winter tires), as traction control is far more precise"
    tweet_words = bag_of_words(test_tweet)

    for word in tweet_words: 
        if word in pos_words: 
            print "POS:", word
        elif word in neg_words:
            print "NEG:", word


    prob_result = classifier.prob_classify(tweet_words)
    print tweet_words, prob_result.prob('pos'), prob_result.prob('neg')

    # data = {}

    # time_series = []

    # num_pos = 0
    # num_neg = 0
    # max_pos = 0
    # max_neg = 0

    # pos_tweet = {}
    # neg_tweet = {}

    # # sentiment_tweets = []
    # for tweet in tweets:
    # 	tweet_words = bag_of_words(tweet[2])
    # 	prob_result = classifier.prob_classify(tweet_words)

    # 	new_data = {}
    # 	# new_data['id'] = tweet[0]
    # 	new_data['time'] = tweet[1]
    # 	new_data['text'] = tweet[2]
    # 	# new_data['sentiment'] = prob_result.max()
    #  #    new_data['score'] = prob_result.prob(new_data['sentiment'])

    #     if prob_result.max() == 'pos': 
    #         num_pos += 1
    #     else: 
    #         num_neg += 1

    #     if prob_result.prob('pos') - prob_result.prob('neg') > max_pos:
    #         pos_tweet = new_data
    #         max_pos = prob_result.prob('pos') - prob_result.prob('neg')
    #     elif prob_result.prob('neg') - prob_result.prob('pos') > max_neg:
    #         neg_tweet = new_data
    #         max_neg = prob_result.prob('neg') - prob_result.prob('pos')

    #     # sentiment_tweets.append(new_data)

    #     entry = {}
    #     entry['time'] = tweet[1]
    #     entry['score'] = prob_result.prob('neg') - prob_result.prob('pos')
    #     entry['pos'] = prob_result.prob('pos')
    #     entry['neg'] = prob_result.prob('neg')
    #     time_series.append(entry)

    # # data['tweets'] = sentiment_tweets
    # data['time_series'] = time_series
    
    # data['num_pos'] = num_pos
    # data['num_neg'] = num_neg

    # data['most_positive'] = pos_tweet
    # data['most_negative'] = neg_tweet

    # return jsonify(data)

if __name__ == '__main__':
    add_user()
