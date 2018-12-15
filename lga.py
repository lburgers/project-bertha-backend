#!/usr/bin/python3

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from nltk.corpus import twitter_samples, stopwords
from nltk.corpus import wordnet as wn
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
from gensim import corpora
import gensim
from nltk import classify
from nltk import NaiveBayesClassifier
import logging
import tweepy 
import os
import sys
import re
import string
from random import shuffle 

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

#Twitter API credentials
consumer_key = 'pf14xpY1B4OS6p5NhVvKRDraQ'
consumer_secret = 'mMVDAiMsrTfRqcZjDpUjEV6TMh0qOYhZeUOT6UQH6EcQUKvlKl'
access_key = '634780289-kQIogwhtHfQIhogOfV7meFIJrf2DPGLEhsSlF94m'
access_secret = 'EitrLcZkkZ1FfgTtroB6ETAbalmg7Iul4lAmcPBkucd9m'

tweet_tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
stemmer = PorterStemmer()

stopwords_english = stopwords.words('english')
pos_tweets = twitter_samples.strings('positive_tweets.json')
neg_tweets = twitter_samples.strings('negative_tweets.json')

 
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
 
def clean_tweets(tweet, stem=True):
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
            
            stem = word
            if stem:
                stem_word = stemmer.stem(word) # stemming word
                
            tweets_clean.append(stem_word)
 
    return tweets_clean

def bag_of_words(tweet):
    words = clean_tweets(tweet)
    words_dictionary = dict([word, True] for word in words)    
    return words_dictionary


# shuffle(pos_tweets_set)
# shuffle(neg_tweets_set)
 
# train_set = pos_tweets_set + neg_tweets_set

# classifier = NaiveBayesClassifier.train(train_set)
 
def get_all_tweets(screen_name, num_tweets):
    #Twitter only allows access to a users most recent 3240 tweets with this method
    
    #authorize twitter, initialize tweepy
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
    
    #initialize a list to hold all the tweepy Tweets
    alltweets = []  
    
    #make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name = screen_name,count=200)
    
    #save most recent tweets
    alltweets.extend(new_tweets)
    
    #save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1
    count = 0
    #keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0 and count <= num_tweets:
        count += 100
        #all subsiquent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(screen_name = screen_name,count=100,max_id=oldest)
        #save most recent tweets
        alltweets.extend(new_tweets)

        #update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1
        
    
    #transform the tweepy tweets into a 2D array that will populate the csv 
    outtweets = [[tweet.id_str, tweet.created_at.strftime('%m/%d/%Y'), tweet.text] for tweet in alltweets]

    return outtweets

def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

train_tweets = []
for tweet in pos_tweets:
    tweet = clean_tweets(tweet, False)
    tweet = [get_lemma(t) for t in tweet]
    train_tweets.append(tweet)
 
for tweet in neg_tweets:
    tweet = clean_tweets(tweet, False)
    tweet = [get_lemma(t) for t in tweet]
    train_tweets.append(tweet)


train_tweets = train_tweets[:600]
print('initializing model')

dictionary = corpora.Dictionary(train_tweets)
corpus = [dictionary.doc2bow(text) for text in train_tweets]
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 5, id2word=dictionary, passes=15)
print('model created')


# username = request.args['username']
# num_tweets = int(request.args['num_tweets'])
tweets = get_all_tweets('lukas', 100)

all_tweets = ''
data = {}

time_series = []

num_pos = 0
num_neg = 0
num_neu = 0
max_pos = 0
max_neg = 0

pos_tweet = {}
neg_tweet = {}

# sentiment_tweets = []
for tweet in tweets:
    print('adding tweet')
    all_tweets += tweet[2]
    # prob_result = analyzer.polarity_scores(tweet[2])

    # new_data = {}
    # # new_data['id'] = tweet[0]
    # new_data['time'] = tweet[1]
    # new_data['text'] = tweet[2]

    # if prob_result['compound'] >= 0.1:
    #     num_pos += 1
    # elif prob_result['compound'] <= -0.1: 
    #     num_neg += 1
    # else:
    #     num_neu += 1

    # if prob_result['compound'] > max_pos:
    #     pos_tweet = new_data
    #     max_pos = prob_result['compound']
    # elif prob_result['compound'] < max_neg:
    #     neg_tweet = new_data
    #     max_neg = prob_result['compound']

    # # sentiment_tweets.append(new_data)

    # entry = {}
    # entry['time'] = tweet[1]
    # entry['score'] = prob_result['compound']
    # entry['pos'] = prob_result['pos']
    # entry['neg'] = prob_result['neg']
    # entry['neu'] = prob_result['neu']
    # time_series.append(entry)

all_tweets = clean_tweets(all_tweets, False)
all_tweets = [tweet for tweet in all_tweets if len(tweet) > 4]
all_tweets = [get_lemma(tweet) for tweet in all_tweets]
all_tweets_bow = dictionary.doc2bow(all_tweets)


# data['tweets'] = sentiment_tweets
# data['time_series'] = time_series
print(ldamodel.print_topics(num_words=8))
print(ldamodel.get_document_topics(all_tweets_bow))
# data['num_pos'] = num_pos
# data['num_neg'] = num_neg
# data['num_neu'] = num_neu

# data['most_positive'] = pos_tweet
# data['most_negative'] = neg_tweet
