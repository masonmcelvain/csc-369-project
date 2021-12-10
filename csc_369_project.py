import time
import pandas as pd
import ray
from dask.distributed import Client
from pyspark.sql import SparkSession

"""## Shared helper function to count words in a tweet"""
def count_words_lower(text):
    counts = {}
    for word in text.lower().split():
        if word:
            if word not in counts:
                counts[word] = 0
            counts[word] += 1
    return counts

"""## Pure Python Implementation"""

def count_words_in_tweets(tweets):
    documents = {}
    for text in tweets:
        documents[text] = count_words_lower(text)
    return documents

def pure_python(tweets_csv_path, col):
    pure_python_csv = pd.read_csv(tweets_csv_path)
    squid_df = pure_python_csv.dropna()
    pure_python_tweets = squid_df[col].to_list()
    return count_words_in_tweets(pure_python_tweets)

"""## Ray Implementation"""

ray.init()

@ray.remote
def no_work(x):
    return x

start = time.time()
num_calls = 1000
[ray.get(no_work.remote(x)) for x in range(num_calls)]
print("Ray per task overhead (ms) =", (time.time() - start)*1000/num_calls)

@ray.remote
def ray_count_words_lower(tweets):
    return {text:count_words_lower(text) for text in tweets}

def count_words_in_tweets_ray(ray_tweets):
    step = int(len(ray_tweets)/100)
    ray_range = range(0, len(ray_tweets), step)
    result_ids = [ray_count_words_lower.remote(ray_tweets[i:step]) for i in ray_range]
    results = ray.get(result_ids)

    documents = {}
    [documents.update(sub_documents) for sub_documents in results]
    return documents

def ray_count_words(tweets_csv_path, col):
    ray_csv_df = pd.read_csv(tweets_csv_path)
    squid_df = ray_csv_df.dropna()
    ray_tweets_np = squid_df[col].to_numpy()
    return count_words_in_tweets_ray(ray_tweets_np)

"""## Dask Distributed Implementation"""

client = Client(processes=False)

start = time.time()
num_calls = 1000
[client.gather(iter(client.map(lambda x: x, range(num_calls))))]
print("Dask distributed per task overhead (ms) =", (time.time() - start)*1000/num_calls)

def dask_count_words_lower(tweets):
    return {text:count_words_lower(text) for text in tweets}

def count_words_in_tweets_dask(tweets):
    step = int(len(tweets)/100)
    chunks = [tweets[i:step] for i in range(0, len(tweets), step)]
    chunk_futures = client.scatter(chunks)
    futures = client.map(dask_count_words_lower, chunk_futures)
    results = client.gather(iter(futures))

    documents = {}
    [documents.update(sub_documents) for sub_documents in results]
    return documents

def dask_count_words(tweets_csv_path, col):
    pandas_csv = pd.read_csv(tweets_csv_path)
    squid_df = pandas_csv.dropna()
    tweets = squid_df[col].to_list()
    return count_words_in_tweets_dask(tweets)

"""## Spark Implementation"""

spark = SparkSession.builder.getOrCreate()

def spark_count_words(tweets_csv_path, col):
    squid_spark_csv_df = (
        spark.read
            .option("header", "true")
            .option("mode", "DROPMALFORMED")
            .csv(tweets_csv_path)
    )
    squid_games_df = squid_spark_csv_df.na.drop(subset=(col,))
    squid_games_rdd = squid_games_df.rdd

    tweet_word_counts = squid_games_rdd.map(lambda row: (row[col], count_words_lower(row[col])))
    return tweet_word_counts.collect()

"""## Time the execution of each implementation"""
def time_it(callback, tweets_csv_path, col):
    start = time.time()
    callback(tweets_csv_path, col)
    end = time.time()
    print(f"{callback.__name__}: {end - start} seconds")

"""
The squid games dataset is 26.1 MB.

https://www.kaggle.com/deepcontractor/squid-game-netflix-twitter-data
"""
squid_path = './tweets_v8.csv'
squid_col = 'text'
print(f"\nPreprocessing {squid_path}")

time_it(pure_python, squid_path, squid_col)

time_it(ray_count_words, squid_path, squid_col)

time_it(dask_count_words, squid_path, squid_col)

time_it(spark_count_words, squid_path, squid_col)

"""#### Let's try with a larger dataset

This dataset of tweets about FIFA world cup 2018 is 183.64 MB.

https://www.kaggle.com/rgupta09/world-cup-2018-tweets
"""

fifa_path = './FIFA.csv'
fifa_col = 'Tweet'
print(f"\nPreprocessing {fifa_path}")

time_it(pure_python, fifa_path, fifa_col)

time_it(ray_count_words, fifa_path, fifa_col)

time_it(dask_count_words, fifa_path, fifa_col)

time_it(spark_count_words, fifa_path, fifa_col)

"""#### Even larger dataset

This dataset of tweets about bitcoin is 864.6 MB.

https://www.kaggle.com/kaushiksuresh147/bitcoin-tweets
"""

bitcoin_path = './Bitcoin_tweets.csv'
bitcoin_col = 'text'
print(f"\nPreprocessing {bitcoin_path}")

time_it(pure_python, bitcoin_path, bitcoin_col)

time_it(ray_count_words, bitcoin_path, bitcoin_col)

time_it(dask_count_words, bitcoin_path, bitcoin_col)

time_it(spark_count_words, bitcoin_path, bitcoin_col)
