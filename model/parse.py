from glob import glob
from datasets import Dataset
from emoji import demojize

import re
import pandas as pd


def normalizeToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL"
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token


def normalizeTweet(tweet, tokenizer):
    tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
    normTweet = " ".join([normalizeToken(token) for token in tokens])

    normTweet = (
        normTweet.replace("cannot ", "can not ")
        .replace("n't ", " n't ")
        .replace("n 't ", " n't ")
        .replace("ca n't", "can't")
        .replace("ai n't", "ain't")
    )
    normTweet = (
        normTweet.replace("'m ", " 'm ")
        .replace("'re ", " 're ")
        .replace("'s ", " 's ")
        .replace("'ll ", " 'll ")
        .replace("'d ", " 'd ")
        .replace("'ve ", " 've ")
    )
    normTweet = (
        normTweet.replace(" p . m .", "  p.m.")
        .replace(" p . m ", " p.m ")
        .replace(" a . m .", " a.m.")
        .replace(" a . m ", " a.m ")
    )

    return " ".join(normTweet.split())


def data_loading(url):
    with open(url, 'r', encoding='utf-8') as f:
        # data = json.loads(f.read())
        df = pd.read_json(f)
        data = df.copy()
        # data = df.loc[:, ['sentiment', 'body']]
        data = data.loc[df['sentiment'].notnull()]
        data['sentiment'] = pd.Categorical(data['sentiment'])
        data['label'] = data['sentiment'].cat.codes
        data = data.rename(columns={'sentiment': 'labels', 'body': 'sentense'})

        symbols = set()
        for symbol_list in data['sentense'].str.findall(r'\$[A-Z]+'):
            for symbol in symbol_list:
                symbols.add(symbol)
        return data, symbols


def mask_data_loading(url, tokenizer):
    def stock_symbol_mask(sentense):
        pattern = r'\$[A-Z]*'
        # symbol += re.findall(pattern, sentense)
        result = re.sub(pattern, tokenizer.mask_token, sentense)

        return result

    with open(url, 'r', encoding='utf-8') as f:
        df = pd.read_json(f)
        data = df.copy()
        data = data.loc[df['sentiment'].notnull()]
        data['sentiment'] = pd.Categorical(data['sentiment'])
        data['sentense'] = data['body'].map(stock_symbol_mask)
        data['labels'] = data['body']

        symbols = set()
        for symbol_list in data['body'].str.findall(r'\$[A-Z]+'):
            for symbol in symbol_list:
                symbols.add(symbol)
        return data, symbols


if __name__ == "__main__":
    data_url = '../crawler/stock/data/**.json'
    url = glob(data_url)[-1]
    data, symbols = mask_data_loading(url)
