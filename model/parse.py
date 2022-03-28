from glob import glob
from datasets import Dataset
from emoji import demojize

import re
import pandas as pd


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


def normalize_except_compony(sentense, tokenizer):
    sentense = re.split(r'(\$[A-Z]+)', sentense)
    a = [tokenizer.normalizeTweet(i) if i.find('$') == -1 else i for i in sentense]
    sentense = " ".join(a)


if __name__ == "__main__":
    data_url = '../crawler/stock/data/**.json'
    url = glob(data_url)[-1]
    data, symbols = mask_data_loading(url)
