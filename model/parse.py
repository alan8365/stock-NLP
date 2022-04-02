from glob import glob
from datasets import Dataset
from emoji import demojize

import re
import pandas as pd


def data_loading(url, is_under_sampling=False):
    with open(url, 'r', encoding='utf-8') as f:
        # data = json.loads(f.read())
        df = pd.read_json(f)
        data = df.copy()
        # data = df.loc[:, ['sentiment', 'body']]
        data = data.loc[df['sentiment'].notnull()]
        data['sentiment'] = pd.Categorical(data['sentiment'])
        data['label'] = data['sentiment'].cat.codes
        data = data.rename(columns={'sentiment': 'labels', 'body': 'sentense'})

        # imbalanced sampling
        if is_under_sampling:
            count_class_0, count_class_1 = data['label'].value_counts()
            small_count = min(count_class_0, count_class_1)
            data_class_0 = data[data['label'] == 0].sample(small_count)
            data_class_1 = data[data['label'] == 1].sample(small_count)
            data = pd.concat([data_class_0, data_class_1], axis=0)

            # data['label'].value_counts().plot(kind='bar', title='Count (label)', color=['#1f77b4', '#ff7f0e'])

        return data


def mask_data_loading(url, tokenizer):
    def stock_symbol_mask(sentense):
        pattern = r'\$[A-Z]*'
        result = re.sub(pattern, tokenizer.mask_token, sentense)

        return result

    with open(url, 'r', encoding='utf-8') as f:
        df = pd.read_json(f)
        data = df.copy()
        data = data.loc[df['sentiment'].notnull()]
        data['sentiment'] = pd.Categorical(data['sentiment'])
        data['body'] = data['body'].apply(normalize_except_compony, args=(tokenizer, ))
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
    return sentense


if __name__ == "__main__":
    data_url = '../crawler/stock/data/**.json'
    url = glob(data_url)[-1]
    data, symbols = mask_data_loading(url)
