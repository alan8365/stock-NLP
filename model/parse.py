import json
import pandas as pd
import torch

from torch.utils.data import DataLoader
from torch.optim import AdamW

from glob import glob
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset, load_metric
from transformers import pipeline, BertTokenizerFast, AutoModelForSequenceClassification, TrainingArguments, Trainer, get_scheduler

data_url = '../crawler/stock/data/**.json'


def data_loading(url):
    with open(url, 'r', encoding='utf-8') as f:
        # data = json.loads(f.read())
        df = pd.read_json(f)
        data = df.loc[:, ['sentiment', 'body']]
        data = data.loc[df['sentiment'].notnull()]
        data = data.rename(columns={'sentiment': 'labels'})
        dataset = Dataset.from_pandas(data)

        return dataset


if __name__ == '__main__':
    url = glob(data_url)[1]
    dataset = data_loading(url)

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    encoded_dataset = dataset.map(
        lambda examples: tokenizer(examples['body']), batched=True)

    small_train_dataset = encoded_dataset.shuffle(seed=42).select(range(1000))
    small_eval_dataset = encoded_dataset.shuffle(seed=42).select(range(1000))

    train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

    # encoded_dataset.set_format(type='torch', columns=[
    #                            'input_ids', 'token_type_ids', 'attention_mask', 'label'])
    # dataloader = torch.utils.data.DataLoader(encoded_dataset, batch_size=32)

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=2)
