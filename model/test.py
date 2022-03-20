import re
import json
import pandas as pd
import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.optim import AdamW

from glob import glob
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset, load_metric
from transformers import pipeline, BertTokenizerFast, AutoModelForSequenceClassification, TrainingArguments, Trainer, get_scheduler, AutoConfig, AutoModelForSequenceClassification

data_url = '../crawler/stock/data/**.json'


def data_loading(url):
    with open(url, 'r', encoding='utf-8') as f:
        # data = json.loads(f.read())
        df = pd.read_json(f)
        data = df.copy()
        # data = df.loc[:, ['sentiment', 'body']]
        data = data.loc[df['sentiment'].notnull()]
        data['sentiment'] = pd.Categorical(data['sentiment'])
        data['label'] = data['sentiment'].cat.codes
        data = data.rename(columns={'sentiment': 'labels'})

        return data


def pre_train_bert(data):
    dataset = Dataset.from_pandas(
        data.loc[:, ['label', 'body']]).remove_columns('__index_level_0__')

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    encoded_dataset = dataset.map(lambda examples: tokenizer(
        examples['body'], padding='max_length', truncation=True), batched=True)

    small_train_dataset = encoded_dataset.shuffle(seed=42).select(range(10))
    small_eval_dataset = encoded_dataset.shuffle(seed=42).select(range(10))

    train_dataloader = DataLoader(
        small_train_dataset, shuffle=True, batch_size=8)
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=2
    )

    training_args = TrainingArguments(
        output_dir="test_trainer", evaluation_strategy="epoch")

    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == '__main__':
    from transformers import BertTokenizer, BertForMaskedLM
    from transformers import Trainer, TrainingArguments
    from glob import glob
    from datasets import Dataset
    from datasets import load_metric

    import re
    import pandas as pd
    import numpy as np

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BertForMaskedLM.from_pretrained('bert-base-cased')

    def mask_data_loading(url):
        def stock_symbol_mask(sentense):
            result = re.sub(r'\$\w*', '[MASK]', sentense)

            return result

        with open(url, 'r', encoding='utf-8') as f:
            df = pd.read_json(f)
            data = df.copy()
            data = data.loc[df['sentiment'].notnull()]
            data['sentiment'] = pd.Categorical(data['sentiment'])
            data['labels'] = data['body'].map(stock_symbol_mask)

            return data

    data_url = '../crawler/stock/data/**.json'
    url = glob(data_url)[-1]
    data = mask_data_loading(url)

    dataset = Dataset.from_pandas(data.loc[:, ['labels', 'body']])
    dataset = dataset.remove_columns('__index_level_0__')

    def encode(example):
        return tokenizer(example['body'], example['labels'], padding='max_length', truncation=True)

    encoded_dataset = dataset.map(encode, batched=True)
    encoded_dataset[0]

    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir="./bert-retrained",
        # overwrite_output_dir=True,
        # num_train_epochs=5,
        # per_device_train_batch_size=8,
        # save_steps=5,
        # save_total_limit=2,
        # seed=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model("./bert-retrained")
