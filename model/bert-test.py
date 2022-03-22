from transformers import BertTokenizer, BertForMaskedLM
from transformers import TrainingArguments, Trainer
from datasets import Dataset
from datasets import load_metric

import numpy as np
import pandas as pd
from glob import glob

import re

if __name__ == '__main__':
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
            data['sentence'] = data['body'].map(stock_symbol_mask)
            data['labels'] = data['body']

            return data


    data_url = '../crawler/stock/data/**.json'
    url = glob(data_url)[-1]
    data = mask_data_loading(url)

    dataset = Dataset.from_pandas(data.loc[:, ['labels', 'sentence']])
    dataset = dataset.remove_columns('__index_level_0__')


    def encode(example):
        label = tokenizer(example['labels'], padding='max_length', truncation=True)
        # 101, 51, 1234, 12541, 151
        result = tokenizer(example['sentence'], padding='max_length', truncation=True)
        # 101, 103, 103, 103
        # result['labels'] = label['input_ids']

        # masked_position = [i for i in range(len(result['input_ids'])) if result['input_ids'][i] == tokenizer.mask_token_id]
        # result['labels'] = label['input_ids']
        result['labels'] = [-100] * len(label['input_ids'])
        # result['decoder_input_ids'] = label['input_ids']
        # for i in range(len(result['labels'])):
        #     if not i in masked_position:
        #         result['labels'][i] = -100

        return result

    encoded_dataset = dataset.map(encode, batched=True)

    # print(encoded_dataset[0]['sentense'])
    # print(encoded_dataset[0]['input_ids'])
    type(encoded_dataset[0]['labels'])
    encoded_dataset[0]['labels'] = [-100] * 10
    print(encoded_dataset)

    metric = load_metric("accuracy")


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)


    training_args = TrainingArguments(output_dir="test_trainer")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
