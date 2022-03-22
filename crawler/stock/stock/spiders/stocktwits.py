from ..items import StockItem

import logging
import scrapy
import json
from datetime import datetime

from scrapy.utils.log import configure_logging

BASE_URL = 'https://api.stocktwits.com/api/2/streams/symbol/TSLA.json?max='


class StockSpider(scrapy.Spider):
    name = "stock"
    configure_logging(install_root_handler=False)
    logging.basicConfig(
        filename=f'logs/{datetime.now().strftime("%Y-%m-%dT%H%M%S")}.txt',
        format='%(levelname)s: %(message)s',
        encoding='utf-8'        
    )

    def __init__(self):
        self.count = 0
        self.data_limit = 10000
        self.NASDQ_top10 = [
            'AAPL',
            'MSFT',
            'GOOG',
            'GOOGL',
            'AMZN',
            'TSLA',
            'NVDA',
            'FB',
            'TSM',
            'UNH'
        ]

    def start_requests(self):
        urls = [
            f'https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json' for symbol in self.NASDQ_top10]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        items = StockItem()

        data = json.loads(response.text)
        messages = data['messages']

        for message in messages:
            sentiment = message['entities']['sentiment']
            if sentiment:
                sentiment = sentiment['basic']

            # Info filter
            items['message_id'] = message['id']
            items['body'] = message['body']
            items['sentiment'] = message['entities']['sentiment']
            items['created_at'] = message['created_at']

            yield items

        self.count += len(messages)

        if self.count < self.data_limit:
            new_max = messages[-1]['id']
            new_url = BASE_URL + str(new_max)
            yield scrapy.Request(url=new_url, callback=self.parse)
        else:
            self.count = 0
