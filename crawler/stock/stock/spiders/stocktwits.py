from ..items import StockItem

import scrapy
import json

BASE_URL = 'https://api.stocktwits.com/api/2/streams/symbol/TSLA.json?max='


class TSLASpider(scrapy.Spider):
    name = "TSLA"

    def __init__(self):
        self.count = 0
        self.data_limit = 100

    def start_requests(self):
        # TODO keep crawl use message ID after max
        # https://api.stocktwits.com/api/2/streams/symbol/TSLA.json?filter=top&limit=20&max=441700147
        urls = [
            'https://api.stocktwits.com/api/2/streams/symbol/TSLA.json'
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        items = StockItem()

        data = json.loads(response.text)
        messages = data['messages']
        self.messages += messages

        for message in messages:
            sentiment = message['entities']['sentiment']
            if sentiment:
                sentiment = sentiment['basic']

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
