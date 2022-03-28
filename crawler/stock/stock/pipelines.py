# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html
import re

from html import unescape
from datetime import datetime
# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem
from scrapy.exporters import JsonItemExporter



class StockPipeline:

    def __init__(self):
        self.ids_seen = set()
        self.file = None

    def open_spider(self, spider):
        self.file = open(f'data/{datetime.now().strftime("%Y-%m-%dT%H%M%S")}.json', 'wb')
        self.exporter = JsonItemExporter(self.file, encoding='utf-8')
        self.exporter.start_exporting()

    def close_spider(self, spider):
        self.exporter.finish_exporting()
        self.file.close()

    def process_item(self, item, spider):
        if item['sentiment']:
            item['sentiment'] = item['sentiment']['basic']

        item['body'] = unescape(item['body'])

        # pattern = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
        # item['body']= re.sub(pattern, 'HTTPURL',item['body'])
        if len(item['body']) < 10:
            raise DropItem(f"Item too short: {item!r}")

        adapter = ItemAdapter(item)
        if adapter['message_id'] in self.ids_seen:
            raise DropItem(f"Duplicate item found: {item!r}")
        else:
            self.ids_seen.add(adapter['message_id'])
            self.exporter.export_item(item)
            return item
