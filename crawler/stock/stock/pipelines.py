# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html

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
        self.file = open(f'data/{datetime.now().strftime("%Y-%M-%d-%H-%M-%S")}.json', 'wb')
        self.exporter = JsonItemExporter(self.file)
        self.exporter.start_exporting()

    def close_spider(self, spider):
        self.exporter.finish_exporting()
        self.file.close()

    def process_item(self, item, spider):
        if item['sentiment']:
            item['sentiment'] = item['sentiment']['basic']

        # if item['body'][0] == item['body'][-1] == '"':
        #     item['body'] = item['body'][1:-1]

        adapter = ItemAdapter(item)
        if adapter['message_id'] in self.ids_seen:
            raise DropItem(f"Duplicate item found: {item!r}")
        else:
            self.ids_seen.add(adapter['message_id'])
            self.exporter.export_item(item)
            return item
