# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html

import os
from datetime import datetime
from itemadapter import ItemAdapter
import pandas as pd


class NewsCrawlerPipeline:
    def __init__(self):
        # Đường dẫn data từ root project
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'raw')
        os.makedirs(self.data_dir, exist_ok=True)
        self.items = []
    
    def open_spider(self, spider):
        """Called when the spider is opened"""
        spider.logger.info('Pipeline opened')
    
    def close_spider(self, spider):
        """Called when the spider is closed"""
        if self.items:
            # Tạo DataFrame
            df = pd.DataFrame(self.items)
            
            # Loại bỏ duplicate dựa trên URL
            initial_count = len(df)
            df = df.drop_duplicates(subset=['url'], keep='first')
            removed = initial_count - len(df)
            
            if removed > 0:
                spider.logger.info(f'Đã loại bỏ {removed} bài trùng lặp')
            
            # Lưu vào Parquet
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(self.data_dir, f'dantri_{timestamp}.parquet')
            
            df.to_parquet(
                filename,
                index=False,
                compression='snappy',
                engine='pyarrow'
            )
            
            file_size_mb = os.path.getsize(filename) / (1024 * 1024)
            spider.logger.info(f'Đã lưu {len(df)} mẫu vào {filename} ({file_size_mb:.2f} MB)')
            
            # Thống kê
            if 'category' in df.columns:
                spider.logger.info(f'\nPhân bố theo category:')
                for cat, count in df['category'].value_counts().items():
                    spider.logger.info(f'  {cat}: {count} mẫu')
            
            self.items = []
    
    def process_item(self, item, spider):
        """Process each item"""
        adapter = ItemAdapter(item)
        
        # Validate và clean data
        if not adapter.get('title') or not adapter.get('content'):
            spider.logger.warning(f'Bỏ qua item thiếu dữ liệu: {adapter.get("url")}')
            return item
        
        # Chuyển đổi thành dict
        item_dict = dict(adapter)
        self.items.append(item_dict)
        
        spider.logger.info(f'Đã crawl: {adapter.get("title")[:50]}...')
        return item

