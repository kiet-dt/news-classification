# Scrapy settings for news_crawler project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     https://docs.scrapy.org/en/latest/topics/settings.html
#     https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
#     https://docs.scrapy.org/en/latest/topics/spider-middleware.html

BOT_NAME = 'news_crawler'

SPIDER_MODULES = ['src.spiders']
NEWSPIDER_MODULE = 'src.spiders'

# Obey robots.txt rules
ROBOTSTXT_OBEY = False  # Tắt để tránh bị chặn bởi robots.txt

# Configure a delay for requests for the same website (default: 0)
# See https://docs.scrapy.org/en/latest/topics/settings.html#download-delay
# See also autothrottle settings and docs
DOWNLOAD_DELAY = 1.5  # Tăng lên 1.5 giây để tránh bị rate limit (428)
# The download delay setting will honor only one of:
RANDOMIZE_DOWNLOAD_DELAY = True

# Configure maximum concurrent requests performed by Scrapy (default: 16)
CONCURRENT_REQUESTS = 8  # Giảm xuống 8 để tránh bị chặn

# Configure a delay for requests for the same website (default: 0)
# See https://docs.scrapy.org/en/latest/topics/autothrottle.html
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 1.0  # Tăng delay ban đầu
AUTOTHROTTLE_MAX_DELAY = 10  # Tăng max delay để xử lý rate limit
AUTOTHROTTLE_TARGET_CONCURRENCY = 4.0  # Giảm xuống 4 để chậm hơn, tránh bị chặn
# Enable showing throttling stats for every response received:
AUTOTHROTTLE_DEBUG = False

# Enable and configure HTTP caching (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html#httpcache-middleware-settings
HTTPCACHE_ENABLED = False  # Tắt cache để test pagination
HTTPCACHE_EXPIRATION_SECS = 3600
HTTPCACHE_DIR = 'httpcache'
HTTPCACHE_IGNORE_HTTP_CODES = []

# User agent - giả lập trình duyệt thật
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

# Default request headers
DEFAULT_REQUEST_HEADERS = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
    'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Cache-Control': 'max-age=0',
}

# Enable and configure the AutoThrottle extension (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/autothrottle.html

# Enable or disable spider middlewares
# See https://docs.scrapy.org/en/latest/topics/spider-middleware.html
#SPIDER_MIDDLEWARES = {
#    'news_crawler.middlewares.NewsCrawlerSpiderMiddleware': 543,
#}

# Enable or disable downloader middlewares
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
#DOWNLOADER_MIDDLEWARES = {
#    'news_crawler.middlewares.NewsCrawlerDownloaderMiddleware': 543,
#}

# Enable or disable extensions
# See https://docs.scrapy.org/en/latest/topics/extensions.html
#EXTENSIONS = {
#    'scrapy.extensions.telnet.TelnetConsole': None,
#}

# Configure item pipelines
# See https://docs.scrapy.org/en/latest/topics/item-pipeline.html
ITEM_PIPELINES = {
    'src.pipelines.NewsCrawlerPipeline': 300,
}

# Retry settings - retry khi gặp lỗi 428 (Too Many Requests)
RETRY_ENABLED = True
RETRY_TIMES = 5  # Retry tối đa 5 lần
RETRY_HTTP_CODES = [500, 502, 503, 504, 408, 429, 428]  # Thêm 428 vào danh sách retry

# HTTP Error handling - cho phép xử lý 428
HTTPERROR_ALLOWED_CODES = [428]  # Cho phép xử lý 428 thay vì bỏ qua

# Set settings whose default value is deprecated to a future-proof value
REQUEST_FINGERPRINTER_IMPLEMENTATION = '2.7'
TWISTED_REACTOR = 'twisted.internet.asyncioreactor.AsyncioSelectorReactor'

