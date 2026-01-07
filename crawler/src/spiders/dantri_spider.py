import scrapy
from scrapy.http import Request
from urllib.parse import urljoin
from src.items import NewsItem
import re


class DantriSpider(scrapy.Spider):
    name = 'dantri'
    allowed_domains = ['dantri.com.vn']
    
    CATEGORY_MAPPING = {
        'the-gioi': 'Thế giới',
        'thoi-su': 'Thời sự',
        'phap-luat': 'Pháp luật',
        'suc-khoe': 'Sức khỏe',
        'doi-song': 'Đời sống',
        'du-lich': 'Du lịch',
        'kinh-doanh': 'Kinh doanh',
        'bat-dong-san': 'Bất động sản',
        'the-thao': 'Thể thao',
        'giai-tri': 'Giải trí',
        'giao-duc': 'Giáo dục',
        'cong-nghe': 'Công nghệ',
    }
    
    def __init__(self, categories=None, max_pages=25, *args, **kwargs):
        super(DantriSpider, self).__init__(*args, **kwargs)
        
        if categories:
            self.categories = categories.split(',')
        else:
            self.categories = [
                'the-gioi',      # Thế giới
                'thoi-su',       # Thời sự
                'phap-luat',     # Pháp luật
                'suc-khoe',      # Sức khỏe
                'doi-song',      # Đời sống
                'du-lich',       # Du lịch
                'kinh-doanh',    # Kinh doanh
                'bat-dong-san',  # Bất động sản
                'the-thao',      # Thể thao
                'giai-tri',      # Giải trí
                'giao-duc',      # Giáo dục
                'cong-nghe',     # Công nghệ
            ]
        
        self.max_pages = int(max_pages) if max_pages else 25
        self.stats = {
            'articles_found': 0,
            'articles_saved': 0,
            'articles_skipped': 0,
            'pages_crawled': 0,
            'skip_reasons': {}
        }
        self.logger.info(f'Crawl {len(self.categories)} categories, tối đa {self.max_pages} trang mỗi category')
    
    def closed(self, reason):
        """Được gọi khi spider đóng"""
        self.logger.info('=' * 60)
        self.logger.info('THỐNG KÊ CRAWL')
        self.logger.info('=' * 60)
        self.logger.info(f'Số trang đã crawl: {self.stats["pages_crawled"]}')
        self.logger.info(f'Số bài viết tìm thấy: {self.stats["articles_found"]}')
        self.logger.info(f'Số bài viết đã lưu: {self.stats["articles_saved"]}')
        self.logger.info(f'Số bài viết bị bỏ qua: {self.stats["articles_skipped"]}')
        
        if self.stats['skip_reasons']:
            self.logger.info('\nLý do bỏ qua:')
            for reason, count in self.stats['skip_reasons'].items():
                self.logger.info(f'  {reason}: {count} bài')
        
        if self.stats['articles_found'] > 0:
            success_rate = (self.stats['articles_saved'] / self.stats['articles_found']) * 100
            self.logger.info(f'\nTỷ lệ thành công: {success_rate:.1f}%')
        
        self.logger.info('=' * 60)
    
    def start_requests(self):
        """Bắt đầu crawl từ các category"""
        for category in self.categories:
            # URL category của Dân trí (có .htm ở cuối)
            category_url = f'https://dantri.com.vn/{category}.htm'
            yield Request(
                url=category_url,
                callback=self.parse_category,
                meta={'category': category, 'page': 1}
            )
    
    def parse_category(self, response):
        """Parse trang danh sách bài viết của category"""
        category = response.meta['category']
        page = response.meta['page']
        
        # Kiểm tra response status
        if response.status != 200:
            self.logger.warning(f'  Response status {response.status} cho URL: {response.url}')
            return
        
        self.logger.info(f'Đang crawl category {category}, trang {page} - URL: {response.url}')
        self.stats['pages_crawled'] += 1
        
        # Tìm các link bài viết
        article_links = response.css('article.article-item a::attr(href)').getall()
        self.logger.info(f'  Tìm thấy {len(article_links)} links trên trang {page}')
        
        # Nếu không tìm thấy link nào, có thể URL không đúng
        if len(article_links) == 0:
            self.logger.warning(f'  KHÔNG TÌM THẤY LINK NÀO trên trang {page}! Có thể URL không đúng: {response.url}')
            # Thử tìm các selector khác để debug
            all_links = response.css('a::attr(href)').getall()
            self.logger.warning(f'  Tổng số link trên trang: {len(all_links)}')
        
        # Loại bỏ duplicate và lọc URLs hợp lệ
        seen_urls = set()
        valid_links = 0
        for link in article_links:
            if link and '.htm' in link:
                # Chuẩn hóa URL
                if link.startswith('http'):
                    full_url = link
                else:
                    full_url = urljoin(response.url, link)
                
                # Loại bỏ duplicate
                if full_url not in seen_urls and 'dantri.com.vn' in full_url:
                    seen_urls.add(full_url)
                    valid_links += 1
                    self.stats['articles_found'] += 1
                    yield Request(
                        url=full_url,
                        callback=self.parse_article,
                        meta={'category': category}
                    )
        
        self.logger.info(f'  Đã tạo {valid_links} requests cho bài viết')
        
        # Crawl trang tiếp theo
        self.logger.info(f'  Kiểm tra pagination: page={page}, max_pages={self.max_pages}')
        if page < self.max_pages:
            # Tìm link trang tiếp theo - thử nhiều selector
            next_page = None
            
            # Thử tìm nút "next" với class next
            next_page = response.css('a.next::attr(href), a.page-item.next::attr(href)').get()
            
            if not next_page:
                # Thử tìm trong pagination với class next
                next_page = response.css('div.pagination a.next::attr(href)').get()
            
            if not next_page:
                # Thử tìm nút phân trang khác với rel="next"
                next_page = response.css('a[rel="next"]::attr(href)').get()
            
            if not next_page:
                # Thử tìm các link phân trang khác (có thể có số trang)
                pagination_links = response.css('div.pagination a::attr(href), nav.pagination a::attr(href)').getall()
                if pagination_links:
                    self.logger.info(f'  Tìm thấy {len(pagination_links)} links phân trang (không có next)')
                    # Tìm link có số trang lớn hơn trang hiện tại
                    for link in pagination_links:
                        if link and '/trang-' in link:
                            # Extract số trang từ URL
                            match = re.search(r'/trang-(\d+)', link)
                            if match:
                                link_page = int(match.group(1))
                                if link_page == page + 1:
                                    next_page = link
                                    self.logger.info(f'  Tìm thấy link trang {link_page}: {link}')
                                    break
            
            if next_page:
                next_url = urljoin(response.url, next_page)
                self.logger.info(f'  [PAGINATION] Tìm thấy next page link: {next_url}')
                yield Request(
                    url=next_url,
                    callback=self.parse_category,
                    meta={'category': category, 'page': page + 1},
                    dont_filter=True,
                    errback=self.handle_error,
                    priority=1  # Ưu tiên cao hơn để crawl nhanh hơn
                )
            else:
                # Nếu không có nút next, tự tạo URL theo pattern
                if '/trang-' in response.url:
                    # URL đã có số trang, tăng lên
                    next_url = re.sub(r'/trang-(\d+)', f'/trang-{page + 1}', response.url)
                else:
                    # Thêm số trang vào URL (thay .htm bằng /trang-X.htm)
                    base_url = response.url.replace('.htm', '')
                    next_url = f"{base_url}/trang-{page + 1}.htm"
                
                self.logger.info(f'  [PAGINATION] Tự tạo next page URL: {next_url} (trang {page + 1}/{self.max_pages})')
                yield Request(
                    url=next_url,
                    callback=self.parse_category,
                    meta={'category': category, 'page': page + 1},
                    dont_filter=True,
                    errback=self.handle_error,
                    priority=1  # Ưu tiên cao hơn
                )
        else:
            self.logger.info(f'  Đã đạt max_pages ({self.max_pages}), dừng crawl category {category}')
    
    def handle_error(self, failure):
        """Xử lý lỗi khi request thất bại"""
        request = failure.request
        category = request.meta.get('category', 'unknown')
        page = request.meta.get('page', 'unknown')
        self.logger.error(f'Lỗi khi crawl {category} trang {page}: {failure.value}')
        self.logger.error(f'URL: {request.url}')
    
    def parse_article(self, response):
        """Parse nội dung một bài viết"""
        category = response.meta.get('category', '')
        
        # Map category từ URL sang tên chuẩn
        category_name = self.CATEGORY_MAPPING.get(category, category.title())
        
        # Extract title - thử nhiều format
        title = response.css('h1.title-page.detail::text').get()
        if not title:
            # Thử e-magazine format với class e-magazine__title
            title = response.css('h1.e-magazine__title::text').get()
        if not title:
            # Thử e-magazine format khác (h1 trong article.e-magazine)
            title = response.css('article.e-magazine h1::text').get()
        if not title:
            # Fallback: lấy h1 đầu tiên trong main hoặc article
            title = response.css('main h1::text, article h1::text').get()
        
        if title:
            title = title.strip()
        
        # Extract content - thử cả 2 loại bài viết
        content_paragraphs = []
        
        # Thử bài viết thông thường
        content_div = response.css('div.singular-content')
        if content_div:
            paragraphs = content_div.css('p')
            for p in paragraphs:
                text = p.css('::text').getall()
                text_clean = ' '.join([t.strip() for t in text if t.strip()])
                if text_clean:
                    content_paragraphs.append(text_clean)
        
        # Nếu không có, thử e-magazine format
        if not content_paragraphs:
            content_div = response.css('div.e-magazine__body')
            if content_div:
                paragraphs = content_div.css('p')
                for p in paragraphs:
                    text = p.css('::text').getall()
                    text_clean = ' '.join([t.strip() for t in text if t.strip()])
                    if text_clean:
                        content_paragraphs.append(text_clean)
        
        content = ' '.join([p.strip() for p in content_paragraphs if p.strip()])
        
        # Chỉ yield item nếu có đủ title và content
        if title and content and len(content) > 50:  # Content phải có ít nhất 50 ký tự
            self.stats['articles_saved'] += 1
            item = NewsItem()
            item['title'] = title
            item['content'] = content
            item['category'] = category_name
            item['url'] = response.url
            item['source'] = 'dantri'
            
            yield item
        else:
            self.stats['articles_skipped'] += 1
            reason = []
            if not title:
                reason.append('no_title')
            if not content:
                reason.append('no_content')
            elif len(content) <= 50:
                reason.append(f'content_too_short_{len(content)}')
            
            reason_str = '_'.join(reason) if reason else 'unknown'
            self.stats['skip_reasons'][reason_str] = self.stats['skip_reasons'].get(reason_str, 0) + 1
            
            self.logger.warning(f'Bỏ qua bài viết: {response.url} - Lý do: {reason_str}')

