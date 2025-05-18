import asyncio
import json
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

class NewsScraper:
    def __init__(self, base_url, max_articles=30, keywords=None):
        self.base_url = base_url
        self.domain = urlparse(base_url).netloc
        self.visited = set()
        self.scraped_data = []
        self.max_articles = max_articles
        self.keywords = keywords or []

    def is_valid_url(self, url):
        parsed = urlparse(url)
        return parsed.netloc == self.domain and url not in self.visited and url.startswith('http')

    def extract_links(self, html, base_url):
        soup = BeautifulSoup(html, 'html.parser')
        links = set()
        for a_tag in soup.find_all('a', href=True):
            full_url = urljoin(base_url, a_tag['href'])
            if self.is_valid_url(full_url):
                links.add(full_url)
        return links

    def parse_article(self, url, html):
        soup = BeautifulSoup(html, 'html.parser')
        title = soup.find('h1')
        content_tags = soup.find_all('p')
        time_tag = soup.find('time')
        content = ' '.join([p.get_text(strip=True) for p in content_tags])
        timestamp = time_tag.get("datetime") if time_tag else ""

        # Keyword/topic filtering
        if self.keywords:
            lower_content = content.lower()
            if not any(kw.lower() in lower_content for kw in self.keywords):
                return None

        if title and content and len(content) > 100:
            return {
                'url': url,
                'title': title.get_text(strip=True),
                'content': content,
                'timestamp': timestamp
            }
        return None

    async def crawl(self, start_url):
        config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
        browser = BrowserConfig(headless=True)

        async with AsyncWebCrawler(config=browser) as crawler:
            queue = [start_url]

            while queue and len(self.scraped_data) < self.max_articles:
                url = queue.pop(0)
                if url in self.visited:
                    continue
                self.visited.add(url)

                try:
                    result = await crawler.arun(url, config)
                    if result.success:
                        article = self.parse_article(url, result.html)
                        if article:
                            self.scraped_data.append(article)
                            print(f"\n[+] Scraped: {url}")

                        links = self.extract_links(result.html, url)
                        for link in links:
                            if link not in self.visited:
                                queue.append(link)
                except Exception as e:
                    print(f"[!] Error: {e} on {url}")

    def save_to_json(self, path='scraped_news.json'):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.scraped_data, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    target_url = input("\nEnter the news website URL to scrape: ")
    keywords = input("\nEnter comma-separated keywords for filtering (or press Enter to skip): ").strip()
    keyword_list = [k.strip() for k in keywords.split(',')] if keywords else []

    scraper = NewsScraper(base_url=target_url, max_articles = 50, keywords=keyword_list)
    asyncio.run(scraper.crawl(scraper.base_url))
    scraper.save_to_json()
    print(f"\n✅ Saved {len(scraper.scraped_data)} articles to scraped_news.json\n")
