import scrapy
from dataclasses import dataclass, field
from pathlib import Path
import json
import re

@dataclass
class ReviewItem:
    artist: str
    album_title: str
    score: float
    is_best_new_music: bool
    review_url: str
    review_text: str
    author: str
    release_year: str
    album_cover_url: str = field(default="N/A")

class PitchforkSpider(scrapy.Spider):
    name = "pitchfork_reviews"

    def __init__(self, start_page=1, max_pages=None, previous_file=None, *args, **kwargs):
        super(PitchforkSpider, self).__init__(*args, **kwargs)
        self.start_page = int(start_page)
        self.current_page = self.start_page
        self.max_pages_to_crawl = int(max_pages) if max_pages else None
        self.pages_crawled = 0

        self.start_urls = [f'https://pitchfork.com/reviews/albums/?page={self.start_page}']
        self.seen_urls = set()
        self.stop_scraping = False

        if previous_file and Path(previous_file).exists():
            self.logger.info(f"Loading previously seen URLs from {previous_file}")
            try:
                with open(previous_file, 'r') as f:
                    for line in f:
                        try:
                            review = json.loads(line)
                            if 'review_url' in review:
                                self.seen_urls.add(review['review_url'])
                        except json.JSONDecodeError:
                            self.logger.warning(f"Could not parse line: {line.strip()}")
                self.logger.info(f"Loaded {len(self.seen_urls)} previously seen URLs.")
            except Exception as e:
                self.logger.error(f"Error reading previous file: {e}")
        else:
            self.logger.info("No previous scrape file provided or found.")

    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(
                url=url,
                meta=dict(
                    playwright=True,
                    playwright_include_page=True,
                )
            )

    async def parse(self, response):
        page = response.meta["playwright_page"]
        self.pages_crawled += 1
        found_seen_url_on_page = False

        review_links = response.css('a[href^="/reviews/albums/"]')

        for link in review_links:
            review_url = response.urljoin(link.attrib['href'])

            if self.seen_urls and review_url in self.seen_urls:
                found_seen_url_on_page = True
                self.logger.debug(f"Encountered previously seen review: {review_url}. Skipping.")
                continue

            self.logger.info(f"Found new review: {review_url}")
            yield response.follow(link, self.parse_review)

        should_stop_pagination = found_seen_url_on_page

        if self.max_pages_to_crawl is not None and self.pages_crawled >= self.max_pages_to_crawl:
            self.logger.info(f"Stopping scrape: Reached max_pages limit ({self.max_pages_to_crawl}).")
            should_stop_pagination = True

        if not should_stop_pagination:
            next_page_link = response.xpath('//span[contains(text(), "Next Page")]/parent::a/@href').get()
            if next_page_link:
                match = re.search(r'page=(\d+)', next_page_link)
                next_page_num = int(match.group(1)) if match else 'unknown'
                self.logger.info(f"Following link to page {next_page_num}")

                yield scrapy.Request(
                    url=response.urljoin(next_page_link),
                    callback=self.parse,
                    meta=dict(
                        playwright=True,
                        playwright_include_page=True,
                    )
                )
            else:
                self.logger.info("Reached the last page of reviews.")
        else:
             if found_seen_url_on_page:
                 self.logger.info("Stopping pagination because previously scraped reviews were found on this page.")

        await page.close()

    def parse_review(self, response):
        artist_name = response.css('div[class*="SplitScreenContentHeaderArtist"] a::text').get()
        if not artist_name:
            artist_name = response.css('div[class*="SplitScreenContentHeaderArtist"]::text').get()

        album_title = response.xpath('string(//h1[@data-testid="ContentHeaderHed"])').get()
        score_str = response.css('p[class*="Rating-"]::text').get()
        best_new_music_tag = response.css('p[class*="BestNewMusicText-"]').get()
        is_bnm = best_new_music_tag is not None
        author = response.css('a[href*="/staff/"]::text').get()
        
        # --- FINAL, CORRECT SELECTOR ---
        album_cover_url = response.css('img[loading="eager"]::attr(src)').get()

        release_year = response.css('time::text').get()
        if not release_year:
            info_items = response.css('div[class*="InfoSliceContent"] li::text').getall()
            for item in info_items:
                if item and (item.strip().startswith('19') or item.strip().startswith('20')):
                    release_year = item.strip()
                    break

        paragraphs = response.css('div[class*="body__inner-container"] p::text').getall()
        review_text = "\n".join(paragraphs)
        score = float(score_str) if score_str else 0.0

        yield ReviewItem(
            artist=artist_name.strip() if artist_name else "N/A",
            album_title=album_title.strip() if album_title else "N/A",
            score=score,
            is_best_new_music=is_bnm,
            review_url=response.url,
            review_text=review_text.strip(),
            author=author.strip() if author else "N/A",
            release_year=release_year if release_year else "N/A",
            album_cover_url=album_cover_url if album_cover_url else "N/A"
        )
