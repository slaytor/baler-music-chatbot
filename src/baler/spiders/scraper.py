import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import scrapy


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

    def __init__(self, start_page=1, max_pages=None, previous_file=None, url_file=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.url_file = url_file

        # url_file mode: skip listing page crawl entirely
        if self.url_file:
            self.logger.info(f"url_file mode: loading URLs from {self.url_file}")
            url_path = Path(self.url_file)
            if not url_path.exists():
                raise FileNotFoundError(f"url_file not found: {self.url_file}")
            self.direct_urls = [u.strip() for u in url_path.read_text().splitlines() if u.strip()]
            self.logger.info(f"Loaded {len(self.direct_urls):,} URLs to scrape")
            return

        # Normal listing-page crawl mode
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
                with open(previous_file) as f:
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
        # url_file mode: send each URL directly to parse_review (no playwright needed)
        if self.url_file:
            for url in self.direct_urls:
                yield scrapy.Request(url=url, callback=self.parse_review)
            return

        # Normal mode: crawl listing pages with playwright
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
            href = link.attrib['href']
            # Skip the base listing URL and bare /reviews/albums/ links without a slug
            if not re.match(r'^/reviews/albums/[^/?]+', href):
                continue

            review_url = response.urljoin(href)

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
        author = response.css('a[href*="/staff/"]::text').get()
        album_cover_url = response.css('img[loading="eager"]::attr(src)').get()

        # Score, BNM flag, and release year are JS-hydrated and wrong in SSR HTML.
        # Parse them from the embedded JSON data instead.
        score = 0.0
        is_bnm = False
        release_year = "N/A"

        rating_match = re.search(
            r'"musicRating":\{"isBestNewMusic":(true|false),"isBestNewReissue":(true|false),"score":(\d+\.?\d*)\}',
            response.text,
        )
        if rating_match:
            is_bnm = rating_match.group(1) == 'true'
            score = float(rating_match.group(3))

        year_match = re.search(r'"releaseYear":"(\d{4})"', response.text)
        if year_match:
            release_year = year_match.group(1)

        paragraphs = response.css('div[class*="body__inner-container"] p::text').getall()
        review_text = "\n".join(paragraphs)

        yield ReviewItem(
            artist=artist_name.strip() if artist_name else "N/A",
            album_title=album_title.strip() if album_title else "N/A",
            score=score,
            is_best_new_music=is_bnm,
            review_url=response.url,
            review_text=review_text.strip(),
            author=author.strip() if author else "N/A",
            release_year=release_year,
            album_cover_url=album_cover_url if album_cover_url else "N/A",
        )
