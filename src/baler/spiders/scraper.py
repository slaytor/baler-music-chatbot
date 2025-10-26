import scrapy
from dataclasses import dataclass
from pathlib import Path
import json
from scrapy.exceptions import CloseSpider


# Remember your IDE's formatting preference for blank lines
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



class PitchforkSpider(scrapy.Spider):
    name = "pitchfork_reviews"

    # --- UPDATED: Accept 'previous_file' argument ---
    def __init__(self, start_page=1, previous_file=None, *args, **kwargs):
        super(PitchforkSpider, self).__init__(*args, **kwargs)
        self.start_page = int(start_page)
        self.start_urls = [f'https://pitchfork.com/reviews/albums/?page={self.start_page}']
        self.seen_urls = set()
        self.stop_scraping = False # Flag to stop pagination

        # --- NEW: Load URLs from the previous scrape file ---
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
                            self.logger.warning(f"Could not parse line in {previous_file}: {line.strip()}")
                self.logger.info(f"Loaded {len(self.seen_urls)} previously seen URLs.")
            except Exception as e:
                self.logger.error(f"Error reading previous file {previous_file}: {e}")
        else:
            self.logger.info("No previous scrape file provided or found. Performing full scrape.")


    # Use start_requests for argument handling
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

        review_links = response.css('a[href^="/reviews/albums/"]')
        links_processed_count = 0

        for link in review_links:
            # Construct the absolute URL
            review_url = response.urljoin(link.attrib['href'])

            # --- NEW: Check if we've seen this review before ---
            if self.seen_urls and review_url in self.seen_urls:
                self.logger.info(f"Stopping scrape: Encountered previously seen review: {review_url}")
                self.stop_scraping = True
                # Optional: Yield the seen item one last time if needed, then break
                # yield response.follow(link, self.parse_review)
                break # Stop processing links on this page

            links_processed_count += 1
            yield response.follow(link, self.parse_review)

        # --- NEW: Only paginate if the stop flag hasn't been set ---
        if not self.stop_scraping:
            next_page_link = response.xpath('//span[contains(text(), "Next Page")]/parent::a/@href').get()
            if next_page_link:
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
            self.logger.info("Stopping pagination because previously scraped reviews were found.")

        await page.close()

        # If we stopped scraping because we hit seen URLs, raise CloseSpider
        if self.stop_scraping:
             raise CloseSpider('Reached previously scraped reviews.')


    def parse_review(self, response):
        # Fallback logic for artist name
        artist_name = response.css('div[class*="SplitScreenContentHeaderArtist"] a::text').get()
        if not artist_name:
            artist_name = response.css('div[class*="SplitScreenContentHeaderArtist"]::text').get()

        album_title = response.xpath('string(//h1[@data-testid="ContentHeaderHed"])').get()
        score_str = response.css('p[class*="Rating-"]::text').get()

        best_new_music_tag = response.css('p[class*="BestNewMusicText-"]').get()
        is_bnm = best_new_music_tag is not None

        author = response.css('a[href*="/staff/"]::text').get()

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
            release_year=release_year if release_year else "N/A"
        )
