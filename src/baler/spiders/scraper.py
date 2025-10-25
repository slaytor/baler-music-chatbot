import scrapy
from dataclasses import dataclass
from pathlib import Path


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

    # --- NEW: Allow specifying a start page via command line ---
    def __init__(self, start_page=1, *args, **kwargs):
        super(PitchforkSpider, self).__init__(*args, **kwargs)
        self.start_page = int(start_page)
        # Construct the initial URL based on the start_page
        self.start_urls = [f'https://pitchfork.com/reviews/albums/?page={self.start_page}']

    # --- Changed back to start_requests for argument handling ---
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
        for link in review_links:
            yield response.follow(link, self.parse_review)

        # Using XPath for the Next Page link for robustness
        next_page_link = response.xpath('//span[contains(text(), "Next Page")]/parent::a/@href').get()
        if next_page_link:
            yield scrapy.Request(
                url=response.urljoin(next_page_link),
                callback=self.parse, # Use self.parse for subsequent pages
                meta=dict(
                    playwright=True,
                    playwright_include_page=True,
                )
            )

        await page.close()

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
                    release_year = item.strip() # Ensure stripping whitespace
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
            release_year=release_year if release_year else "N/A" # Handle potential None
        )
