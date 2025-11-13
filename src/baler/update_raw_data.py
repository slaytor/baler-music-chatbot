import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from pathlib import Path
import logging
import json
import pandas as pd

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
S3_BUCKET_NAME = "baler-music-chatbot"
S3_DAILY_PREFIX = "daily_scrapes/"
LOCAL_RAW_FILE = PROJECT_ROOT / "reviews.jsonl"
PROCESSED_FILES_LOG = PROJECT_ROOT / ".processed_s3_files.log"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_processed_files() -> set:
    """Reads the log of already processed S3 files."""
    if not PROCESSED_FILES_LOG.exists():
        return set()
    try:
        with open(PROCESSED_FILES_LOG, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    except Exception as e:
        logging.error(f"Could not read {PROCESSED_FILES_LOG}: {e}")
        return set()


def log_processed_file(s3_key: str):
    """Adds a new S3 file key to the processed log."""
    try:
        with open(PROCESSED_FILES_LOG, 'a') as f:
            f.write(f"{s3_key}\n")
    except Exception as e:
        logging.error(f"Could not write to {PROCESSED_FILES_LOG}: {e}")


def get_local_seen_urls() -> set:
    """Reads the master reviews.jsonl and returns a set of all review_urls."""
    if not LOCAL_RAW_FILE.exists():
        return set()
    try:
        # Using pandas is efficient for large files
        df = pd.read_json(LOCAL_RAW_FILE, lines=True)
        if 'review_url' in df.columns:
            return set(df['review_url'])
        else:
            return set()
    except ValueError:
         logging.warning(f"{LOCAL_RAW_FILE} is empty or malformed. Starting with an empty URL set.")
         return set()
    except Exception as e:
        logging.error(f"Error reading {LOCAL_RAW_FILE}: {e}")
        return set()


def sync_s3_to_local():
    """
    Downloads new daily scrapes from S3 and appends only unique,
    new reviews to the local reviews.jsonl.
    """
    logging.info("Starting S3 sync process...")
    try:
        s3_client = boto3.client('s3')
        processed_s3_files = get_processed_files()

        # --- THE FIX: Load all URLs we already have locally ---
        local_seen_urls = get_local_seen_urls()
        logging.info(f"Loaded {len(local_seen_urls)} unique review URLs from local {LOCAL_RAW_FILE}.")

        objects = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=S3_DAILY_PREFIX)
        if 'Contents' not in objects:
            logging.info(f"No files found in S3 at prefix {S3_DAILY_PREFIX}.")
            return

        new_files_to_process = []
        for obj in objects['Contents']:
            s3_key = obj['Key']
            if obj['Size'] > 0 and s3_key not in processed_s3_files:
                new_files_to_process.append(s3_key)

        if not new_files_to_process:
            logging.info("No new S3 files found to process. Local data is up-to-date.")
            return

        logging.info(f"Found {len(new_files_to_process)} new S3 files to download...")

        total_new_reviews_appended = 0
        with open(LOCAL_RAW_FILE, 'a') as master_file:
            for s3_key in sorted(new_files_to_process):
                try:
                    local_temp_path = f"./temp_{os.path.basename(s3_key)}"
                    logging.info(f"Downloading {s3_key}...")
                    s3_client.download_file(S3_BUCKET_NAME, s3_key, local_temp_path)

                    logging.info(f"Scanning {s3_key} for new reviews...")
                    new_reviews_from_this_file = 0

                    # --- THE FIX: Check for duplicates before appending ---
                    with open(local_temp_path, 'r') as temp_f:
                        for line in temp_f:
                            try:
                                review = json.loads(line)
                                url = review.get('review_url')
                                if url and url not in local_seen_urls:
                                    master_file.write(line) # line already has newline
                                    local_seen_urls.add(url)
                                    new_reviews_from_this_file += 1
                            except json.JSONDecodeError:
                                logging.warning(f"Skipping malformed line in {s3_key}: {line.strip()}")

                    logging.info(f"Appended {new_reviews_from_this_file} new unique reviews from {s3_key}.")
                    total_new_reviews_appended += new_reviews_from_this_file

                    os.remove(local_temp_path)
                    log_processed_file(s3_key)

                except ClientError as e:
                    logging.error(f"Error downloading {s3_key}: {e}")
                except Exception as e:
                    logging.error(f"Error processing {s3_key}: {e}")

        logging.info(f"S3 sync complete. Appended a total of {total_new_reviews_appended} new reviews.")

    except NoCredentialsError:
        logging.error("AWS credentials not found. Make sure you have configured the AWS CLI.")
    except ClientError as e:
        logging.error(f"An AWS error occurred: {e}")


if __name__ == "__main__":
    sync_s3_to_local()

### Your New Workflow:

# 1.  **Run the S3 Syncer (Whenever you want to update):**
#     ```bash
#     poetry run python -m baler.update_raw_data
#     ```
#     This will now be much faster and will only append new, unique reviews.
# 2.  **Run the Knowledge Base Builder (After syncing):**
#     ```bash
#     poetry run python -m baler.create_knowledge_base
#
#
