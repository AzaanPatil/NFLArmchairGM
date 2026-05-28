import pandas as pd

from scraping.data_scraping_drafts import scrape_draft
from Draft_IQ.systems.compare_drafts import compare_drafts
from systems.accuracy_metrics import mean_error


def main():
    print("Starting DraftIQ pipeline...")

    # Step 1: Scrape actual draft
    real_df = scrape_draft(2020)
    print("Scraped real draft")

    # Step 2: Load mock draft (manual for now)
    mock_df = pd.read_csv("data/raw/mock_2020.csv")
    print("Loaded mock draft")

    # Step 3: Compare
    merged_df = compare_drafts(mock_df, real_df)
    print("Merged drafts")

    # Step 4: Compute metrics
    error = mean_error(merged_df)
    print(f"Average Draft Error: {error}")


if __name__ == "__main__":
    main()