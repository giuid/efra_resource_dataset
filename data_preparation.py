#%%
import os
import json
import pandas as pd
from tqdm import tqdm
from typing import Callable, List, Dict
from resources.data import HTMLSplitter


def load_data(data_path: str) -> pd.DataFrame:
    """
    Loads JSON data into a Pandas DataFrame.

    Args:
        data_path (str): Path to the JSON file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    with open(data_path, 'r') as file:
        data = json.load(file)
    return pd.DataFrame(data)


def parse_htmls(
    html_dir: str, 
    output_dir: str, 
    html_names: List[str], 
    batch_size: int = 500
):
    """
    Parses HTML files into tokenized paragraphs and saves them as JSON.

    Args:
        html_dir (str): Directory containing HTML files.
        output_dir (str): Directory to save parsed JSON files.
        html_names (List[str]): List of HTML filenames to parse.
        batch_size (int): Number of HTML files to process per batch.
    """
    os.makedirs(output_dir, exist_ok=True)

    for batch_start in tqdm(range(0, len(html_names), batch_size)):
        parsed = {}
        batch_htmls = html_names[batch_start:batch_start + batch_size]

        for name in batch_htmls:
            html_path = os.path.join(html_dir, f"{name}.html")
            if os.path.exists(html_path):
                with open(html_path, 'r') as file:
                    html_content = file.read()

                paragraphs = HTMLSplitter()(html_content)
                parsed[name] = paragraphs

        output_path = os.path.join(output_dir, f"parsed_htmls_{batch_start}.json")
        with open(output_path, 'w') as outfile:
            json.dump(parsed, outfile, indent=4)


def main():
    # Paths and parameters
    data_path = "/path/to/efra.json"
    html_dir = "/path/to/htmls/english"
    output_dir = "/path/to/parsed_htmls"
    dataset_filter = "manual_summary"

    # Load and process data
    print("Loading data...")
    df = load_data(data_path)

    print("Filtering and removing duplicates...")
    filtered_df = df[df['dataset'] == dataset_filter]
    no_duplicates_df = filtered_df.drop_duplicates(subset="english_summary", keep="first")
    htmls_with_summaries = no_duplicates_df.post_id.tolist()
    html_files = [name for name in os.listdir(html_dir) if name.replace('.html', '') in htmls_with_summaries]

    print("Parsing HTML files...")
    parse_htmls(html_dir, output_dir, htmls_with_summaries)

    print("Parsing complete.")


if __name__ == "__main__":
    main()