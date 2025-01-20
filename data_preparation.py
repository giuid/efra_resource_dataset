#%%
import os
import json
import pandas as pd
from tqdm import tqdm
from typing import Callable, List, Dict
from resources.data import word_tokenize, SENT_TOKENIZER, HTMLSplitter


def split_text_in_paragraphs(
    html: str, 
    window: int = 512, 
    tokenize: Callable[[str], List[str]] = word_tokenize
) -> Dict[str, List]:
    """
    Splits HTML text into paragraphs and tokenizes them.

    Args:
        html (str): The HTML content to process.
        window (int): Maximum number of tokens per part.
        tokenize (Callable[[str], List[str]]): Function to tokenize text.

    Returns:
        dict: A dictionary with parts, paragraphs, tokens, and texts.
    """
    parts, paragraphs, tokens, texts = [], [], [], []
    parser = HTMLSplitter()
    parser.feed(html)
    html = parser.get_data()

    part, cursor, remaining_tokens = 0, 0, -1
    for paragraph_idx, paragraph_text in enumerate(html.split('\n\n')):
        for start, end in SENT_TOKENIZER.span_tokenize(paragraph_text):
            sentence_tokens = tokenize(paragraph_text[start:end + 1])
            remaining_tokens -= len(sentence_tokens) - 1

            if remaining_tokens <= 0:
                parts.append(str(part))
                paragraphs.append(str(paragraph_idx))
                tokens.append(sentence_tokens[:window])
                texts.append(paragraph_text[start:end + 1])
                remaining_tokens = window - len(sentence_tokens) + 1
                part += 1
                cursor = start
            else:
                tokens[-1].extend(sentence_tokens[:window])
                texts[-1] = paragraph_text[cursor:end + 1]

        remaining_tokens = -1

    return {'part': parts, 'paragraph': paragraphs, 'tokens': tokens, 'texts': texts}


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

                paragraphs = split_text_in_paragraphs(html_content)
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