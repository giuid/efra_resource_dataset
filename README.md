# Legal Document Summarization Script

This script processes legal documents, splits them into paragraphs, tokenizes the content, and generates summaries using a pre-trained transformer model. It leverages Hugging Face tools, custom HTML splitting logic, and a summarization pipeline. Below is a breakdown of its functionality:

## Features

- **HTML Paragraph Splitting**: Extracts paragraphs from HTML documents and tokenizes them.
- **Input Preparation**: Converts textual data into tokenized inputs for the model, including user instructions and assistant prompts.
- **Summarization Generation**: Utilizes a transformer-based model to generate concise summaries of legal documents.
- **Customizable Parameters**: Allows adjustment of tokenization window sizes, input lengths, and beam search width for summarization.

## Requirements

- Python 3.8 or higher
- Required Python libraries:
  - `pandas`
  - `datasets`
  - `transformers`
  - `torch`
  - `huggingface_hub`
  - Custom `resources` package (for `HTMLSplitter` and tokenization logic)

## Setup

1. **Install dependencies**:
   ```bash
   pip install pandas datasets transformers torch huggingface_hub
2. **Set up environment variables:
	-	TRANSFORMERS_CACHE: Path to the Hugging Face transformers cache.
	-	CUDA_VISIBLE_DEVICES: List of GPUs to use for processing.
3. **Authentication:
Log in to Hugging Face Hub using your API token:
```
from huggingface_hub import login
login(token='your_huggingface_token')
```

## Usage

1. **Prepare HTML Files**:  
   Place HTML documents in `htmls_path/english` and `htmls_path/original`.

2. **Run the Script**:  
   Execute the script to:  
   - Parse HTML documents and tokenize content.  
   - Prepare tokenized inputs with user instructions and prompts.  
   - Generate summaries using a specified transformer model.

3. **Clean Summaries**:  
   Use the `clean_output` function to refine and extract clean summaries from model outputs.

---

## Key Functions

- **`split_text_in_paragraphs`**:  
  Splits HTML content into paragraphs and tokenizes them.

- **`prepare_input`**:  
  Prepares input prompts and tokenized tensors for the model.

- **`generate_summaries`**:  
  Generates summaries using a transformer model with customizable parameters like `max_output_length` and `num_beams`.

- **`clean_output`**:  
  Cleans and extracts meaningful summary content from model-generated outputs.

---

## Model Configuration

- Example transformer models used:  
  - `meta-llama/Llama-3.1-405B-Instruct`  
  - `meta-llama/Llama-3.3-70B-Instruct`  
  - `meta-llama/Llama-3.2-3B-Instruct`

---

## Output

The script produces:  
- Summaries of legal documents in a refined format.  
- A dictionary containing processed HTML content and metadata.

---

## Notes

- Ensure you have the necessary GPU resources for model inference.  
- Adjust parameters like `max_input_length` and `num_beams` based on your hardware capabilities and summarization needs.
