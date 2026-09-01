# pip install easyocr pillow rapidfuzz pandas

import os
import re
import json
import argparse
import pandas as pd
import easyocr
from PIL import Image
from rapidfuzz import fuzz
import numpy as np

# Initialize EasyOCR reader (loads once globally)
_ocr_reader = None

def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _ocr_reader


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"[^a-z0-9@.:/ ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def run_ocr(image_path: str) -> str:
    """
    Run OCR using EasyOCR with PIL-based image loading.

    Uses PIL to load image and passes numpy array to EasyOCR,
    bypassing cv2.imread which requires libX11/libxcb.
    Filters results by confidence >= 0.6.
    """
    reader = get_ocr_reader()
    # Load with PIL, convert to numpy array
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    # Pass array instead of path - bypasses cv2.imread
    results = reader.readtext(img_array, detail=1)
    # Filter by confidence >= 0.6
    texts = [r[1] for r in results if r and len(r) >= 2 and r[1] and r[2] >= 0.6]
    return ' '.join(texts).strip()


def extract_bigrams(text: str) -> list:
    """
    Extract unique word-level bigrams from text.
    Bigrams are more robust to OCR errors than trigrams.
    """
    text = text.lower()
    words = re.findall(r'\b[a-z0-9]{2,}\b', text)  # Min 2 chars to skip noise

    bigrams = set()
    for i in range(len(words) - 1):
        bigrams.add(f"{words[i]} {words[i+1]}")

    return list(bigrams)


def extract_trigrams(text: str) -> list:
    """
    Extract unique word-level trigrams from text.

    Trigrams are more robust to OCR errors and UI formatting differences
    compared to full sentences or single words.
    """
    text = text.lower()
    words = re.findall(r'\b[a-z0-9]{2,}\b', text)  # Min 2 chars to skip noise

    trigrams = set()
    for i in range(len(words) - 2):
        trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
        trigrams.add(trigram)

    return list(trigrams)


def is_bigram_supported(bigram: str, agent_response: str, threshold: int = 75) -> bool:
    """Check if a bigram is supported by the agent response."""
    response_n = normalize(agent_response)
    bigram_n = normalize(bigram)

    if bigram_n in response_n:
        return True

    score = fuzz.partial_ratio(bigram_n, response_n)
    return score >= threshold


def is_trigram_supported(trigram: str, agent_response: str, threshold: int = 75) -> bool:
    """Check if a trigram is supported by the agent response."""
    response_n = normalize(agent_response)
    trigram_n = normalize(trigram)

    if trigram_n in response_n:
        return True

    score = fuzz.partial_ratio(trigram_n, response_n)
    return score >= threshold


def extract_atomic_facts(text: str) -> list:
    """Backward compatibility: extract trigrams as atomic facts."""
    return extract_trigrams(text)


def is_supported(fact: str, agent_response: str, threshold: int = 75) -> bool:
    """Backward compatibility: check if trigram/fact is supported."""
    return is_trigram_supported(fact, agent_response, threshold)


def compute_factscore(agent_response: str, ocr_text: str, threshold: int = 75, ngram_type: str = "bigram"):
    """
    Compute FactScore using n-gram based matching.

    Args:
        agent_response: The source response text
        ocr_text: The OCR extracted text from UI image
        threshold: Fuzzy matching threshold (default 75)
        ngram_type: 'bigram' or 'trigram' (default 'bigram')

    Returns:
        Dictionary with factscore, supported_facts, total_facts, etc.
    """
    if ngram_type == "bigram":
        ngrams = extract_bigrams(ocr_text)
        is_supported_fn = is_bigram_supported
    else:
        ngrams = extract_trigrams(ocr_text)
        is_supported_fn = is_trigram_supported

    results = []
    supported_count = 0

    for ngram in ngrams:
        supported = is_supported_fn(ngram, agent_response, threshold)
        if supported:
            supported_count += 1
        results.append({
            "fact": ngram,
            "supported": supported
        })

    total = len(ngrams)
    factscore = supported_count / total if total else 0.0

    return {
        "facts": results,
        "supported_facts": supported_count,
        "total_facts": total,
        "factscore": factscore,
        "hallucination_rate": 1 - factscore
    }


def load_agent_responses(json_path: str):
    """
    JSON format:
    {
      "image1.png": "Agent response text here",
      "image2.png": "Agent response text here"
    }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_folder(image_folder: str, responses_json: str, output_csv: str, threshold: int, ngram_type: str = "bigram"):
    responses = load_agent_responses(responses_json)
    rows = []

    for filename in os.listdir(image_folder):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in IMAGE_EXTS:
            continue

        image_path = os.path.join(image_folder, filename)
        agent_response = responses.get(filename)

        if not agent_response:
            print(f"Skipping {filename}: no agent response found")
            continue

        ocr_text = run_ocr(image_path)
        score = compute_factscore(agent_response, ocr_text, threshold, ngram_type)

        rows.append({
            "image": filename,
            "ocr_text": ocr_text,
            "supported_facts": score["supported_facts"],
            "total_facts": score["total_facts"],
            "factscore": score["factscore"],
            "hallucination_rate": score["hallucination_rate"],
            "facts": json.dumps(score["facts"], ensure_ascii=False)
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", required=True)
    parser.add_argument("--responses_json", required=True)
    parser.add_argument("--output_csv", default="factscore_results.csv")
    parser.add_argument("--threshold", type=int, default=75)
    parser.add_argument("--ngram", type=str, default="bigram", choices=["bigram", "trigram"])

    args = parser.parse_args()

    evaluate_folder(
        image_folder=args.image_folder,
        responses_json=args.responses_json,
        output_csv=args.output_csv,
        threshold=args.threshold,
        ngram_type=args.ngram
    )
