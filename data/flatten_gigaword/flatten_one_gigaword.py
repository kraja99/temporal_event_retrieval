# Adapted from https://github.com/nelson-liu/flatten_gigaword

import logging
import os
import re
import spacy

from argparse import ArgumentParser
from bs4 import BeautifulSoup
from tqdm import tqdm

en_nlp = spacy.load("en_core_web_sm")

def get_bad_id_set(gigaword_path):
    bad_ids = []
    with open(gigaword_path + "/docs/nyt_noisy_paragraphs.tab", "r") as noisy:
        for i, line in enumerate(noisy):
            if i == 0:
                continue
            bad_ids.append(line.split("\t")[0])
    with open(gigaword_path + "/docs/other.file-doc.map", "r") as other:
        for line in other:
            bad_ids.append(line.split(" ")[1].strip())
    with open(gigaword_path + "/docs/spanish.file-doc.map", "r") as spanish:
        for line in spanish:
            bad_ids.append(line.split(" ")[1].strip())
    return set(bad_ids)


def flatten_one_gigaword_file(root, filename, bad_ids):
    # Parse the text with BeautifulSoup
    file_path = root + "/data/afp_eng/" + filename
    soup = BeautifulSoup(open(file_path), "html.parser")

    # Iterate over all <p> items and get the text for each.
    docid_to_text = {}
    for document in soup("doc"):
        if document.get("type") != "story" or document.get("id") in bad_ids:
            continue
        all_paragraphs = []
        for paragraph in document("p"):
            # Turn inter-paragraph newlines into spaces
            paragraph = paragraph.get_text()
            paragraph = re.sub(r"\n+", "\n", paragraph)
            paragraph = paragraph.replace("\n", " ")
            # Tokenize the paragraph into words
            tokens = en_nlp.tokenizer(paragraph)
            words = [str(token) for token in tokens if not
                    str(token).isspace()]
            if len(words) < 3:
                continue
            all_paragraphs.append(words)
        docid_to_text[document.get("id")] = [" ".join(paragraph) for paragraph in all_paragraphs]
    # Return a list of strings, where each string is a
    # space-tokenized paragraph.
    return docid_to_text


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    parser = ArgumentParser(description=("Flatten a gigaword data file for "
                                         "use in language modeling."))
    parser.add_argument("--gigaword-root", required=False,
                        metavar="<gigaword_root>", type=str,
                        help=("Path to Gigaword toot directory"))
    parser.add_argument("--output-dir", required=True, metavar="<output_dir>",
                        type=str, help=("Directory to write final flattened "
                                        "Gigaword file."))

    A = parser.parse_args()
    bad_ids = get_bad_id_set(A.gigaword_root)
    files = os.listdir(A.gigaword_root + "data/afp_eng/")
    for f in tqdm(files):
        docid_to_text = flatten_one_gigaword_file(A.gigaword_root, f, bad_ids)
        output_path = os.path.join(A.output_dir, "data/afp_eng_flat/" + f)
        os.makedirs(output_path, exist_ok=True)
        for docid in docid_to_text:
            with open(output_path + "/" + docid, "w") as output_file:
                for paragraph in docid_to_text[docid]:
                    output_file.write("{}\n".format(paragraph))
