"""
The main file to control the complete flow of the `Grammar` module as mentioned by `explanation.txt`.
No other code needs to be called if using this file.
"""

import create_json, prepare_text, tagger, lemma
from pathlib import Path
import time


start_time = time.time()
# dir_location = Path(__file__).resolve().parent
# base_save_location = Path.cwd() / 'data'  # saving the sample files in the folder itself
base_save_location = Path("D:/Datasets/IsItCorrect/temp")

# original_file_location = Path('..') / 'sample_text.txt'
original_file_location = Path('D:/Datasets/IsItCorrect') / "parsed_text.txt"
converted_file_location = base_save_location / 'output.txt'
tagged_file_location = base_save_location / 'tagged.txt'
pos_json = base_save_location / 'pos.json'
word_json = base_save_location / 'word.json'
lemma_json = base_save_location / 'lemma.json'
word_lemma_json = base_save_location / 'word_lemma.json'

# The files shouldn't exist, except for the original file.
assert original_file_location.exists(), f"Assert Error: {original_file_location} doesn't exists."

assert not tagged_file_location.exists(),    f"Assert Error: {tagged_file_location} already exists."
assert not converted_file_location.exists(), f"Assert Error: {converted_file_location} already exists."
assert not pos_json.exists(),  f"Assert Error: {pos_json} already exists."
assert not word_json.exists(), f"Assert Error: {word_json} already exists."
assert not lemma_json.exists(), f"Assert Error: {lemma_json} already exists."
assert not word_lemma_json.exists(), f"Assert Error: {word_lemma_json} already exists."

# convert original to converted:
prepare_text.convert(input_file=original_file_location,
                     output_file=converted_file_location)

# use the converted file to make tagged, delete_residue will delete `converted_file`:
tagger.convert(input_file=converted_file_location,
               output_file=tagged_file_location,
               delete_residue=True)

# use tagged and convert file to make both json files:
create_json.convert(tag_loc=tagged_file_location,
                    tag_json_loc=pos_json,
                    word_json_loc=word_json)

# create {lemma:words} pair and {word:lemma} pair, delete_residue will delete `tagged_file`:
lemma.convert(tag_loc=tagged_file_location,
              lemma_json_loc=lemma_json,
              word_lemma_json_loc=word_lemma_json,
              delete_residue=True)

total_duration = time.time() - start_time
print("Total time taken for the synchronous version ->", total_duration)
