"""
This code takes streams from `sample_output.txt` and `sample_tagged.txt` to create a json file `sample_json.json`
with (tag, word) as the key-value pair.
"""
import json
from pathlib import Path


def convert(tag_loc=Path('data/sample_tagged.txt'),
            tag_json_loc=Path('data/sample.json'), word_json_loc=Path('data/word.json')):
    # the dict will contain keys as per tag-list html and list of words as the corresponding values:
    tag_dict = {}
    word_dict = {}
    lines = None
    with open(str(tag_loc), 'r', encoding='utf-8') as file:
        lines = file.read().splitlines()
        lines = [line.split('\t') for line in lines]

    with open(tag_json_loc, 'w', encoding='utf-8') as jfile:
        try:
            for i, (w, t, _) in enumerate(lines):
                try:
                    tag_dict[t].append(w)
                except KeyError:
                    tag_dict[t] = []
                    tag_dict[t].append(w)
        except ValueError:
            pass

        for key in tag_dict.keys():
            tag_dict[key] = list(set(tag_dict[key]))  # remove same entries
        json.dump(tag_dict, jfile, ensure_ascii=False, indent=4)
        print("We're done for tags, folks.")
    with open(word_json_loc, 'w', encoding='utf-8') as jfile:
        try:
            for (w, t, _) in lines:
                word_dict[w] = t
        except ValueError:
            pass
        json.dump(word_dict, jfile, ensure_ascii=False, indent=4)
        print("We're done for words, folks.")

# if __name__ == '__main__':
#     convert()
