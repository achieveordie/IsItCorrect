"""
This code takes streams from `sample_output.txt` and `sample_tagged.txt` to create a json file `sample_json.json`
with (tag, word) as the key-value pair.
"""
import json
from pathlib import Path

# the dict will contain keys as per tag-list html and list of words as the corresponding values:
tag_dict = {}
word_dict = {}


def create_tag_json(word_loc=Path('sample_output.txt'), tag_loc=Path('sample_tagged.txt'),
                    tag_json_loc=Path('sample.json'), word_json_loc=Path('word.json')):
    word = Path.open(word_loc, encoding='utf-8').read().splitlines()
    tag = Path.open(tag_loc).read().splitlines()
    word = [w for w in word if w]  # remove the empty lines(if present)
    with open(tag_json_loc, 'w', encoding='utf-8') as jfile:
        for i, (t, w) in enumerate(zip(tag, word)):
            try:
                tag_dict[t].append(w)
            except KeyError:
                tag_dict[t] = []
                tag_dict[t].append(w)

        for key in tag_dict.keys():
            tag_dict[key] = list(set(tag_dict[key]))  # remove same entries
        json.dump(tag_dict, jfile, ensure_ascii=False, indent=4)
        print("We're done for tags, folks.")
    with open(word_json_loc, 'w', encoding='utf-8') as jfile:
        for (w, t) in zip(word, tag):
            word_dict[w] = t
        json.dump(word_dict, jfile, ensure_ascii=False, indent=4)
        print("We're done for words, folks.")


if __name__ == '__main__':
    create_tag_json()