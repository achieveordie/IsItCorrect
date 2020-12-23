from pathlib import Path
import json


def convert(tag_loc=Path("data/sample_tagged.txt"), lemma_json_loc=Path("data/lemma.json"),
            word_lemma_json_loc=Path("data/word_lemma.json"), delete_residue=True):
    """
    Uses `subprocess` module to execute shell commands for TreeTagger to tag.
    :param delete_residue: <bool> should the tagged-file be deleted after operation?
    :param word_lemma_json_loc: <Path-Location> location of file to be saved for {word:lemma}
    :param tag_loc: <Path-Location> (complete) location of file saved from `prepare_text.py`
    :param lemma_json_loc: <Path-Location> location of file to be saved for {lemma: [words]}
    :return: None
    """
    assert tag_loc.exists(), f"Assert Error: {tag_loc} doesn't exists."

    lines = None
    lemma_dict = {}
    with open(str(tag_loc), 'r', encoding='utf-8') as file:
        lines = file.read().splitlines()
        lines = [line.split('\t') for line in lines]

    with open(str(lemma_json_loc), 'w', encoding='utf-8') as jfile:
        for (word, _, lemma) in lines:
            try:
                lemma_dict[lemma].append(word)
            except KeyError:
                lemma_dict[lemma] = []
                lemma_dict[lemma].append(word)

        # We will have nouns prepositions etc who's lemma is not useful.
        to_delete = []
        for key in lemma_dict.keys():
            lemma_dict[key] = list(set(lemma_dict[key]))
            if len(lemma_dict[key]) == 1:
                to_delete.append(key)
        for to_del in to_delete:
            del lemma_dict[to_del]
        json.dump(lemma_dict, jfile, ensure_ascii=False, indent=4)
        print("Done with Creating lemma file.")

    lemma_dict = {}
    with open(str(word_lemma_json_loc), 'w', encoding='utf-8') as jfile:
        for (word, _, lemma) in lines:

            lemma_dict[word] = lemma
        json.dump(lemma_dict, jfile, ensure_ascii=False, indent=4)
        print("Done with Creating word_lemma file.")

    if delete_residue:
        try:
            tag_loc.unlink()
        except OSError:
            print("Shouldn't reach here but anyways.. {} not deleted".format(tag_loc))