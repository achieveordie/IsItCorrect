import json
from pathlib import Path
import re


base_save_location = Path.cwd() / 'Grammar' / 'data'
pos_json_loc = base_save_location / 'pos.json'
pos_json = json.load(open(str(pos_json_loc), 'r', encoding='utf-8'))
verb_grammar = [pos for pos in pos_json.keys() if pos[0:3] == "VER"]
print(verb_grammar)
pronoun_grammar = [pos for pos in pos_json.keys() if pos[0:3] == "PRO"]
print(pronoun_grammar)
pattern = re.compile("<START>(.*?)<END>")
with open('sample_text.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()
    lines = pattern.findall(lines[0])

formatted = []
format_location = Path.cwd() / 'formatted.txt'
with open(str(format_location), 'w', encoding='utf-8') as file:
    for line in lines:
        single_line = []
        for word in line.split(" "):
            if len(word) > 0:
                if word[-1] in ('.', ','):
                    broken = word[0:-2].split("'")
                    if len(broken) > 1:
                        broken[0] += " '"
                    [single_line.append(broke) for broke in broken]
                    single_line.append(word[-1])
                else:
                    broken = word.split("'")
                    if len(broken) > 1:
                        broken[0] += " '"
                    [single_line.append(broke) for broke in broken]
        formatted.append('<START>' + " ".join(single_line) + '<END>')
    [file.write(line + '\n') for line in formatted]
