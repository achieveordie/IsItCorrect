"""
This code is responsible for converting a given input text file into the appropriate TreeTagger-text input file.
"""
from pathlib import Path
import re


def convert(input_file, output_file=Path('data/sample_output.txt')):
    """
    To convert a given input text into apt. text file for POS-tagging.
    :param output_file: <Path-Location> location to save the file
    :param input_file:  <Path-Location> the file to be converted
    :return: None
    """
    pattern = re.compile("<START>(.*?)<END>")
    with open(str(input_file), 'r', encoding='utf-8') as file:
        lines = file.readlines()
        lines = pattern.findall(lines[0])
    with open(str(output_file), 'w', encoding='utf-8') as file:
        for line in lines:
            for word in line.split(" "):
                if len(word) > 0:
                    if word[-1] in ('.', ','):
                        broken = word[0:-2].split("'")
                        if len(broken) > 1:
                            broken[0] += "'"
                        [file.write(broke + '\n') for broke in broken]
                        # file.write(word[0:len(word)-2]+'\n')
                        file.write(word[-1]+'\n')
                    else:
                        broken = word.split("'")
                        if len(broken) > 1:
                            broken[0] += "'"
                        [file.write(broke + '\n') for broke in broken]
                        # file.write(word+'\n')
    print("Converted to {}".format(str(output_file)))
