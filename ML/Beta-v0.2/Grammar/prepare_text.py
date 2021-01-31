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
    chunksize = 404700
    file_no = -1
    pattern = re.compile("<START>(.*?)<END>")
    with open(str(input_file), 'r', encoding='utf-8') as rfile:
        while True:
            file_no += 1
            lines = rfile.read(chunksize)
            if not lines:
                break
            lines = pattern.findall(lines)
            with open(str(output_file) + str(file_no) + '.txt', 'w', encoding='utf-8') as wfile:
                for line in lines:
                    for word in line.split(" "):
                        if len(word) > 0:
                            if word[-1] in ('.', ','):
                                broken = word[0:-2].split("'")
                                if len(broken) > 1:
                                    broken[0] += "'"
                                [wfile.write(broke + '\n') for broke in broken]
                                # file.write(word[0:len(word)-2]+'\n')
                                wfile.write(word[-1] + '\n')
                            else:
                                broken = word.split("'")
                                if len(broken) > 1:
                                    broken[0] += "'"
                                [wfile.write(broke + '\n') for broke in broken]
    print("Converted file for Tagger to work on.")


if __name__ == "__main__":
    convert(
        input_file=Path('D:/Datasets/IsItCorrect') / "beta2_test.txt",
        output_file=Path("D:/Datasets/IsItCorrect/temp") / 'output'
    )