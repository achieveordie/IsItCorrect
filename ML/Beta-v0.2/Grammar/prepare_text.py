"""
This code is responsible for converting a given input text file into the appropriate TreeTagger-text input file.
"""
from pathlib import Path


def convert(file_location, output_location='sample_output.txt'):
    """
    To convert a given input text into apt. text file for POS-tagging.
    :param output_location: location to save the file
    :param file_location: the file to be converted
    :return: None
    """
    with open(file_location, 'r') as file:
        lines = file.readlines()
    with open(output_location, 'w') as file:
        for line in lines:
            for word in line.split(" "):
                file.write(word+'\n')
    print("Converted to {}".format(output_location))


if __name__ == "__main__":
    print(Path.cwd())
    file_location = Path('D:/IIC/IIC-ML/ML/Beta-v0.2/sample_text.txt')
    convert(str(file_location))
