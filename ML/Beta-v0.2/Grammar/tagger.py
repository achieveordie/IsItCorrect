from pathlib import Path
import subprocess


def tag_file(input_file, output_file="sample_tagged.txt"):
    """
    Uses `subprocess` module to execute shell commands for TreeTagger to tag.
    :param input_file: (complete) location of file saved from `prepare_text.py`
    :param output_file: location of file to be saved
    :return: None
    """
    tagger_location = Path(r"D:/TreeTagger")
    param_file = tagger_location / 'lib' / 'french.par'
    process = subprocess.Popen([str(tagger_location / 'bin' / 'tree-tagger.exe'),
                                str(param_file),
                                input_file,
                                output_file])
    stdout, stderror = process.communicate()
    print("Stdout is -> ", stdout)
    print("StdError is -> ", stderror)
    print("Done and saved to ", output_file)


if __name__ == '__main__':
    main_dir = Path.cwd()
    input = main_dir / "sample_output.txt"
    output = main_dir / "sample_tagged.txt"
    tag_file(input_file=str(input), output_file=str(output))
