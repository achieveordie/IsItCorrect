from pathlib import Path
import subprocess


def convert(input_file, output_file=Path("data/sample_tagged.txt"),
            delete_residue=True):
    """
    Uses `subprocess` module to execute shell commands for TreeTagger to tag.
    :param delete_residue: <bool> Should the single-word-per-line be deleted after operation?
    :param input_file:  <Path-Location> (complete) location of file saved from `prepare_text.py`
    :param output_file: <Path-Location> location of file to be saved
    :return: None
    """
    tagger_location = Path(r"D:/TreeTagger")

    process_batch = subprocess.Popen([
        str(tagger_location / 'bin' / 'tag-french.bat'),
        str(input_file),
        str(output_file)
    ])
    stdout, stderror = process_batch.communicate()
    print("Stdout is -> ", stdout)
    print("StdError is -> ", stderror)
    print("Done and saved to ", output_file)

    print("Deleting single word/line file..")

    if delete_residue:
        try:
            input_file.unlink()
        except OSError:
            print("Shouldn't reach here but anyways.. {} not deleted".format(input_file))


# if __name__ == '__main__':
#     main_dir = Path.cwd()
#     input = main_dir / "sample_output.txt"
#     output = main_dir / "sample_tagged.txt"
#     convert(input_file=input, output_file=output)
