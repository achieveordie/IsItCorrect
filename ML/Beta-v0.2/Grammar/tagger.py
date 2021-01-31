from pathlib import Path
import subprocess
import multiprocessing


def convert_multi(input_file, output_file):
    tagger_location = Path(r"D:/TreeTagger")

    process_batch = subprocess.Popen([
        str(tagger_location / 'bin' / 'tag-french.bat'),
        str(input_file),
        str(output_file)
    ])

    return tuple(process_batch.communicate())


def convert(input_file, output_file=Path("data/sample_tagged.txt"),
            delete_residue=True):
    """
    Uses `subprocess` module to execute shell commands for TreeTagger to tag.
    :param delete_residue: <bool> Should the single-word-per-line be deleted after operation?
    :param input_file:  <Path-Location> (complete) location of file saved from `prepare_text.py`
    :param output_file: <Path-Location> location of file to be saved
    :return: None
    """
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    results = []
    pool_data = ([str(input_file)+str(i), output_file] for i in range(multiprocessing.cpu_count()))
    mapping = pool.map_async(convert_multi, pool_data, callback=results.append)
    mapping.wait()

    print(results)

    if delete_residue:
        try:
            print("Deleting single word/line file..")
            input_file.unlink()
        except OSError:
            print("Shouldn't reach here but anyways.. {} not deleted".format(input_file))


if __name__ == '__main__':
    main_dir = Path.cwd()
    input = main_dir / "sample_output.txt"
    output = main_dir / "sample_tagged.txt"
    convert(input_file=input, output_file=output)
