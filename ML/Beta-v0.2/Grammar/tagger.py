from pathlib import Path
import subprocess
import multiprocessing
from timeit import default_timer as timer


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
    pool_data = ([str(input_file)+str(i)+'.txt', output_file+str(i)+'.txt'] for i in range(multiprocessing.cpu_count()))
    mapping = pool.starmap_async(convert_multi, pool_data, callback=results.append)
    mapping.wait()

    if delete_residue:
        try:
            print("Deleting single word/line file..")
            [Path(input_file+str(i)+'.txt').unlink() for i in range(multiprocessing.cpu_count())]
        except OSError:
            print("Shouldn't reach here but anyways.. {} not deleted".format(input_file))

    file_no = 0
    with open(str(output_file)+'.txt', 'w', encoding='utf-8') as wfile:
        while file_no < multiprocessing.cpu_count():
            with open(str(output_file)+str(file_no)+'.txt', 'r', encoding='utf-8') as rfile:
                wfile.write(rfile.read())
            file_no += 1

    try:
        [Path(output_file + str(i) + '.txt').unlink() for i in range(multiprocessing.cpu_count())]
    except OSError:
        print("Had problem removing files created by processes, delete manually.")


# if __name__ == '__main__':
#     start = timer()
#     input_fi = r"D:\Datasets\IsItCorrect\temp\output"
#     output_fi = r"D:\Datasets\IsItCorrect\temp\tagged"
#     convert(input_file=input_fi, output_file=output_fi, delete_residue=True)
#     end = timer()
#     print(end-start)
