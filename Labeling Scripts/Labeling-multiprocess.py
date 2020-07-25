import time
import multiprocessing as mp
import os
import re
import random
testing_location = r'D:\Datasets\IsItCorrect\testing.txt'
pattern = re.compile('<START>.*?<END>')


def process_wrapper(chunkStart, chunkSize):
    with open(testing_location, 'r', encoding='utf8') as file:
        file.seek(chunkStart)
        lines = file.read(chunkSize)
        lines = pattern.findall(lines)
        for line in lines:
            print(line)
            break


def chunkify(fname, size=1000000):
    fileEnd = os.path.getsize(fname)
    with open(fname, 'rb') as file:
        chunkEnd = file.tell()

        while True:
            chunkStart = chunkEnd
            file.seek(size, 1)
            file.readline()
            chunkEnd = file.tell()
            yield chunkStart, chunkEnd - chunkStart
            if chunkEnd > fileEnd:
                break


if __name__ == '__main__':
    mp.freeze_support()
    start = time.time()
    pool = mp.Pool(4)
    jobs = []

    for chunkStart, chunkSize in chunkify(testing_location):
        jobs.append(pool.apply_async(process_wrapper, (chunkStart, chunkSize, )))
    for job in jobs:
        job.get()
    pool.close()
    end = time.time()
    print(end-start)