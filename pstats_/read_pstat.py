# A Script to read the Stats offered by cProfile to measure the
# time taken for different processes in prints in the outstream

import pstats
from pstats import SortKey
stats_location = r'step_1_reddit_pstats.txt'
p = pstats.Stats(stats_location)
p.sort_stats(SortKey.CUMULATIVE).print_stats(15)

# Above line sorts the processes with descending cumulative time and prints the top
# 15 processes
