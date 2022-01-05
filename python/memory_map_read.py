import os
import mmap
import time

n = 1000
mm = mmap.mmap(-1, n, "ARandomTagName")
while True:
    # read image
    start = time.perf_counter()
    mm.seek(0)
    buf = mm.read(12)
    print(buf)
    stop = time.perf_counter()

mm.close()