import os
import mmap
import time
import win32pipe

n = 272727 * 4
mm = mmap.mmap(-1, n, "ARandomTag")

while True:
    print("Reading...")
    mm_size = mm.size()
    if mm_size != 0:
        time_start = time.time()
        buf = mm.read()
        time_end = time.time()
        print("Read %d bytes" % (mm_size))
        print("Buf", buf[0:100])
        print("Time: ", time_end - time_start)
        mm.seek(0)

    time.sleep(1)