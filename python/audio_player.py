import queue
import socket
import struct
import sys
import time
import threading

import simpleaudio as sa
import numpy as np


SAMPLE_RATE = 48000
BYTES_PER_FRAME = 8192
FRAMES_PER_SECOND = 48
BYTES_PER_SECOND = BYTES_PER_FRAME * FRAMES_PER_SECOND

# Need to open the incoming port in your firewall first.
HOST = '169.254.103.120'
PORT = 11009

def server_thread(q):
    # create TCP socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.settimeout(120)

    print("Listening to socket...")
    s.listen()
    try:
        print("Waiting for connection")
        conn, addr = s.accept()
    except:
        print("Timed out waiting for connection")
        return

    print("Connected!!")

    conn.settimeout(60)

    frames_recvd = 0
    prev_time = -1
    sound_data = bytearray()

    while True:
        # wait for a message
        try:
            data = conn.recv(8)
        except:
            break

        if data[0:4] != b'\x1a\xcf\xfc\x1d':
            print("Invalid sync pattern", data[0:4])
            print(data[0:4].decode())
            break

        frames_recvd += 1
        if prev_time == -1:
            prev_time = time.time()
        elif (time.time() - prev_time > 1):
            print("Samples rcvd", frames_recvd)
            frames_recvd = 0
            prev_time = time.time()

        total_message_length = list(bytes(data[4:8]))
        total_message_length = ((total_message_length[0] << 24) |
                                (total_message_length[1] << 16) |
                                (total_message_length[2] << 8) |
                                (total_message_length[3] << 0))

        # read the rest of the message from the socket using the given length
        audio_data = bytearray()
        bytes_read = 0
        default_read_size = 8192
        while (bytes_read != total_message_length):
            bytes_remaining = total_message_length - bytes_read

            if default_read_size > bytes_remaining:
                read_size = bytes_remaining
            elif default_read_size > total_message_length:
                read_size = total_message_length
            else:
                read_size = default_read_size

            message = conn.recv(read_size)
            bytes_read += len(message)
            audio_data.extend(message)

        q.put(audio_data)


def play_audio_thread(q):
    audio_stream = bytearray()
    while True:
        try:
            audio = q.get(timeout=60)
            audio_stream.extend(audio)
        except:
            break

        if len(audio_stream) >= (BYTES_PER_SECOND):
            #print ("got audio", len(audio_stream))
            play_obj = sa.play_buffer(audio_stream, 2, 4, SAMPLE_RATE)
            audio_stream = bytearray()


def main():
    q = queue.Queue()
    server_t1 = threading.Thread(target=server_thread, args=(q, ))
    server_t1.daemon = True
    server_t1.start()

    server_t2 = threading.Thread(target=play_audio_thread, args=(q, ))
    server_t2.daemon = True
    server_t2.start()

    server_t1.join()
    server_t2.join()


if __name__ == "__main__":
    main()
