import numpy as np
from PIL import Image, ImageDraw
import queue
import socket
import struct
import sys
import time
import threading

from matplotlib import pyplot as plot
import matplotlib


# Need to open the incoming port in your firewall first.
HOST = '192.168.1.89'
PORT = 11000
IMAGE_FILENAME = "C:\\Users\\josh.anderson\\Desktop\\hololens_image_"

# Enable interactive mode
plot.ion()
plot.show(block=False)


def get_im():
    try:
        return get_im._im1
    except AttributeError:
        get_im._ax1 = ax1 = plot.subplot(111)
        get_im._im1 = im1 = (
            ax1.imshow(np.zeros(shape=(480, 640, 1)), cmap='gray', vmin=0, vmax=255)
        )
    return get_im()


# get_im()


def server_thread(name, port):
    # create TCP socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, int(port)))
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
    prev_screenshot_data = []
    idx = 0

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
            print("Frames rcvd", name, frames_recvd)
            frames_recvd = 0
            prev_time = time.time()

        total_message_length = list(bytes(data[4:8]))
        total_message_length = ((total_message_length[0] << 24) |
                                (total_message_length[1] << 16) |
                                (total_message_length[2] << 8) |
                                (total_message_length[3] << 0))

        #print("message length", total_message_length)

        # read the rest of the message from the socket using the given length
        screenshot_data = []
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

            message = list(conn.recv(read_size))
            bytes_read += len(message)
            screenshot_data.extend(message)

        #continue # NOTE: just to test out the framerate

        image = screenshot_data

        width = ((image[0] & 0xFF << 24) |
                 (image[1] << 16) | 
                 (image[2] << 8) | 
                 (image[3] << 0)) 
        height = ((image[4] << 24) |
                 (image[5] << 16) | 
                 (image[6] << 8) | 
                 (image[7] << 0)) 
        image = image[8:]

        # convert to np-array
        image_np_orig = np.array(image)

        #print("image_np stuff", image_np.shape, image_np.dtype, image_np[0:12])

        image_np = np.reshape(image_np_orig, (height, width, 1))
        image_np = image_np.astype(np.uint8)

        if name == "LF":
            image_np = np.rot90(image_np, k=3)
        elif name == "RF":
            image_np = np.rot90(image_np, k=1)
        elif name == "LL":
            image_np = np.rot90(image_np, k=1)
        elif name == "RR":
            image_np = np.rot90(image_np, k=3)



        # NOTE: to save images to disk
        '''
        im = Image.fromarray(np.squeeze(image_np), mode='L')
        image_filename = IMAGE_FILENAME + ("%d" % (idx)) + ".jpeg"
        im.save(image_filename, "JPEG")
        idx += 1
        '''

        get_im().set_data(image_np)

        plot.gcf().canvas.flush_events()
        plot.show(block=False)


def main():
    '''
    server_t1 = threading.Thread(target=server_thread, args=("LF", 11000, ))
    server_t1.daemon = True
    server_t1.start()

    server_t2 = threading.Thread(target=server_thread, args=("RF", 11001, ))
    server_t2.daemon = True
    server_t2.start()

    server_t3 = threading.Thread(target=server_thread, args=("LL", 11002, ))
    server_t3.daemon = True
    server_t3.start()

    #server_t4 = threading.Thread(target=server_thread, args=("RR", 11003, ))
    #server_t4.daemon = True
    #server_t4.start()


    server_t1.join()
    server_t2.join()
    server_t3.join()
    #server_t4.join()
    '''

    server_thread(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
