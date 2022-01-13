import numpy as np
from PIL import Image, ImageDraw
import queue
import socket
import struct
import time
import threading
import torch
from torchvision import datasets, transforms

from matplotlib import pyplot as plot
from smqtk_detection.impls.detect_image_objects.resnet_frcnn import ResNetFRCNN

HOST = '192.168.1.89'  # Standard loopback interface address (localhost)
PORT = 11000        # Port to listen on (non-privileged ports are > 1023)
IMAGE_FILENAME = "C:\\Users\\josh.anderson\\Desktop\\hololens_image.jpeg"

def server_thread(image_queue, bb_queue):
    # create TCP socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.settimeout(120)

    print("Waiting for connection")
    s.listen()
    try:
        conn, addr = s.accept()
    except:
        print("Timed out waiting for connection")
        return

    print("Connected!!")

    conn.settimeout(60)

    while True:
        # wait for a message
        try:
            data = conn.recv(8)
        except:
            break

        start_time = time.time()

        if data[0:4] != b'\x1a\xcf\xfc\x1d':
            print("Invalid sync pattern")
            break

        total_message_length = list(bytes(data[4:8]))
        total_message_length = ((total_message_length[0] << 24) |
                                (total_message_length[1] << 16) | 
                                (total_message_length[2] << 8) | 
                                (total_message_length[3] << 0)) 

        # read the rest of the message from the socket using the given length
        screenshot_data = []
        bytes_read = 0
        read_size = 4096
        while (bytes_read != total_message_length):
            if read_size > total_message_length:
                read_size = total_message_length

            message = list(conn.recv(read_size))
            bytes_read += len(message)
            screenshot_data.extend(message)

        end_time = time.time()
        print("Read %d bytes in %f seconds" % (bytes_read, end_time - start_time))

        # send image to the queue
        image_queue.put(screenshot_data)

        # try and get the bounding box result
        object_type = bb_queue.get()
        bounding_box = bb_queue.get()
        print("Got bounding box", bounding_box, object_type)
        print("Min vertex", list(bounding_box.min_vertex))
        print("Max vertex", list(bounding_box.max_vertex))
        
        # convert to bytes
        object_type = bytearray(struct.pack("I", object_type))
        min_vertex0 = bytearray(struct.pack("f", bounding_box.min_vertex[0])) 
        min_vertex1 = bytearray(struct.pack("f", bounding_box.min_vertex[1]))  
        max_vertex0 = bytearray(struct.pack("f", bounding_box.max_vertex[0]))  
        max_vertex1 = bytearray(struct.pack("f", bounding_box.max_vertex[1]))  

        vertex_bytes = object_type + min_vertex0 + min_vertex1 + max_vertex0 + max_vertex1

        # send result via socket
        bytes_sent = conn.send(vertex_bytes)

        print("Sent bytes", bytes_sent)


        '''
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
        image_np = np.array(image)

        print("image_np stuff", image_np.shape, image_np.dtype, image_np[0:12])

        image_np = np.reshape(image_np, (height, width, 3))
        image_np = np.flip(image_np, axis=0).copy()

        print("image_np stuff", image_np.shape, image_np.dtype)

        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1))
        plot.imshow(image_tensor.permute(1, 2, 0))
        plot.show()
        '''


def detector_thread(image_queue, bb_queue):
    print("Hello from detector thread")

    # instantiate detector
    detector = ResNetFRCNN()
    print("Ready to detect", detector, detector.use_cuda)

    while True:
        print("Waiting for image")

        # wait for an image
        try:
            image = image_queue.get(timeout=60)
        except:
            print("Timed out waiting for image")
            break

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
        image_np = np.array(image)
        print("image_np stuff", image_np.shape, image_np.dtype, height, width)

        image_np = np.reshape(image_np, (height, width, 3))
        image_np = np.flip(image_np, axis=0).copy()

        im = Image.fromarray(image_np.astype(np.uint8))
        im.save(IMAGE_FILENAME)

        image_np = (image_np / 255.0).astype(np.float32)
        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1))

        transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        image_tensor = transform(image_tensor)

        #print("image!", image_tensor[0][0], image_np.shape)

        # send to detector
        detections = detector.detect_objects(image_tensor)
        for detection in detections:
            #print("detection", detection)
            for i in detection:
                bounding_box = i[0]
                class_dict = i[1]
                
                class_dict_sorted = {k: v for k, v in sorted(class_dict.items(), key=lambda item: item[1], reverse=True)}
                #print("Results sorted:", list(class_dict_sorted.items())[:5])

                if list(class_dict_sorted.items())[0][1] > 0.90:
                    print("Found something:", list(class_dict_sorted.items())[0], bounding_box)
                
                    source_img = Image.open(IMAGE_FILENAME).convert("RGB")

                    draw = ImageDraw.Draw(source_img)
                    draw.rectangle(((bounding_box.min_vertex[0], bounding_box.min_vertex[1]),
                                    (bounding_box.max_vertex[0], bounding_box.max_vertex[1])),
                                    outline="black", width=10)

                    source_img.save(IMAGE_FILENAME, "JPEG")

                    # send bounding box coordinates back
                    #if list(class_dict_sorted.items())[0][0] == 76:
                    bb_queue.put(list(class_dict_sorted.items())[0][0])
                    bb_queue.put(bounding_box)
                else:
                    #print("Results sorted:", list(class_dict_sorted.items())[:5])
                    #print(list(class_dict_sorted.items())[0][1])
                    pass


def main():
    image_q = queue.Queue()
    bb_q = queue.Queue()
    
    server_t = threading.Thread(target=server_thread, args=(image_q, bb_q, ))
    server_t.daemon = True
    server_t.start()

    detector_t = threading.Thread(target=detector_thread, args=(image_q, bb_q, ))
    detector_t.daemon = True
    detector_t.start()

    server_t.join()
    detector_t.join()
    
    #server_thread(q)

if __name__ == "__main__":
    main()