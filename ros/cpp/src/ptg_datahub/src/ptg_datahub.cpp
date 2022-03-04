#include <chrono>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <thread>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/byte_multi_array.hpp"
#include "std_msgs/msg/u_int8_multi_array.hpp"

#include "sensor_msgs/msg/image.hpp"
#include "angel_msgs/msg/spatial_mesh.hpp"
#include "angel_msgs/msg/headset_pose_data.hpp"
#include "angel_msgs/msg/object_detection3d_set.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN

#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>

// Need to link with Ws2_32.lib, Mswsock.lib, and Advapi32.lib
#pragma comment (lib, "Ws2_32.lib")
#pragma comment (lib, "Mswsock.lib")
#pragma comment (lib, "AdvApi32.lib")
#endif

#ifdef __linux__
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <unistd.h>

typedef int SOCKET;

#endif

#define AUDIO_HEADER_LEN     (8)
#define VIDEO_HEADER_LEN     (16)
#define SM_HEADER_LEN        (8)
#define PV_VIDEO_HEADER_LEN  (16 + 64)
#define DEFAULT_READ_SIZE    (8192)
#define DEFAULT_BUFLEN       (1024 * 1024)

#define LF_VLC_TCP_PORT        (11000)
#define RF_VLC_TCP_PORT        (11001)
#define LL_VLC_TCP_PORT        (11002)
#define RR_VLC_TCP_PORT        (11003)
#define DEPTH_TCP_PORT         (11004)
#define DEPTH_AB_TCP_PORT      (11005)
#define LONG_DEPTH_TCP_PORT    (11006)
#define LONG_DEPTH_AB_TCP_PORT (11007)
#define PV_TCP_PORT            (11008)
#define AUDIO_TCP_PORT         (11009)
#define SM_TCP_PORT            (11010)

using namespace std::chrono_literals;
using std::placeholders::_1;

std::map<int, std::string> PORT_TOPIC_MAP = {
    { LF_VLC_TCP_PORT,        "LFFrames" },
    { RF_VLC_TCP_PORT,        "RFFrames" },
    { LL_VLC_TCP_PORT,        "LLFrames" },
    { RR_VLC_TCP_PORT,        "RRFrames" },
    { DEPTH_TCP_PORT,         "DepthFrames" },
    { DEPTH_AB_TCP_PORT,      "DepthABFrames" },
    { LONG_DEPTH_TCP_PORT,    "LongDepthFrames" },
    { LONG_DEPTH_AB_TCP_PORT, "LongDepthABFrames" },
    { PV_TCP_PORT,            "PVFrames" },
    { AUDIO_TCP_PORT,         "AudioData" },
    { SM_TCP_PORT,            "SpatialMapData" }
};

class PTGDataHub : public rclcpp::Node
{
  public:
    PTGDataHub()
    : Node("ptg_datahub"), count_(0)
    {
      try
      {
        tcp_server_uri =
          this->declare_parameter( "tcp_server_uri" ).get< std::string >();
      }
      catch( rclcpp::ParameterTypeException const &ex )
      {
        std::stringstream ss;
        ss << "Server IP address cannot be empty: " << ex.what();
        throw std::invalid_argument( ss.str() );
      }
      RCLCPP_INFO( this->get_logger(),
                   "Starting talker, intending to connect to TCP server @ %s",
                   tcp_server_uri.c_str() );

      _object_3d_subscriber = this->create_subscription<angel_msgs::msg::ObjectDetection3dSet>(
        "ObjectDetections3d", 100,
        std::bind(&PTGDataHub::object_detection_3d_callback, this, _1 )
      );

    }

    std::thread StartTCPServerThread(int port)
    {
      std::thread t;
      if ((port < LF_VLC_TCP_PORT) || (port > SM_TCP_PORT))
      {
        std::cout << "Invalid port number entered!\n";
      }

      if (port == AUDIO_TCP_PORT)
      {
        t = std::thread(&PTGDataHub::TCPServerAudioThread, this, port);
      }
      else if (port == SM_TCP_PORT)
      {
        t = std::thread(&PTGDataHub::TCPServerSMThread, this, port);
      }
      else
      {
        t = std::thread(&PTGDataHub::TCPServerVideoThread, this, port);
      }

      return t;
    }


  private:
    rclcpp::Subscription< angel_msgs::msg::ObjectDetection3dSet >::SharedPtr _object_3d_subscriber;
    size_t count_;
    std::string tcp_server_uri;
    std::vector<angel_msgs::msg::ObjectDetection3dSet::SharedPtr> _detections;
    std::mutex _detection_mutex;

    std::vector<unsigned char> uint_to_vector(unsigned int x)
    {
      std::vector<unsigned char> v(4);
      memcpy(&v[0], &x, sizeof(x));
      return v;
    }

    std::vector<unsigned char> float_to_vector(float x)
    {
      std::vector<unsigned char> v(4);
      memcpy(&v[0], &x, sizeof(x));
      return v;
    }

    std::vector<unsigned char> serialize_detection_message( angel_msgs::msg::ObjectDetection3dSet::SharedPtr
                                                            const detection_msg )
    {
      // serialize the detection message
      std::vector<unsigned char> byte_message;

      unsigned int ros_message_length = 0;

      // PTG header:
      //   -- 32-bit sync = 4 bytes
      //   -- 32-bit ros msg length = 4 bytes
      // ROS2 message:
      //  header
      //   -- 32 bit seconds = 4 bytes
      //   -- 32 bit nanoseconds = 4 bytes
      //   -- frame id string
      //  source_stamp
      //   -- 32 bit seconds
      //   -- 32 bit nanoseconds
      //  num_objects
      //   -- 32 bit num objects
      //  labels
      //   -- string * num_objects
      //  3d points: 12 * 4 * num_objects
      //   -- left points (12 bytes)
      //     -- 32 bit float x
      //     -- 32 bit float y
      //     -- 32 bit float z
      //   -- top points (12 bytes)
      //     -- 32 bit float x
      //     -- 32 bit float y
      //     -- 32 bit float z
      //   -- right points (12 bytes)
      //     -- 32 bit float x
      //     -- 32 bit float y
      //     -- 32 bit float z
      //   -- bottom points (12 bytes)
      //     -- 32 bit float x
      //     -- 32 bit float y
      //     -- 32 bit float z

      // convert the frame id to bytes
      std::vector<char> frame_id_bytes(detection_msg->header.frame_id.begin(),
                                       detection_msg->header.frame_id.end());
      frame_id_bytes.push_back('\0');
      ros_message_length += frame_id_bytes.size();

      // convert the object labels to bytes
      std::vector<char> object_labels;
      for (int i = 0; i < detection_msg->num_objects; i++)
      {
        object_labels.insert(object_labels.end(),
                             detection_msg->object_labels[i].begin(),
                             detection_msg->object_labels[i].end());
        object_labels.push_back('\0');
      }
      ros_message_length += object_labels.size();

      ros_message_length += (4 + // header seconds
                             4 + // header nanoseconds
                             // frame id length added already
                             4 + // source seconds
                             4 + // source nanoseconds
                             4 + // num objects
                             // labels length added already
                             (detection_msg->num_objects * 4 * 12));

      // add sync
      std::vector<unsigned char> sync = uint_to_vector(0x1ACFFC1D);
      byte_message.insert(byte_message.end(), sync.begin(), sync.end());

      // add length
      std::vector<unsigned char> length = uint_to_vector(ros_message_length);
      byte_message.insert(byte_message.end(), length.begin(), length.end());

      // add header time stamp
      std::vector<unsigned char> seconds = uint_to_vector(detection_msg->header.stamp.sec);
      byte_message.insert(byte_message.end(), seconds.begin(), seconds.end());
      std::vector<unsigned char> nanoseconds = uint_to_vector(detection_msg->header.stamp.nanosec);
      byte_message.insert(byte_message.end(), nanoseconds.begin(), nanoseconds.end());

      // add frame id
      byte_message.insert(byte_message.end(), frame_id_bytes.begin(), frame_id_bytes.end());

      // add source stamp
      seconds = uint_to_vector(detection_msg->source_stamp.sec);
      byte_message.insert(byte_message.end(), seconds.begin(), seconds.end());
      nanoseconds = uint_to_vector(detection_msg->source_stamp.nanosec);
      byte_message.insert(byte_message.end(), nanoseconds.begin(), nanoseconds.end());

      // add num objects
      std::vector<unsigned char> num_objects = uint_to_vector(detection_msg->num_objects);
      byte_message.insert(byte_message.end(), num_objects.begin(), num_objects.end());

      // add labels
      byte_message.insert(byte_message.end(), object_labels.begin(), object_labels.end());

      // add points
      for (int i = 0; i < 4; i++)
      {
        for (int j = 0; j < detection_msg->num_objects; j++)
        {
          geometry_msgs::msg::Point p;
          if (i == 0)
          {
            p = detection_msg->left[j];
          }
          else if (i == 1)
          {
            p = detection_msg->top[j];
          }
          else if (i == 2)
          {
            p = detection_msg->right[j];
          }
          else if (i == 3)
          {
            p = detection_msg->bottom[j];
          }

          std::vector<unsigned char> point_x = float_to_vector(p.x);
          std::vector<unsigned char> point_y = float_to_vector(p.y);
          std::vector<unsigned char> point_z = float_to_vector(p.z);
          byte_message.insert(byte_message.end(), point_x.begin(), point_x.end());
          byte_message.insert(byte_message.end(), point_y.begin(), point_y.end());
          byte_message.insert(byte_message.end(), point_z.begin(), point_z.end());
        }
      }


      return byte_message;
    }

    void object_detection_3d_callback( angel_msgs::msg::ObjectDetection3dSet::SharedPtr
                                       const detection_msg )
    {
        _detection_mutex.lock();
        _detections.insert(_detections.end(), detection_msg);
        _detection_mutex.unlock();
    }

    void TCPServerVideoThread(int port)
    {
      rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
      rclcpp::Publisher<angel_msgs::msg::HeadsetPoseData>::SharedPtr pose_publisher_;
      std::string frame_id;
      int recv_buf_hdr_len = VIDEO_HEADER_LEN;
      char * recv_buf_hdr = new char[recv_buf_hdr_len];
      char * recv_buf = new char[DEFAULT_READ_SIZE];

      unsigned int frames_received = 0;

      publisher_ = this->create_publisher<sensor_msgs::msg::Image>(PORT_TOPIC_MAP[port], 10);

      if (port == PV_TCP_PORT)
      {
        pose_publisher_ = this->create_publisher<angel_msgs::msg::HeadsetPoseData>("HeadsetPoseData", 10);
      }
      frame_id = PORT_TOPIC_MAP[port];
      std::cout << "[" << frame_id << "] Created publisher\n";

      SOCKET cs = connectSocket(port);

      std::chrono::steady_clock::time_point time_prev = std::chrono::steady_clock::now();

      while (true)
      {
        recv(cs, recv_buf_hdr, recv_buf_hdr_len, 0);

        if (!(((unsigned char) recv_buf_hdr[0] == 0x1A)
          && ((unsigned char) recv_buf_hdr[1] == 0xCF)
          && ((unsigned char) recv_buf_hdr[2] == 0xFC)
          && ((unsigned char) recv_buf_hdr[3] == 0x1D)))
        {
          std::cout << "[" << frame_id << "] sync mismatch!" <<  std::to_string((unsigned char)recv_buf_hdr[0]) << std::endl;
          break;
        }

        int total_bytes_read = 0;
        unsigned int bytes_remaining, height, width, read_size;
        std::vector<unsigned char> frame_data;

        // length in sent message includes width and height (8 bytes)
        // so subtract that to get frame length
        int frame_length = (((unsigned char)recv_buf_hdr[4] << 24) |
                            ((unsigned char)recv_buf_hdr[5] << 16) |
                            ((unsigned char)recv_buf_hdr[6] << 8) |
                            ((unsigned char)recv_buf_hdr[7] << 0)) - 8;

        width = (((unsigned char)recv_buf_hdr[8] << 24) |
                 ((unsigned char)recv_buf_hdr[9] << 16) |
                   ((unsigned char)recv_buf_hdr[10] << 8) |
                   ((unsigned char)recv_buf_hdr[11] << 0));

        height = (((unsigned char)recv_buf_hdr[12] << 24) |
                  ((unsigned char)recv_buf_hdr[13] << 16) |
                  ((unsigned char)recv_buf_hdr[14] << 8) |
                  ((unsigned char)recv_buf_hdr[15] << 0));

        while (total_bytes_read != (frame_length))
        {
          bytes_remaining = frame_length - total_bytes_read;

          if (DEFAULT_READ_SIZE > bytes_remaining)
          {
            read_size = bytes_remaining;
          }
          else if (DEFAULT_READ_SIZE > frame_length)
          {
            read_size = frame_length;
          }
          else
          {
            read_size = DEFAULT_READ_SIZE;
          }

          int bytes_read = recv(cs, recv_buf, read_size, 0);
          total_bytes_read += bytes_read;

          // append buffer to our frame structure
          frame_data.insert(frame_data.end(), &recv_buf[0], &recv_buf[bytes_read]);
        }

        frames_received++;

        //std::cout << std::to_string(frame_data.size()) << std::endl;

        std::chrono::steady_clock::time_point time_now = std::chrono::steady_clock::now();

        if (std::chrono::duration_cast<std::chrono::seconds> (time_now - time_prev).count() >= 1)
        {
          std::cout << frame_id << " frames received: " << std::to_string(frames_received) << std::endl;
          frames_received = 0;
          time_prev = time_now;
        }

        //continue;

        // send ROS message
        sensor_msgs::msg::Image image_message = sensor_msgs::msg::Image();

        image_message.header.stamp = this->now();
        image_message.header.frame_id = frame_id;
        image_message.height = height;
        image_message.width = width;
        image_message.is_bigendian = false;

        if (port == PV_TCP_PORT)
        {
          // create pose message
          angel_msgs::msg::HeadsetPoseData pose_message = angel_msgs::msg::HeadsetPoseData();
          pose_message.header.stamp = image_message.header.stamp;
          pose_message.header.frame_id = image_message.header.frame_id;

          // copy world matrix
          for (int i = 0; i < 64; i+=4)
          {
            float value;
            memcpy(&value, &frame_data[i], sizeof(value));
            pose_message.world_matrix.insert(pose_message.world_matrix.end(), value);
          }

          for (int i = 64; i < 128; i+=4)
          {
            float value;
            memcpy(&value, &frame_data[i], sizeof(value));
            pose_message.projection_matrix.insert(pose_message.projection_matrix.end(), value);
          }
          pose_publisher_->publish(pose_message);

          // convert NV12 image data to RGB8
          cv::Mat nv12_image = cv::Mat(height * 3/2, width, CV_8UC1, &frame_data[128]);
          cv::Mat rgb_image;
          cv::cvtColor(nv12_image, rgb_image, cv::COLOR_YUV2RGB_NV12);
          image_message.encoding = "rgb8";
          image_message.step = width * 3;
          std::vector<unsigned char> v(rgb_image.data, rgb_image.data + height * width * 3);
          image_message.data = v;
        }
        else
        {
          image_message.encoding = "mono8";
          image_message.step = width;
          image_message.data = frame_data;
        }

        publisher_->publish(image_message);
        //std::cout << "Published!" << std::endl;
        //std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }

    void TCPServerAudioThread(int port)
    {
      rclcpp::Publisher<std_msgs::msg::UInt8MultiArray>::SharedPtr publisher_;
      std::string frame_id;
      int recv_buf_hdr_len = AUDIO_HEADER_LEN;

      char * recv_buf_hdr = new char[recv_buf_hdr_len];
      char * recv_buf = new char[DEFAULT_READ_SIZE];
      unsigned int frames_received = 0;

      publisher_ = this->create_publisher<std_msgs::msg::UInt8MultiArray>(PORT_TOPIC_MAP[port], 10);
      frame_id = PORT_TOPIC_MAP[port];
      std::cout << "Created " << PORT_TOPIC_MAP[port] << " publisher\n";

      SOCKET cs = connectSocket(port);

      std::chrono::steady_clock::time_point time_prev = std::chrono::steady_clock::now();

      while (true)
      {
        recv(cs, recv_buf_hdr, recv_buf_hdr_len, 0);

        if (!(((unsigned char) recv_buf_hdr[0] == 0x1A)
          && ((unsigned char) recv_buf_hdr[1] == 0xCF)
          && ((unsigned char) recv_buf_hdr[2] == 0xFC)
          && ((unsigned char) recv_buf_hdr[3] == 0x1D)))
        {
          std::cout << frame_id << ": sync mismatch!";
          break;
        }

        int total_bytes_read = 0;
        unsigned int bytes_remaining, read_size;
        std::vector<unsigned char> frame_data;

        int data_length = (((unsigned char)recv_buf_hdr[4] << 24) |
                           ((unsigned char)recv_buf_hdr[5] << 16) |
                           ((unsigned char)recv_buf_hdr[6] << 8) |
                           ((unsigned char)recv_buf_hdr[7] << 0));

        while (total_bytes_read != (data_length))
        {
          bytes_remaining = data_length - total_bytes_read;

          if (DEFAULT_READ_SIZE > bytes_remaining)
          {
            read_size = bytes_remaining;
          }
          else if (DEFAULT_READ_SIZE > data_length)
          {
            read_size = data_length;
          }
          else
          {
            read_size = DEFAULT_READ_SIZE;
          }

          int bytes_read = recv(cs, recv_buf, read_size, 0);
          total_bytes_read += bytes_read;

          // append buffer to our frame structure
          frame_data.insert(frame_data.end(), &recv_buf[0], &recv_buf[bytes_read]);
        }

        frames_received++;

        std::chrono::steady_clock::time_point time_now = std::chrono::steady_clock::now();

        if (std::chrono::duration_cast<std::chrono::seconds> (time_now - time_prev).count() >= 1)
        {
          std::cout << frame_id << " frames received: " << std::to_string(frames_received) << std::endl;
          frames_received = 0;
          time_prev = time_now;
        }

        //continue;

        // send ROS message
        std_msgs::msg::UInt8MultiArray message = std_msgs::msg::UInt8MultiArray();
        message.data = frame_data;
        publisher_->publish(message);
        //std::cout << "Published audio!" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }

    void TCPServerSMThread(int port)
    {
      rclcpp::Publisher<angel_msgs::msg::SpatialMesh>::SharedPtr publisher_;
      std::string frame_id;
      int recv_buf_hdr_len = SM_HEADER_LEN;

      char * recv_buf_hdr = new char[recv_buf_hdr_len];
      char * recv_buf = new char[DEFAULT_READ_SIZE];
      unsigned int frames_received = 0;

      publisher_ = this->create_publisher<angel_msgs::msg::SpatialMesh>(PORT_TOPIC_MAP[port], 10);
      frame_id = PORT_TOPIC_MAP[port];
      std::cout << "Created " << PORT_TOPIC_MAP[port] << " publisher\n";

      SOCKET cs = connectSocket(port);

      std::chrono::steady_clock::time_point time_prev = std::chrono::steady_clock::now();

      while (true)
      {
        // send any detections if there are any
        _detection_mutex.lock();
        for (auto d : _detections)
        {
            // serialize the detection message
            std::vector<unsigned char> byte_message = serialize_detection_message(d);

            // send via the TCP socket
            int bytes_sent = send(cs, &byte_message[0], byte_message.size(), 0);
        }
        _detections.clear();
        _detection_mutex.unlock();

        // wait for spatial mesh from Hololens
        recv(cs, recv_buf_hdr, recv_buf_hdr_len, 0);

        if (!(((unsigned char) recv_buf_hdr[0] == 0x1A)
           && ((unsigned char) recv_buf_hdr[1] == 0xCF)
           && ((unsigned char) recv_buf_hdr[2] == 0xFC)
           && ((unsigned char) recv_buf_hdr[3] == 0x1D)))
        {
          std::cout << frame_id << ": sync mismatch!";
          break;
        }

        int total_bytes_read = 0;
        unsigned int bytes_remaining, read_size;
        std::vector<unsigned char> frame_data;

        int data_length = (((unsigned char)recv_buf_hdr[4] << 24) |
                           ((unsigned char)recv_buf_hdr[5] << 16) |
                           ((unsigned char)recv_buf_hdr[6] << 8) |
                           ((unsigned char)recv_buf_hdr[7] << 0));

        while (total_bytes_read != (data_length))
        {
          bytes_remaining = data_length - total_bytes_read;

          if (DEFAULT_READ_SIZE > bytes_remaining)
          {
            read_size = bytes_remaining;
          }
          else if (DEFAULT_READ_SIZE > data_length)
          {
            read_size = data_length;
          }
          else
          {
            read_size = DEFAULT_READ_SIZE;
          }

          int bytes_read = recv(cs, recv_buf, read_size, 0);
          total_bytes_read += bytes_read;

          // append buffer to our frame structure
          frame_data.insert(frame_data.end(), &recv_buf[0], &recv_buf[bytes_read]);
        }

        frames_received++;

        std::chrono::steady_clock::time_point time_now = std::chrono::steady_clock::now();

        if (std::chrono::duration_cast<std::chrono::seconds> (time_now - time_prev).count() >= 1)
        {
          std::cout << frame_id << " frames received: " << std::to_string(frames_received) << std::endl;
          frames_received = 0;
          time_prev = time_now;

        }

        // SpatialMesh TCP message format:
        // 32-bit sync
        // 32-bit length
        // 32-bit mesh ID
        // 32-bit vertex count
        // 32-bit triangle count
        // Vertex list:
        //   - 32 bit x
        //   - 32 bit y
        //   - 32 bit z
        // Triangle list:
        //   - 32 bit index

        // form the spatial mesh message
        angel_msgs::msg::SpatialMesh message = angel_msgs::msg::SpatialMesh();
        message.mesh_id = ((frame_data[0] << 24) |
                           (frame_data[1] << 16) |
                           (frame_data[2] << 8) |
                           (frame_data[3]));
        int vertex_count = ((frame_data[4] << 24) |
                           (frame_data[5] << 16) |
                           (frame_data[6] << 8) |
                           (frame_data[7]));
        int triangle_count = ((frame_data[8] << 24) |
                             (frame_data[9] << 16) |
                             (frame_data[10] << 8) |
                             (frame_data[11]));
        triangle_count /= 3;

        // check if this mesh should be removed from the spatial map
        if ((vertex_count == 0) && (triangle_count == 0))
        {
          message.removal = true;
          message.mesh = shape_msgs::msg::Mesh();
        }
        else
        {
          message.removal = false;
          message.mesh = shape_msgs::msg::Mesh();
          std::vector<shape_msgs::msg::MeshTriangle> triangle_list;
          std::vector<geometry_msgs::msg::Point> vertex_list;

          int vertex_offset = 12;
          for (int i = 0; i < (vertex_count * 12); i += 12)
          {
            geometry_msgs::msg::Point p = geometry_msgs::msg::Point();

            float x;
            float y;
            float z;
            memcpy(&x, &frame_data[vertex_offset + i], sizeof(x));
            memcpy(&y, &frame_data[vertex_offset + i + 4], sizeof(y));
            memcpy(&z, &frame_data[vertex_offset + i + 8], sizeof(z));
            p.x = x;
            p.y = y;
            p.z = z;

            vertex_list.insert(vertex_list.end(), p);
          }

          int triangle_offset = vertex_offset + 12 * vertex_count;
          for (int i = 0; i < (triangle_count * 12); i += 12)
          {
            shape_msgs::msg::MeshTriangle t = shape_msgs::msg::MeshTriangle();
            memcpy(&t.vertex_indices[0], &frame_data[triangle_offset + i], sizeof(t.vertex_indices[0]));
            memcpy(&t.vertex_indices[1], &frame_data[triangle_offset + i + 4], sizeof(t.vertex_indices[1]));
            memcpy(&t.vertex_indices[2], &frame_data[triangle_offset + i + 8], sizeof(t.vertex_indices[2]));
            triangle_list.insert(triangle_list.end(), t);
          }

          message.mesh.vertices = vertex_list;
          message.mesh.triangles = triangle_list;
        }

        publisher_->publish(message);
        //std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }


    SOCKET connectSocket(int port)
    {
      SOCKET s;

#ifdef _WIN32
      s = socket(AF_INET, SOCK_STREAM, 0);
      SOCKADDR_IN addr;
      WSADATA w;
      if (WSAStartup (0x0202, &w))
      {
      }

      addr.sin_family = AF_INET;
      addr.sin_port = htons(port);
      inet_pton(AF_INET, tcp_server_uri.c_str(), &(addr.sin_addr));

      if (connect(s, (SOCKADDR *)&addr, sizeof(addr)) < 0)
      {
          std::cout << "Error creating socket for port: " << port << std::endl;
      }
#endif
#ifdef __linux__
      s = socket(AF_INET, SOCK_STREAM, 0);

      struct sockaddr_in addr;
      addr.sin_family = AF_INET;
      addr.sin_port = htons(port);
      inet_pton(AF_INET, tcp_server_uri.c_str(), &(addr.sin_addr));

      if (connect(s, (struct sockaddr*)&addr, sizeof(addr)) < 0)
      {
          std::cout << "Error creating socket for port: " << port << std::endl;
      }
#endif
      else
      {
        std::cout << "Socket connected!" << std::endl;
      }

      return s;
    }

};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  std::shared_ptr<PTGDataHub> mp = std::make_shared<PTGDataHub>();

  std::thread t1 = mp->StartTCPServerThread(LF_VLC_TCP_PORT);
  std::thread t2 = mp->StartTCPServerThread(RF_VLC_TCP_PORT);
  std::thread t3 = mp->StartTCPServerThread(LL_VLC_TCP_PORT);
  std::thread t4 = mp->StartTCPServerThread(RR_VLC_TCP_PORT);
  std::thread t5 = mp->StartTCPServerThread(PV_TCP_PORT);
  std::thread t6 = mp->StartTCPServerThread(DEPTH_TCP_PORT);
  std::thread t7 = mp->StartTCPServerThread(DEPTH_AB_TCP_PORT);
  std::thread t8 = mp->StartTCPServerThread(LONG_DEPTH_TCP_PORT);
  std::thread t9 = mp->StartTCPServerThread(LONG_DEPTH_AB_TCP_PORT);
  std::thread t10 = mp->StartTCPServerThread(AUDIO_TCP_PORT);
  std::thread t11 = mp->StartTCPServerThread(SM_TCP_PORT);

  rclcpp::spin(mp);

  t1.join();
  t2.join();
  t3.join();
  t4.join();
  t5.join();
  t6.join();
  t7.join();
  t8.join();
  t9.join();
  t10.join();
  t11.join();

  rclcpp::shutdown();
  return 0;
}
