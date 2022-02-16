#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <thread>
#include <iostream>
#include <map>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/byte_multi_array.hpp"
#include "std_msgs/msg/u_int8_multi_array.hpp"

#include "sensor_msgs/msg/image.hpp"

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

#define AUDIO_HEADER_LEN  (8)
#define VIDEO_HEADER_LEN  (16)
#define DEFAULT_READ_SIZE (8192)
#define DEFAULT_BUFLEN    (1024 * 1024)

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

using namespace std::chrono_literals;

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
    { AUDIO_TCP_PORT,         "AudioData" }

};

class MinimalPublisher : public rclcpp::Node
{
  public:
    MinimalPublisher()
    : Node("minimal_publisher"), count_(0)
    {}

    std::thread StartTCPServerThread(int port)
    {
      std::thread t;
      if ((port < LF_VLC_TCP_PORT) || (port > AUDIO_TCP_PORT))
      {
        std::cout << "Invalid port number entered!\n";
      }

      if (port == AUDIO_TCP_PORT)
      {
        t = std::thread(&MinimalPublisher::TCPServerAudioThread, this, port);
      }
      else
      {
        t = std::thread(&MinimalPublisher::TCPServerVideoThread, this, port);
      }

      return t;
    }

  private:
    size_t count_;
    std::string server_ip_addr = "192.167.1.90";

    void TCPServerVideoThread(int port)
    {
      rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
      std::string frame_id;
      int recv_buf_hdr_len = VIDEO_HEADER_LEN;
      char * recv_buf_hdr = new char[recv_buf_hdr_len];
      char * recv_buf = new char[DEFAULT_READ_SIZE];

      unsigned int frames_received = 0;

      publisher_ = this->create_publisher<sensor_msgs::msg::Image>(PORT_TOPIC_MAP[port], 10);
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
          std::cout << frame_id << ": sync mismatch!" <<  std::to_string((unsigned char)recv_buf_hdr[0]) << std::endl;
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
        sensor_msgs::msg::Image message = sensor_msgs::msg::Image();

        message.header.stamp = this->now();
        message.header.frame_id = frame_id;
        message.height = height;
        message.width = width;
        message.is_bigendian = false;

        if (port == PV_TCP_PORT)
        {
          message.encoding = "rgb8";
          message.step = 1280;
        }
        else
        {
          message.encoding = "mono8";
          message.step = width;
        }

        message.data = frame_data;

        publisher_->publish(message);
        //std::cout << "Published!" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
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

        //std::cout << "Got something!!\n";

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
        std_msgs::msg::UInt8MultiArray message = std_msgs::msg::UInt8MultiArray();
        message.data = frame_data;
        publisher_->publish(message);
        //std::cout << "Published audio!" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }


    SOCKET connectSocket(int port)
    {
      SOCKET s, cs;

#ifdef _WIN32
      SOCKADDR_IN addr;
      WSADATA w;
      if (WSAStartup (0x0202, &w))
      {
      }

      addr.sin_family = AF_INET;
      addr.sin_port = htons(port);
      inet_pton(AF_INET, server_ip_addr.c_str(), &(addr.sin_addr));

      s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
      if (s == INVALID_SOCKET)
      {
          std::cout << "Error creating socket" << std::endl;
      }
      if (::bind(s, (SOCKADDR *)&addr, sizeof(addr)) == SOCKET_ERROR)
      {
          std::cout << "Socket bind error" << std::endl;
          std::cout << std::to_string(WSAGetLastError());
      }
      if (listen(s, SOMAXCONN ) == SOCKET_ERROR)
      {
          closesocket(s);
          WSACleanup();
      }

      std::cout << "Listening for connection..." << std::endl;
      cs = accept(s, NULL, NULL);
      if (cs == INVALID_SOCKET)
      {
          std::cout << "Invalid socket!\n";
          closesocket(s);
          WSACleanup();
      }

      std::cout << "Connected port " << port << std::endl;
      closesocket(s);
      WSACleanup();
#endif
#ifdef __linux__
      struct sockaddr_in addr;

      addr.sin_family = AF_INET;
      addr.sin_port = htons(port);
      inet_pton(AF_INET, server_ip_addr.c_str(), &(addr.sin_addr));

      s = socket(AF_INET, SOCK_STREAM, 0);
      if (s < 0)
      {
          std::cout << "Error creating socket" << std::endl;
      }
      if (bind(s, (struct sockaddr *)&addr, sizeof(addr)) < 0)
      {
          std::cout << "Socket bind error" << std::endl;
      }
      listen(s, SOMAXCONN);

      std::cout << "Listening for connection..." << std::endl;
      cs = accept(s, NULL, NULL);
      if (cs == -1)
      {
          std::cout << "Invalid socket!" << std::endl;
          close(s);
      }

      std::cout << "Connected port " << port << std::endl;
      close(s);
#endif

      return cs;
    }

};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  std::shared_ptr<MinimalPublisher> mp = std::make_shared<MinimalPublisher>();

  std::cout << "Hi starting threads!" << std::endl;

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

  rclcpp::shutdown();
  return 0;
}
