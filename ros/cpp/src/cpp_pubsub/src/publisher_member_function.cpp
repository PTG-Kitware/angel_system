#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <thread>
#include <iostream>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/byte_multi_array.hpp"
#include "sensor_msgs/msg/image.hpp"

#define WIN32_LEAN_AND_MEAN

#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>

// Need to link with Ws2_32.lib, Mswsock.lib, and Advapi32.lib
#pragma comment (lib, "Ws2_32.lib")
#pragma comment (lib, "Mswsock.lib")
#pragma comment (lib, "AdvApi32.lib")

#define HEADER_LEN        (16)
#define DEFAULT_READ_SIZE (8192)
#define DEFAULT_BUFLEN    (1024 * 1024)
#define LF_VLC_TCP_PORT   (11000)
#define RF_VLC_TCP_PORT   (11001)
#define LL_VLC_TCP_PORT   (11002)
#define RR_VLC_TCP_PORT   (11003)
#define DEPTH_TCP_PORT    (11004)
#define DEPTH_AB_TCP_PORT (11005)
#define PV_TCP_PORT       (11006)


using namespace std::chrono_literals;

class MinimalPublisher : public rclcpp::Node
{
  public:
    MinimalPublisher()
    : Node("minimal_publisher"), count_(0)
    {
      //publisher_ = this->create_publisher<std_msgs::msg::ByteMultiArray>("LFFrames", 10);
    }

    std::thread StartTCPServerThread(int port)
    {
      std::thread t = std::thread(&MinimalPublisher::TCPServerThread, this, port);
      return t;
    }

  private:
    size_t count_;
    std::string server_ip_addr = "169.254.103.120";
    
    void TCPServerThread(int port)
    {
      rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
      std::string frame_id;
      SOCKET s, cs;
      WSADATA w;
      SOCKADDR_IN addr;

      char recv_buf_hdr[HEADER_LEN];
      int recv_buf_hdr_len = HEADER_LEN;
      char recv_buf[DEFAULT_READ_SIZE];
      unsigned int frames_received = 0;

      if (port == LF_VLC_TCP_PORT) 
      {
        publisher_ = this->create_publisher<sensor_msgs::msg::Image>("LFFrames", 10);
        frame_id = "LFFrame VLC";
        std::cout << "Created LFFrame publisher\n";
      } 
      else if (port == RF_VLC_TCP_PORT)
      {
        publisher_ = this->create_publisher<sensor_msgs::msg::Image>("RFFrames", 10);
        frame_id = "RFFrame VLC";
        std::cout << "Created RFFrame publisher\n";
      }
      else if (port == LL_VLC_TCP_PORT)
      {
        publisher_ = this->create_publisher<sensor_msgs::msg::Image>("LLFrames", 10);
        frame_id = "LLFrame VLC";
        std::cout << "Created LLFrame publisher\n";
      }
      else if (port == RR_VLC_TCP_PORT)
      {
        publisher_ = this->create_publisher<sensor_msgs::msg::Image>("RRFrames", 10);
        frame_id = "RRFrame VLC";
        std::cout << "Created RRFrame publisher\n";
      }
      else if (port == PV_TCP_PORT)
      {
        publisher_ = this->create_publisher<sensor_msgs::msg::Image>("PVFrames", 10);
        frame_id = "PV camera";
        std::cout << "Created PV publisher\n";
      }
      else if (port == DEPTH_TCP_PORT)
      {
        publisher_ = this->create_publisher<sensor_msgs::msg::Image>("DepthFrames", 10);
        frame_id = "Depth camera";
        std::cout << "Created Depth publisher\n";
      }
      else if (port == DEPTH_AB_TCP_PORT)
      {
        publisher_ = this->create_publisher<sensor_msgs::msg::Image>("DepthABFrames", 10);
        frame_id = "Depth AB camera";
        std::cout << "Created AB Depth publisher\n";
      }
      else
      {
        std::cout << "Invalid port number entered!\n";
        return;
      }

      if (WSAStartup (0x0202, &w))
      {
          return;
      }

      addr.sin_family = AF_INET;
      addr.sin_port = htons(port);
      inet_pton(AF_INET, server_ip_addr.c_str(), &(addr.sin_addr));

      s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
      if (s == INVALID_SOCKET)
      {
          return;
      }
      if (::bind(s, (SOCKADDR *)&addr, sizeof(addr)) == SOCKET_ERROR)
      {
          std::cout << "socket error\n";
          std::cout << std::to_string(WSAGetLastError());
          return;
      }
      if (listen(s, SOMAXCONN ) == SOCKET_ERROR)
      {
          closesocket(s);
          WSACleanup();
          return;
      }

      std::cout << "Listening for connection...\n";
      cs = accept(s, NULL, NULL);
      if (cs == INVALID_SOCKET) {
          closesocket(s);
          WSACleanup();
          return;
      }

      std::cout << "Connected!!\n";
      closesocket(s);

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

        unsigned int frame_length, height, width;
        int total_bytes_read = 0;
        unsigned int bytes_remaining;
        unsigned int read_size;
        std::vector<unsigned char> frame_data;

        // length in sent message includes width and height (8 bytes)
        // so subtract that to get frame length
        frame_length = (((unsigned char)recv_buf_hdr[4] << 24) |
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
        else if ((port == DEPTH_TCP_PORT) || (port == DEPTH_AB_TCP_PORT))
        {
          message.encoding = "mono16";
          message.step = width * 2;
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
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  /*
  int port;
  if (argc >= 2)
  {
    std::istringstream iss( argv[1] );

    if (!(iss >> port))
    {
      return 0;
    }
  }
  else 
  {
    std::cout << "Please enter port number" << std::endl;
    return 0;
  }
  */

  std::shared_ptr<MinimalPublisher> mp = std::make_shared<MinimalPublisher>();

  // start the server threads
  std::thread t1 = mp->StartTCPServerThread(LF_VLC_TCP_PORT);
  std::thread t2 = mp->StartTCPServerThread(RF_VLC_TCP_PORT);
  std::thread t3 = mp->StartTCPServerThread(LL_VLC_TCP_PORT);
  std::thread t4 = mp->StartTCPServerThread(RR_VLC_TCP_PORT);
  std::thread t5 = mp->StartTCPServerThread(PV_TCP_PORT);
  std::thread t6 = mp->StartTCPServerThread(DEPTH_TCP_PORT);
  std::thread t7 = mp->StartTCPServerThread(DEPTH_AB_TCP_PORT);

  rclcpp::spin(mp);

  t1.join();
  t2.join();
  t3.join();
  t4.join();
  t5.join();
  t6.join();
  t7.join();

  // start the server threads
  //std::thread t1 = mp->StartTCPServerThread(port);
  //rclcpp::spin(mp);
  //t1.join();

  rclcpp::shutdown();
  return 0;
}