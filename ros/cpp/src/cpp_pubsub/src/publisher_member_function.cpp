#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <thread>
#include <iostream>
#include <opencv2/opencv2.hpp>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/byte_multi_array.hpp"


#define WIN32_LEAN_AND_MEAN

#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>

// Need to link with Ws2_32.lib, Mswsock.lib, and Advapi32.lib
#pragma comment (lib, "Ws2_32.lib")
#pragma comment (lib, "Mswsock.lib")
#pragma comment (lib, "AdvApi32.lib")


#define HEADER_LEN        (8)
#define DEFAULT_READ_SIZE (8192)
#define DEFAULT_BUFLEN    (1024 * 1024)

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
    //rclcpp::Publisher<std_msgs::msg::ByteMultiArray>::SharedPtr publisher_;
    size_t count_;
    
    void TCPServerThread(int port)
    {
      rclcpp::Publisher<std_msgs::msg::ByteMultiArray>::SharedPtr publisher_;

      if (port == 11000) 
      {
        publisher_ = this->create_publisher<std_msgs::msg::ByteMultiArray>("LFFrames", 10);
        std::cout << "Created LFFrame publisher\n";
      } 
      else if (port == 11001)
      {
        publisher_ = this->create_publisher<std_msgs::msg::ByteMultiArray>("RFFrames", 10);
        std::cout << "Created RFFrame publisher\n";
      }
      else if (port == 11002)
      {
        publisher_ = this->create_publisher<std_msgs::msg::ByteMultiArray>("LLFrames", 10);
        std::cout << "Created LLFrame publisher\n";
      }
      else if (port == 11003)
      {
        publisher_ = this->create_publisher<std_msgs::msg::ByteMultiArray>("RRFrames", 10);
        std::cout << "Created RRFrame publisher\n";
      }
      else
      {
        return;
      }

      SOCKET s, cs;
      WSADATA w;
      if (WSAStartup (0x0202, &w))
      {
          return;
      }

      SOCKADDR_IN addr;
      addr.sin_family = AF_INET;
      addr.sin_port = htons(port);
      inet_pton(AF_INET, "169.254.103.120", &(addr.sin_addr));

      s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
      if (s == INVALID_SOCKET)
      {
          std::cout << "socket invalid\n";
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

      // Accept a client socket
      cs = accept(s, NULL, NULL);
      if (cs == INVALID_SOCKET) {
          closesocket(s);
          WSACleanup();
          return;
      }

      std::cout << "Connected!!\n";
      closesocket(s);

      char recv_buf_hdr[HEADER_LEN];
      int recv_buf_hdr_len = HEADER_LEN;
      char recv_buf[DEFAULT_READ_SIZE];

      while (true)
      {
        int iResult = recv(cs, recv_buf_hdr, recv_buf_hdr_len, 0);

        if (!(((unsigned char) recv_buf_hdr[0] == 0x1A) 
          && ((unsigned char) recv_buf_hdr[1] == 0xCF)
          && ((unsigned char) recv_buf_hdr[2] == 0xFC)
          && ((unsigned char) recv_buf_hdr[3] == 0x1D)))
        {
          std::cout << "Sync mismatch!";
          break;
        }

        unsigned int total_message_length;
        int total_bytes_read = 0;
        unsigned int bytes_remaining;
        unsigned int read_size;
        
        total_message_length = (((unsigned char)recv_buf_hdr[4] << 24) |
                                ((unsigned char)recv_buf_hdr[5] << 16) | 
                                ((unsigned char)recv_buf_hdr[6] << 8) | 
                                ((unsigned char)recv_buf_hdr[7] << 0));
        //std::cout << "Message length " << std::to_string(total_message_length);

        std::vector<unsigned char> frame_data;

        while (total_bytes_read != total_message_length)
        {
          bytes_remaining = total_message_length - total_bytes_read;
          
          if (DEFAULT_READ_SIZE > bytes_remaining)
          {
            read_size = bytes_remaining;
          }
          else if (DEFAULT_READ_SIZE > total_message_length)
          {
            read_size = total_message_length;
          }
          else
          {
            read_size = DEFAULT_READ_SIZE;
          }

          int bytes_read = recv(cs, recv_buf, read_size, 0);
          total_bytes_read += bytes_read;

          //std::cout << "Bytes read: " << std::to_string(bytes_read) << " " << std::to_string(read_size) << std::endl;
             
          // append buffer to our frame structure
          frame_data.insert(frame_data.end(), &recv_buf[0], &recv_buf[bytes_read]);

          //std::cout << "Frame size " << std::to_string(frame_data.size()) << std::endl;
        }

        //std::cout << "Frame size " << std::to_string(frame_data.size()) << std::endl;

        // send ROS message
        auto message = std_msgs::msg::ByteMultiArray();
        message.data = frame_data;
        publisher_->publish(message);
        //std::cout << "Published!" << std::endl;

        std::this_thread::sleep_for(std::chrono::milliseconds(5));

      }
    }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  std::shared_ptr<MinimalPublisher> mp = std::make_shared<MinimalPublisher>();

  // start the server threads
  std::thread t1 = mp->StartTCPServerThread(11000);
  std::thread t2 = mp->StartTCPServerThread(11001);
  std::thread t3 = mp->StartTCPServerThread(11002);
  std::thread t4 = mp->StartTCPServerThread(11003);

  rclcpp::spin(mp);

  t1.join();
  t2.join();
  t3.join();
  t4.join();

  rclcpp::shutdown();
  return 0;
}