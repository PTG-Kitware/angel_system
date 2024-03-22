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
#include <rclcpp_components/register_node_macro.hpp>
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/byte_multi_array.hpp"
#include "std_msgs/msg/u_int8_multi_array.hpp"

#include "sensor_msgs/msg/image.hpp"
#include "angel_msgs/msg/spatial_mesh.hpp"
#include "angel_msgs/msg/headset_pose_data.hpp"
#include "angel_msgs/msg/object_detection3d_set.hpp"
#include "angel_msgs/msg/task_update.hpp"

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
#define TASK_UPDATE_TCP_PORT   (11011)

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
    { SM_TCP_PORT,            "SpatialMapData" },
    { TASK_UPDATE_TCP_PORT,   "TaskUpdates" }
};

using namespace std::chrono_literals;
using std::placeholders::_1;

namespace angel_datahub {

namespace {

// ----------------------------------------------------------------------------
#define DEFINE_PARAM_NAME( var_name, param_name_str ) \
  static constexpr char const* var_name = param_name_str

// IP address of the Hololens
DEFINE_PARAM_NAME( PARAM_TCP_SERVER_ADDR, "tcp_server_uri" );
DEFINE_PARAM_NAME( PARAM_TOPIC_INPUT_DET_3D, "det_3d_topic" );
DEFINE_PARAM_NAME( PARAM_TOPIC_OUTPUT_HEADSET_POSE, "headset_pose_topic" );
DEFINE_PARAM_NAME( PARAM_TOPIC_INPUT_TASK_UPDATE, "task_update_topic" );

#undef DEFINE_PARAM_NAME

} // namespace

class AngelDataHub : public rclcpp::Node
{
  public:
    AngelDataHub( rclcpp::NodeOptions const& options );
    ~AngelDataHub();

  private:
    rclcpp::Subscription< angel_msgs::msg::ObjectDetection3dSet >::SharedPtr _object_3d_subscriber;
    rclcpp::Subscription< angel_msgs::msg::TaskUpdate>::SharedPtr _task_update_subscriber;
    std::string tcp_server_uri;
    std::string det_3d_topic;
    std::string headset_pose_topic;
    std::string task_update_topic;
    std::vector<angel_msgs::msg::ObjectDetection3dSet::SharedPtr> _detections;
    std::mutex _detection_mutex;

    // thread objects for each TCP socket
    std::thread lf_vlc_t;
    std::thread rf_vlc_t;
    std::thread ll_vlc_t;
    std::thread rr_vlc_t;
    std::thread depth_t;
    std::thread depth_ab_t;
    std::thread long_depth_t;
    std::thread long_depth_ab_t;
    std::thread pv_t;
    std::thread audio_t;
    std::thread sm_t;
    std::thread tu_t;

    // socket for sending task updates to the Hololens
    SOCKET task_update_socket = -1;

    std::vector<unsigned char> uint_to_vector(unsigned int x);
    std::vector<unsigned char> float_to_vector(float x);
    std::vector<char> string_to_vector(std::string x);

    std::vector<unsigned char> serialize_detection_message(
      angel_msgs::msg::ObjectDetection3dSet::SharedPtr const detection_msg );
    std::vector<unsigned char> serialize_task_update_message(
      angel_msgs::msg::TaskUpdate::SharedPtr const task_update_msg );

    void object_detection_3d_callback( angel_msgs::msg::ObjectDetection3dSet::SharedPtr
                                       const detection_msg );
    void task_update_callback( angel_msgs::msg::TaskUpdate::SharedPtr
                                       const task_update_msg );

    void TCPServerVideoThread(int port);
    void TCPServerAudioThread(int port);
    void TCPServerSMThread(int port);
    void ConnectTaskUpdateSocket();
    SOCKET connectSocket(int port);

};

AngelDataHub
::AngelDataHub( rclcpp::NodeOptions const& options )
  : Node("AngelDataHub", options)
{
  auto log = this->get_logger();

  // This two-stage declare->get allows the lack of passing a parameter to
  // throw an error with the parameter name in the error so the user has a
  // clue what is going wrong.
  declare_parameter( PARAM_TCP_SERVER_ADDR );
  declare_parameter( PARAM_TOPIC_INPUT_DET_3D );
  declare_parameter( PARAM_TOPIC_OUTPUT_HEADSET_POSE );
  declare_parameter( PARAM_TOPIC_INPUT_TASK_UPDATE );
  // ROS2 Iron version
  //declare_parameter< std::string >( PARAM_TCP_SERVER_ADDR );
  //declare_parameter< std::string >( PARAM_TOPIC_INPUT_DET_3D );
  //declare_parameter< std::string >( PARAM_TOPIC_OUTPUT_HEADSET_POSE );
  //declare_parameter< std::string >( PARAM_TOPIC_INPUT_TASK_UPDATE );

  tcp_server_uri =
    this->get_parameter( PARAM_TCP_SERVER_ADDR ).as_string();
  det_3d_topic =
    this->get_parameter( PARAM_TOPIC_INPUT_DET_3D ).as_string();
  headset_pose_topic =
    this->get_parameter( PARAM_TOPIC_OUTPUT_HEADSET_POSE ).as_string();
  task_update_topic =
    this->get_parameter( PARAM_TOPIC_INPUT_TASK_UPDATE ).as_string();

  RCLCPP_INFO( log,
               "Starting datahub, intending to connect to TCP server @ %s",
               tcp_server_uri.c_str() );

  // create ROS subscribers
  _object_3d_subscriber = this->create_subscription<angel_msgs::msg::ObjectDetection3dSet>(
    det_3d_topic, 100,
    std::bind(&AngelDataHub::object_detection_3d_callback, this, _1 )
  );

  _task_update_subscriber = this->create_subscription<angel_msgs::msg::TaskUpdate>(
    task_update_topic, 100,
    std::bind(&AngelDataHub::task_update_callback, this, _1 )
  );

  // start the video threads
  lf_vlc_t = std::thread(&AngelDataHub::TCPServerVideoThread, this, LF_VLC_TCP_PORT);
  rf_vlc_t = std::thread(&AngelDataHub::TCPServerVideoThread, this, RF_VLC_TCP_PORT);
  ll_vlc_t = std::thread(&AngelDataHub::TCPServerVideoThread, this, LL_VLC_TCP_PORT);
  rr_vlc_t = std::thread(&AngelDataHub::TCPServerVideoThread, this, RR_VLC_TCP_PORT);
  depth_t = std::thread(&AngelDataHub::TCPServerVideoThread, this, DEPTH_TCP_PORT);
  depth_ab_t = std::thread(&AngelDataHub::TCPServerVideoThread, this, DEPTH_AB_TCP_PORT);
  long_depth_t = std::thread(&AngelDataHub::TCPServerVideoThread, this, LONG_DEPTH_TCP_PORT);
  long_depth_ab_t = std::thread(&AngelDataHub::TCPServerVideoThread, this, LONG_DEPTH_AB_TCP_PORT);
  pv_t = std::thread(&AngelDataHub::TCPServerVideoThread, this, PV_TCP_PORT);

  // start the audio and spatial mapping threads
  audio_t = std::thread(&AngelDataHub::TCPServerAudioThread, this, AUDIO_TCP_PORT);
  sm_t = std::thread(&AngelDataHub::TCPServerSMThread, this, SM_TCP_PORT);

  // attempt to connect to the task update TCP server
  tu_t = std::thread(&AngelDataHub::ConnectTaskUpdateSocket, this);
}

AngelDataHub::~AngelDataHub()
{
  tu_t.join();
  lf_vlc_t.join();
  rf_vlc_t.join();
  ll_vlc_t.join();
  rr_vlc_t.join();
  depth_t.join();
  depth_ab_t.join();
  long_depth_t.join();
  long_depth_ab_t.join();
  pv_t.join();
  audio_t.join();
  sm_t.join();
}

std::vector<unsigned char> AngelDataHub::uint_to_vector(unsigned int x)
{
  std::vector<unsigned char> v(4);
  memcpy(&v[0], &x, sizeof(x));
  return v;
}

std::vector<unsigned char> AngelDataHub::float_to_vector(float x)
{
  std::vector<unsigned char> v(4);
  memcpy(&v[0], &x, sizeof(x));
  return v;
}

std::vector<char> AngelDataHub::string_to_vector(std::string x)
{
  std::vector<char> v(x.begin(), x.end());
  v.push_back('\0');
  return v;
}

std::vector<unsigned char> AngelDataHub::serialize_detection_message( angel_msgs::msg::ObjectDetection3dSet::SharedPtr
                                                        const detection_msg )
{
  // serialize the detection message
  std::vector<unsigned char> byte_message;

  unsigned int ros_message_length = 0;

  // Angel header:
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
  std::vector<char> frame_id_bytes = string_to_vector(detection_msg->header.frame_id);
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

std::vector<unsigned char> AngelDataHub::serialize_task_update_message( angel_msgs::msg::TaskUpdate::SharedPtr
                                                        const task_update_msg )
{
  auto log = this->get_logger();
  std::vector<unsigned char> byte_message;
  unsigned int ros_message_length = 0;

  // Angel header:
  //   -- 32-bit sync = 4 bytes
  //   -- 32-bit ros msg length = 4 bytes
  // ROS2 message:
  //  header
  //   -- 32 bit seconds = 4 bytes
  //   -- 32 bit nanoseconds = 4 bytes
  //   -- frame id string
  //  task_name
  //   -- string
  //  num steps
  //   -- int
  //  steps
  //   -- string list
  //  current_step
  //   -- string
  //  previous_step
  //   -- string
  //  current_activity
  //   -- string
  //  next_activity
  //   -- string

  // convert the frame id to bytes
  std::vector<char> frame_id_bytes = string_to_vector(task_update_msg->header.frame_id);
  ros_message_length += frame_id_bytes.size();

  // convert task name to bytes
  std::vector<char> task_name_bytes = string_to_vector(task_update_msg->task_name);
  ros_message_length += task_name_bytes.size();

  // convert step list to bytes
  std::vector<char> steps_bytes;
  for (unsigned int i = 0; i < task_update_msg->steps.size(); i++)
  {
    steps_bytes.insert(steps_bytes.end(),
                       task_update_msg->steps[i].begin(),
                       task_update_msg->steps[i].end());
    steps_bytes.push_back('\0');
  }
  ros_message_length += steps_bytes.size();

  // convert current step to bytes
  std::vector<char> curr_step_bytes = string_to_vector(task_update_msg->current_step);
  ros_message_length += curr_step_bytes.size();

  // convert previous step to bytes
  std::vector<char> prev_step_bytes = string_to_vector(task_update_msg->previous_step);
  ros_message_length += prev_step_bytes.size();

  // convert current activity to bytes
  std::vector<char> curr_activity_bytes = string_to_vector(task_update_msg->current_activity);
  ros_message_length += curr_activity_bytes.size();

  // convert next activity to bytes
  std::vector<char> next_activity_bytes = string_to_vector(task_update_msg->next_activity);
  ros_message_length += next_activity_bytes.size();

  ros_message_length += 12; // ROS header timestamp size + num bytes size

  // add sync
  std::vector<unsigned char> sync = uint_to_vector(0x1ACFFC1D);
  byte_message.insert(byte_message.end(), sync.begin(), sync.end());

  // add length
  std::vector<unsigned char> length = uint_to_vector(ros_message_length);
  byte_message.insert(byte_message.end(), length.begin(), length.end());

  // add header time stamp
  std::vector<unsigned char> seconds = uint_to_vector(task_update_msg->header.stamp.sec);
  byte_message.insert(byte_message.end(), seconds.begin(), seconds.end());
  std::vector<unsigned char> nanoseconds = uint_to_vector(task_update_msg->header.stamp.nanosec);
  byte_message.insert(byte_message.end(), nanoseconds.begin(), nanoseconds.end());

  // add frame id
  byte_message.insert(byte_message.end(), frame_id_bytes.begin(), frame_id_bytes.end());

  // add task_name
  byte_message.insert(byte_message.end(), task_name_bytes.begin(), task_name_bytes.end());

  // add num steps
  std::vector<unsigned char> num_steps_bytes = uint_to_vector(task_update_msg->steps.size());
  byte_message.insert(byte_message.end(), num_steps_bytes.begin(), num_steps_bytes.end());

  // add steps
  byte_message.insert(byte_message.end(), steps_bytes.begin(), steps_bytes.end());

  // add current_step
  byte_message.insert(byte_message.end(), curr_step_bytes.begin(), curr_step_bytes.end());

  // add previous_step
  byte_message.insert(byte_message.end(), prev_step_bytes.begin(), prev_step_bytes.end());

  // add current_activity
  byte_message.insert(byte_message.end(), curr_activity_bytes.begin(), curr_activity_bytes.end());

  // add next_activity
  byte_message.insert(byte_message.end(), next_activity_bytes.begin(), next_activity_bytes.end());

  return byte_message;
}

void AngelDataHub::object_detection_3d_callback( angel_msgs::msg::ObjectDetection3dSet::SharedPtr
                                   const detection_msg )
{
    _detection_mutex.lock();
    _detections.insert(_detections.end(), detection_msg);
    _detection_mutex.unlock();
}

void AngelDataHub::task_update_callback( angel_msgs::msg::TaskUpdate::SharedPtr
                                   const task_update_msg )
{
  auto log = this->get_logger();
  RCLCPP_INFO( log,
               "Task update message: Task name: %s, step: %s",
               task_update_msg->task_name.c_str(), task_update_msg->current_step.c_str());

  if (task_update_socket != -1)
  {
    // serialize the update message
    std::vector<unsigned char> byte_message = serialize_task_update_message(task_update_msg);

    // send via the TCP socket
    unsigned int bytes_sent = send(task_update_socket, &byte_message[0], byte_message.size(), 0);
    if (bytes_sent != byte_message.size())
    {
      RCLCPP_WARN(log, "Did not send full detection message");
    }
  }
}

void AngelDataHub::ConnectTaskUpdateSocket()
{
  task_update_socket = connectSocket(TASK_UPDATE_TCP_PORT);
}

void AngelDataHub::TCPServerVideoThread(int port)
{
  auto log = this->get_logger();

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
    pose_publisher_ = this->create_publisher<angel_msgs::msg::HeadsetPoseData>(headset_pose_topic, 10);
  }
  frame_id = PORT_TOPIC_MAP[port];
  RCLCPP_INFO( log,
                "[%s] Created publisher",
               frame_id.c_str() );

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
      RCLCPP_WARN( log,
                   "[%s] sync mismatch!",
                   frame_id.c_str() );
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

    std::chrono::steady_clock::time_point time_now = std::chrono::steady_clock::now();

    if (std::chrono::duration_cast<std::chrono::seconds> (time_now - time_prev).count() >= 1)
    {
      RCLCPP_INFO( log,
                   "[%s] frames received: %d",
                   frame_id.c_str(), frames_received );
      frames_received = 0;
      time_prev = time_now;
    }

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
  }
}

void AngelDataHub::TCPServerAudioThread(int port)
{
  auto log = this->get_logger();

  rclcpp::Publisher<std_msgs::msg::UInt8MultiArray>::SharedPtr publisher_;
  std::string frame_id;
  int recv_buf_hdr_len = AUDIO_HEADER_LEN;

  char * recv_buf_hdr = new char[recv_buf_hdr_len];
  char * recv_buf = new char[DEFAULT_READ_SIZE];
  unsigned int frames_received = 0;

  publisher_ = this->create_publisher<std_msgs::msg::UInt8MultiArray>(PORT_TOPIC_MAP[port], 10);
  frame_id = PORT_TOPIC_MAP[port];
  RCLCPP_INFO( log,
                "[%s] Created publisher",
               frame_id.c_str() );

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
      RCLCPP_WARN( log,
                   "[%s] sync mismatch!",
                   frame_id.c_str() );
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
      RCLCPP_INFO( log,
                   "[%s] frames received: %d",
                   frame_id.c_str(), frames_received );
      frames_received = 0;
      time_prev = time_now;
    }

    // send ROS message
    std_msgs::msg::UInt8MultiArray message = std_msgs::msg::UInt8MultiArray();
    message.data = frame_data;
    publisher_->publish(message);
  }
}

void AngelDataHub::TCPServerSMThread(int port)
{
  auto log = this->get_logger();

  rclcpp::Publisher<angel_msgs::msg::SpatialMesh>::SharedPtr publisher_;
  std::string frame_id;
  int recv_buf_hdr_len = SM_HEADER_LEN;

  char * recv_buf_hdr = new char[recv_buf_hdr_len];
  char * recv_buf = new char[DEFAULT_READ_SIZE];
  unsigned int frames_received = 0;

  publisher_ = this->create_publisher<angel_msgs::msg::SpatialMesh>(PORT_TOPIC_MAP[port], 10);
  frame_id = PORT_TOPIC_MAP[port];
  RCLCPP_INFO( log,
                "[%s] Created publisher",
               frame_id.c_str() );

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
      unsigned int bytes_sent = send(cs, &byte_message[0], byte_message.size(), 0);
      if (bytes_sent != byte_message.size())
      {
        RCLCPP_WARN(log, "Did not send full detection message");
      }
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
      RCLCPP_WARN( log,
                   "[%s] sync mismatch!",
                   frame_id.c_str() );
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
      RCLCPP_INFO( log,
                   "[%s] frames received: %d",
                   frame_id.c_str(), frames_received );
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
  }
}


SOCKET AngelDataHub::connectSocket(int port)
{
  auto log = this->get_logger();
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
    RCLCPP_WARN( log,
                 "Error creating socket for port : %d",
                 port);
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
    RCLCPP_WARN( log,
                 "Error creating socket for port : %d",
                 port);
  }
#endif
  else
  {
    RCLCPP_INFO(log, "Socket connected!");
  }

  return s;
}

} // namespace angel_datahub

RCLCPP_COMPONENTS_REGISTER_NODE( angel_datahub::AngelDataHub )
