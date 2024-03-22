#include <chrono>
#include <cmath>
#include <exception>
#include <iomanip>
#include <memory>
#include <numeric>

// ROS2 things
#include <cv_bridge/cv_bridge.h>
//#include <cv_bridge/cv_bridge.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <rcl_interfaces/msg/parameter_type.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <angel_msgs/srv/query_image_size.hpp>

// Other stuff
#include <opencv2/opencv.hpp>

using rcl_interfaces::msg::ParameterDescriptor;
using rcl_interfaces::msg::ParameterType;
using std::placeholders::_1;
using std::placeholders::_2;

namespace angel_datahub {

namespace {

// ----------------------------------------------------------------------------
#define DEFINE_PARAM_NAME( var_name, param_name_str ) \
  static constexpr char const* var_name = param_name_str

// Topic we expect to receive headset NV12 images from.
DEFINE_PARAM_NAME( PARAM_TOPIC_INPUT_IMAGES, "topic_input_images" );
// Topic we will output RGB8 images to.
DEFINE_PARAM_NAME( PARAM_TOPIC_OUTPUT_IMAGE, "topic_output_images" );
// Drop every other Nth frame (Or None if set to 1)
DEFINE_PARAM_NAME( PARAM_DROP_Nth_FRAME, "drop_nth_frame" );
// Convert NV12 to RGB
DEFINE_PARAM_NAME( PARAM_CONVERT_NV12_TO_RGB, "convert_nv12_to_rgb" ); // TODO: Figure out how to read encoding instead
#undef DEFINE_PARAM_NAME


} // namespace

// ----------------------------------------------------------------------------
/// Simple conversion to decimal seconds from a ROS time structure.
inline double time_to_decimal(rclcpp::Time t)
{
  return (double)t.seconds() + (((double)t.nanoseconds()) / 1e9);
}

// ----------------------------------------------------------------------------
/// Converts NV12 images published from the headset to RGB8 images suitable for
/// use by the other ROS nodes.
///
class ImageConverter
  : public rclcpp::Node
{
public:
  ImageConverter( rclcpp::NodeOptions const& options );
  ~ImageConverter() override = default;

  /// Receive an image, overlaying current detections, and emitting an overlaid
  /// version to our output topic.
  void convert_images( sensor_msgs::msg::Image::SharedPtr const image_msg );

  /// Populate the image size query with the current image dimensions.
  void query_image_size_cb(const std::shared_ptr<angel_msgs::srv::QueryImageSize::Request> request,
                           std::shared_ptr<angel_msgs::srv::QueryImageSize::Response> response);

private:
  rclcpp::Subscription< sensor_msgs::msg::Image >::SharedPtr m_sub_input_image;
  rclcpp::Publisher< sensor_msgs::msg::Image >::SharedPtr m_pub_output_image;
  rclcpp::Service< angel_msgs::srv::QueryImageSize>::SharedPtr m_image_size_service;

  std::string frame_id = "PVFrameRGB";

  int m_image_width = -1;
  int m_image_height = -1;

  int m_image_id = 0;
  int m_drop_nth_frame = 1;
  bool m_convert_nv12_to_rgb = true;
};

// ----------------------------------------------------------------------------
ImageConverter
::ImageConverter( rclcpp::NodeOptions const& options )
  : Node( "ImageConverter", options )
{
  auto log = this->get_logger();

  // This two-stage declare->get allows the lack of passing a parameter to
  // throw an error with the parameter name in the error so the user has a
  // clue what is going wrong.
  declare_parameter( PARAM_TOPIC_INPUT_IMAGES );
  declare_parameter( PARAM_TOPIC_OUTPUT_IMAGE );
  declare_parameter( PARAM_DROP_Nth_FRAME );
  declare_parameter( PARAM_CONVERT_NV12_TO_RGB );
  // ROS2 Iron version
  //declare_parameter<std::string>( PARAM_TOPIC_INPUT_IMAGES );
  //declare_parameter<std::string>( PARAM_TOPIC_OUTPUT_IMAGE );
  //declare_parameter<int>( PARAM_DROP_Nth_FRAME, 1 );
  //declare_parameter<bool>( PARAM_CONVERT_NV12_TO_RGB );

  auto topic_input_images =
    this->get_parameter( PARAM_TOPIC_INPUT_IMAGES ).as_string();
  auto topic_output_image =
    this->get_parameter( PARAM_TOPIC_OUTPUT_IMAGE ).as_string();
  auto topic_drop_nth_frame =
    this->get_parameter( PARAM_DROP_Nth_FRAME ).as_int();
  m_drop_nth_frame = topic_drop_nth_frame;
  auto topic_convert_nv12_to_rgb =
    this->get_parameter( PARAM_CONVERT_NV12_TO_RGB ).as_bool();
  m_convert_nv12_to_rgb = topic_convert_nv12_to_rgb;

  RCLCPP_INFO( log, "Creating subscribers and publishers" );
  // Alternative "best effort" QoS: rclcpp::SensorDataQoS().keep_last( 1 )
  m_sub_input_image = this->create_subscription< sensor_msgs::msg::Image >(
    topic_input_images, 1,
    std::bind( &ImageConverter::convert_images, this, _1 )
    );
  m_pub_output_image = this->create_publisher< sensor_msgs::msg::Image >(
    topic_output_image, 1
    );

  // Create the image size service
  RCLCPP_INFO( log, "Creating image size query service" );
  m_image_size_service = this->create_service< angel_msgs::srv::QueryImageSize >(
    "query_image_size",
    std::bind( &ImageConverter::query_image_size_cb, this, _1, _2 )
    );
}

// ----------------------------------------------------------------------------
void
ImageConverter
::convert_images( sensor_msgs::msg::Image::SharedPtr const image_msg )
{
  auto log = this->get_logger();

  rclcpp::Time nv12_image_time = image_msg->header.stamp;
  rclcpp::Time nv12_receive_time = this->now();

  // Store the current image dimensions
  m_image_width = image_msg->width;
  m_image_height = image_msg->height;
  m_image_id = m_image_id + 1;

  if(m_image_id % m_drop_nth_frame == 0){
    cv::Mat rgb_image;

    std::string encoding;
    sensor_msgs::msg::Image::SharedPtr image_message;
    if(m_convert_nv12_to_rgb){
      // Convert from NV12 image data to RGB8
      cv::Mat nv12_image = cv::Mat(image_msg->height * 3/2, image_msg->width,
                                  CV_8UC1, &image_msg->data[0]);
      cv::cvtColor(nv12_image, rgb_image, cv::COLOR_YUV2RGB_NV12);
      encoding = "rgb8";

      // Convert the cv::Mat to a ROS ImageMsg
      image_message = cv_bridge::CvImage(std_msgs::msg::Header(),
                                        encoding,
                                        rgb_image).toImageMsg();
    }
    else{
      cv::Mat nv12_image = cv::Mat(image_msg->height, image_msg->width,
                                  CV_8UC1, &image_msg->data[0]);
      rgb_image = nv12_image; // actually this is bgr, don't convert
      encoding = "bgr8";

      image_message = image_msg;
    }

    image_message->header.stamp = image_msg->header.stamp;
    image_message->header.frame_id = frame_id;

    // Publish the RGB image
    rclcpp::Time rgb_publish_time = this->now();
    m_pub_output_image->publish(*image_message);

    // Log latencies
    {
      double time_nv12_image = time_to_decimal(nv12_image_time),
            time_nv12_receive = time_to_decimal(nv12_receive_time),
            time_rgb_publish = time_to_decimal(rgb_publish_time);
      double delta_image_receive = time_nv12_receive - time_nv12_image,
            delta_receive_publish = time_rgb_publish - time_nv12_receive,
            delta_image_publish = time_rgb_publish - time_nv12_image;
      auto ss = std::stringstream();
      ss << std::endl << std::setprecision(16)
        << "Received frame with header time @ " << time_nv12_image << std::endl
        << "Callback triggered @ " << time_to_decimal(nv12_receive_time) << std::endl
        << "Sending converted image time @ " << time_to_decimal(rgb_publish_time) << std::endl
        << "Capture --> Receive Latency: " << delta_image_receive << std::endl
        << "Receive --> Publish Latency: " << delta_receive_publish << std::endl
        << "Capture --> Publish Latency: " << delta_image_publish << std::endl
        << "Image ID: " << m_image_id << std::endl;
      // Foxy version?
      //RCLCPP_INFO( log, ss.str() );
      RCLCPP_INFO_STREAM( log, ss.str() );
    }
  }
}

// ----------------------------------------------------------------------------
void
ImageConverter
::query_image_size_cb(const std::shared_ptr<angel_msgs::srv::QueryImageSize::Request> request,
                      std::shared_ptr<angel_msgs::srv::QueryImageSize::Response> response)
{
  (void) request; // silence unused variable warning
  response->image_width = m_image_width;
  response->image_height= m_image_height;
}

} // namespace angel_datahub

RCLCPP_COMPONENTS_REGISTER_NODE( angel_datahub::ImageConverter )
