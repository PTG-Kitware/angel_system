#include <chrono>
#include <cmath>
#include <exception>
#include <memory>
#include <numeric>

// ROS2 things
#include <builtin_interfaces/msg/time.hpp>
#include <cv_bridge/cv_bridge.h>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <rcl_interfaces/msg/parameter_type.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/image.hpp>

// Other stuff
#include <opencv2/opencv.hpp>

using rcl_interfaces::msg::ParameterDescriptor;
using rcl_interfaces::msg::ParameterType;
using std::placeholders::_1;

namespace image_converter {

namespace {

// ----------------------------------------------------------------------------
#define DEFINE_PARAM_NAME( var_name, param_name_str ) \
  static constexpr char const* var_name = param_name_str

// Topic we expect to receive headset NV12 images from.
DEFINE_PARAM_NAME( PARAM_TOPIC_INPUT_IMAGES, "topic_input_images" );
// Topic we will outpit RGB8 images to.
DEFINE_PARAM_NAME( PARAM_TOPIC_OUTPUT_IMAGE, "topic_output_images" );

#undef DEFINE_PARAM_NAME


} // namespace

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

private:
  rclcpp::Subscription< sensor_msgs::msg::Image >::SharedPtr m_sub_input_image;
  rclcpp::Publisher< sensor_msgs::msg::Image >::SharedPtr m_pub_output_image;

  std::string frame_id = "PVFrameRGB";
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

  auto topic_input_images =
    this->get_parameter( PARAM_TOPIC_INPUT_IMAGES ).as_string();
  auto topic_output_image =
    this->get_parameter( PARAM_TOPIC_OUTPUT_IMAGE ).as_string();

  RCLCPP_INFO( log, "Creating subscribers and publishers" );
  // Alternative "best effort" QoS: rclcpp::SensorDataQoS().keep_last( 1 )
  m_sub_input_image = this->create_subscription< sensor_msgs::msg::Image >(
    topic_input_images, 1,
    std::bind( &ImageConverter::convert_images, this, _1 )
    );
  m_pub_output_image = this->create_publisher< sensor_msgs::msg::Image >(
    topic_output_image, 1
    );
}

// ----------------------------------------------------------------------------
void
ImageConverter
::convert_images( sensor_msgs::msg::Image::SharedPtr const image_msg )
{
  auto log = this->get_logger();

  // Convert from NV12 image data to RGB8
  cv::Mat nv12_image = cv::Mat(image_msg->height * 3/2, image_msg->width,
                               CV_8UC1, &image_msg->data[0]);
  cv::Mat rgb_image;
  cv::cvtColor(nv12_image, rgb_image, cv::COLOR_YUV2RGB_NV12);

  // Create the output Image message
  sensor_msgs::msg::Image image_message = sensor_msgs::msg::Image();
  image_message.header.stamp = image_msg->header.stamp;
  image_message.header.frame_id = frame_id;
  image_message.height = image_msg->height;
  image_message.width = image_msg->width;
  image_message.is_bigendian = false;

  image_message.encoding = "rgb8";
  image_message.step = image_msg->width * 3;
  std::vector<unsigned char> v(rgb_image.data, rgb_image.data +
                                               image_msg->height * image_msg->width * 3);
  image_message.data = v;

  // Publish the RGB image
  m_pub_output_image->publish(image_message);
}

} // namespace image_converter

RCLCPP_COMPONENTS_REGISTER_NODE( image_converter::ImageConverter)
