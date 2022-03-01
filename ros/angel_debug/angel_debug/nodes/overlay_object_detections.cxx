#include <chrono>
#include <memory>
#include <numeric>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/u_int8_multi_array.hpp>

using std::placeholders::_1;
using std::cout;
using std::endl;

namespace angel_debug {

namespace {

// ----------------------------------------------------------------------------
static constexpr char const* PARAM_INPUT_IMAGE_TOPIC = "input_image_topic";
static constexpr char const* PARAM_INPUT_DET_TOPIC = "input_detections_topic";
static constexpr char const* PARAM_OUTPUT_IMAGE_TOPIC = "output_image_topic";

} // namespace

// ----------------------------------------------------------------------------
/// Overlay input object detections onto input imagery, outputting such
/// composite images.
///
/// Overlaying of detections occurs upon receiving an image. This node is
/// likely to be configured to take in the same image input as the detector
class OverlayImageObjectDetections
  : public rclcpp::Node
{
public:
  OverlayImageObjectDetections( rclcpp::NodeOptions const& options )
    : Node( "OverlayImageObjectDetections", options )
  {
    this->declare_parameter( PARAM_INPUT_IMAGE_TOPIC );
//    this->declare_parameter( PARAM_INPUT_DET_TOPIC );
    this->declare_parameter( PARAM_OUTPUT_IMAGE_TOPIC );

    auto input_image_topic =
      this->get_parameter( PARAM_INPUT_IMAGE_TOPIC ).get_value< std::string >();
//    auto input_detections_topic =
//      this->get_parameter( PARAM_INPUT_DET_TOPIC ).get_value< std::string
// >();
    auto output_image_topic =
      this->get_parameter( PARAM_OUTPUT_IMAGE_TOPIC ).get_value< std::string >();

    std::cout << "Creating subscribers and publishers" << std::endl;
    m_sub_input_image = this->create_subscription< sensor_msgs::msg::Image >(
      input_image_topic, 10, // rclcpp::SensorDataQoS().keep_last( 1 ),
      std::bind( &OverlayImageObjectDetections::collect_images, this, _1 )
      );
    // TODO: Collect detections
    m_pub_overlay_image = this->create_publisher< sensor_msgs::msg::Image >(
      output_image_topic, 10 // rclcpp::SensorDataQoS().keep_last( 1 )
      );
  }

  ~OverlayImageObjectDetections() override = default;

  /// Receive an image, overlaying current detections, and emitting an overlaid
  /// version to our output topic.
  void collect_images( sensor_msgs::msg::Image::SharedPtr const image );

  /// Receive and record emitted detections. Assumedly correlated to the
  /// imagery we are receiving.
//  void collect_detections( std_msgs)

private:
  typedef std::chrono::high_resolution_clock clock_t;

  rclcpp::Subscription< sensor_msgs::msg::Image >::SharedPtr m_sub_input_image;
  rclcpp::Publisher< sensor_msgs::msg::Image >::SharedPtr m_pub_overlay_image;

  size_t m_image_count = 0;

  // Measure and report receive/publish FPS
  // TODO: Parameterize?
  size_t m_fps_window_size = 10;
  bool m_first_report = true;
  std::chrono::time_point< clock_t > m_last_pub_time;
  std::vector< double > m_frame_time_vec;
  size_t m_frame_time_vec_i = 0;
};

// ----------------------------------------------------------------------------
void
OverlayImageObjectDetections
::collect_images( sensor_msgs::msg::Image::SharedPtr const image_msg )
{
  std::cout << "Received image #" << m_image_count << std::endl;

  auto cv_image_sptr = cv_bridge::toCvCopy( image_msg );

  // TODO: Overlay stuff

  auto out_img_msg_sptr = cv_image_sptr->toImageMsg();
  this->m_pub_overlay_image->publish( *out_img_msg_sptr );

  // Manage + Report FPS measurement over a window.
  double avg_fps( -1 );
  auto pub_time = clock_t::now();
  double time_since_last_pub =
    std::chrono::duration< double >( pub_time - m_last_pub_time ).count();
  if( m_first_report )
  {
    m_first_report = false;
  }
  else
  {
    // insert time measurement appropriately into window appropriately.
    if( m_frame_time_vec.size() < m_fps_window_size )
    {
      // add measurement
      m_frame_time_vec.push_back( time_since_last_pub );
    }
    else
    {
      // m_frame_time_vec is full, so now we start rotating new-measurement
      // insertion
      m_frame_time_vec[ m_frame_time_vec_i ] = time_since_last_pub;
      m_frame_time_vec_i = ( m_frame_time_vec_i + 1 ) %
                           m_frame_time_vec.size();
    }

    // average measurements.
    auto avg_time = std::accumulate( m_frame_time_vec.begin(),
                                     m_frame_time_vec.end(),
                                     0.0 ) /
                    static_cast< double >( m_frame_time_vec.size() );
    avg_fps = 1.0 / avg_time;
  }
  std::cout     << "Published debug image #" << m_image_count
                << " (fps: " << avg_fps << ")" << std::endl;
  m_last_pub_time = pub_time;
  ++m_image_count;
}

} // namespace angel_debug

RCLCPP_COMPONENTS_REGISTER_NODE( angel_debug::OverlayImageObjectDetections )
