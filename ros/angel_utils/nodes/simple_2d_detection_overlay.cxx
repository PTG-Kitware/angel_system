#include <chrono>
#include <cmath>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>

// ROS2 things
#include <builtin_interfaces/msg/time.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <rcl_interfaces/msg/parameter_type.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/image.hpp>

// Other stuff
#include <opencv2/opencv.hpp>

// Our stuff
#include <angel_msgs/msg/hand_joint_poses_update.hpp>
#include <angel_msgs/msg/object_detection2d_set.hpp>
#include <angel_utils/rate_tracker.hpp>

using angel_msgs::msg::ObjectDetection2dSet;
using angel_msgs::msg::HandJointPosesUpdate;
using angel_utils::RateTracker;
using rcl_interfaces::msg::ParameterDescriptor;
using rcl_interfaces::msg::ParameterType;
using std::placeholders::_1;

namespace angel_utils {

namespace {

// ----------------------------------------------------------------------------
#define DEFINE_PARAM_NAME( var_name, param_name_str ) \
        static constexpr char const* var_name = param_name_str

// Topic we expect to receive headset RGB images from.
DEFINE_PARAM_NAME( PARAM_TOPIC_INPUT_IMAGES, "topic_input_images" );
// Topic we expect to receive 2D detections from.
DEFINE_PARAM_NAME( PARAM_TOPIC_INPUT_DET_2D, "topic_input_det_2d" );
// Topic we expect to receive joints from.
DEFINE_PARAM_NAME( PARAM_TOPIC_INPUT_JOINTS, "topic_input_joints" );
// Topic we will output debug overlay images to. This needs to be a compressed
// image-transport topic.
DEFINE_PARAM_NAME( PARAM_TOPIC_OUTPUT_IMAGE, "topic_output_images" );
// Maximum temporal volume of images and other data to retain as "history"
// for use in overlaying on imagery.
DEFINE_PARAM_NAME( PARAM_MAX_IMAGE_HISTORY_SECONDS, "max_image_history_seconds" );
// The amount of time, in seconds, to wait before publishing an image with any
// analytics overlaid on it.
DEFINE_PARAM_NAME( PARAM_PUBLISH_LATENCY_SECONDS, "publish_latency_seconds" );
// Filter detections to the highest <filter_top_k> values
// Value of -1 means display all detections
DEFINE_PARAM_NAME( PARAM_FILTER_TOP_K, "filter_top_k" );

#undef DEFINE_PARAM_NAME

// Nano-scale
static constexpr size_t const TENe9 = 1000000000;
// Factor of image max dim to determine line thickness, round result.
static constexpr double const LINE_FACTOR = 0.0015;
// Max length of the bounding box label displayed before wrapping the text
static constexpr int const MAX_LINE_LEN = 15;
// Map indicating joint->joint connecting lines for drawing the pose on an
// image.
static std::map< std::string, std::vector< std::string > > JOINT_CONNECTION_LINES = {
  { "nose", { "mouth" } },
  { "mouth", { "throat" } },
  { "throat", { "chest", "left_upper_arm", "right_upper_arm" } },
  { "chest", { "back" } },
  { "left_upper_arm", { "left_lower_arm" } },
  { "left_lower_arm", { "left_wrist" } },
  { "left_wrist", { "left_hand" } },
  { "right_upper_arm", { "right_lower_arm" } },
  { "right_lower_arm", { "right_wrist" } },
  { "right_wrist", { "right_hand" } },
  { "back", { "left_upper_leg", "right_upper_leg" } },
  { "left_upper_leg", { "left_knee" } },
  { "left_knee", { "left_lower_leg" } },
  { "left_lower_leg", { "left_foot" } },
  { "right_upper_leg", { "right_knee" } },
  { "right_knee", { "right_lower_leg" } },
  { "right_lower_leg", { "right_foot" } } };

/**
 * @brief Convert a seconds value into nanoseconds.
 */
constexpr
size_t
sec2nano( double seconds )
{
  return static_cast< size_t >( seconds * TENe9 );
}

/// Convert a header instance into a single-value time component to be used as
/// and order-able key.
constexpr
size_t
time_key_from_header( builtin_interfaces::msg::Time const& header )
{
  return ( header.sec * TENe9 ) + header.nanosec;
}

/**
 * @brief Deduce a line thickness to use for drawing shapes on a given matrix.
 *
 * Larger image matrices will yield a larger line thickness, which is useful
 * when viewing such images scaled down.
 *
 * @param img_mat Image matrix to base thickness on.
 * @return Line thickness in pixels.
 */
int
thickness_for_drawing( cv::Mat const& img_mat )
{
  auto max_dim = std::max( img_mat.rows, img_mat.cols );
  return round( max_dim * LINE_FACTOR );
}

} // namespace

// ----------------------------------------------------------------------------

/**
 * @brief A collection of ROS message data associated to a single point in
 * time.
 */
struct TimeDataBlock
{
  using ptr = std::shared_ptr< TimeDataBlock >;

  sensor_msgs::msg::Image::SharedPtr image_msg;
  ObjectDetection2dSet::SharedPtr dets_msg;
  HandJointPosesUpdate::SharedPtr joints_msg;
};

// ----------------------------------------------------------------------------
/// Overlay input object detections onto the image that they were predicted
/// over.
///
/// Overlaying of detections occurs upon receiving a detection set.
///
/// Overlaying may fail if we don't have the image the parent image the
/// detections were predicted upon. If this occurrs, the detection set is
/// simply dropped.
///
/// Assumes that images and headset have the same capture times. If this
/// becomes NOT the case in the future, then a setup using float-comparison,
/// most-recent-fallback will need to be added.
class Simple2dDetectionOverlay
  : public rclcpp::Node
{
public:
  Simple2dDetectionOverlay( rclcpp::NodeOptions const& options );
  virtual ~Simple2dDetectionOverlay() override = default;

  /// Receive an image, overlaying current detections, and emitting an overlaid
  /// version to our output topic.
  void collect_images( sensor_msgs::msg::Image::SharedPtr const image_msg );

  /// Receive and record emitted detections.
  void collect_detections( ObjectDetection2dSet::SharedPtr const det_set );

  /// Receive and record emitted joints.
  void collect_joints( HandJointPosesUpdate::SharedPtr const joints_msg );

private:
  /**
   * @brief Create an overlay image from the data in the given block of data
   * and publish the result as an image message
   *
   * @param tdb The time-data block to use for creating the overlay.
   *
   * @throws std::invalid_argument If there is no image set in the time-data.
   *
   * Side effect of running this method is an image will be published if there
   * was a valid image message given in the `tdb` parameter. The other items
   * of the `tdb` may be null, merely indicating that those items will not be
   * overlaid on to the image.
   */
  void overlay_and_publish( TimeDataBlock const& tdb ) const;

  /**
   * @brief Draw a set of 2D object detections on top of the image provided.
   *
   * This method will update the image provided in-place.
   *
   * @param img_ptr Image to draw on.
   * @param det_set Set of 2D object detections to plot.
   */
  void overlay_object_detections( cv_bridge::CvImagePtr img_ptr,
                                  ObjectDetection2dSet::SharedPtr det_set ) const;

  /**
   * @brief Draw a set of Pose Joints on top of the image provided.
   *
   * This method will update the image provided in-place.
   *
   * @param img_ptr Image to draw on.
   * @param joints_msg Set of pose oints to plot.
   */
  void overlay_pose_joints( cv_bridge::CvImagePtr img_ptr,
                            HandJointPosesUpdate::SharedPtr joints_msg ) const;

  /// Maximum temporal volume of images and other data to retain as "history"
  /// for overlaying on imagery.
  size_t m_max_image_history_ns;
  /// The amount of time, in nanoseconds, to wait before publishing an image
  /// with any analytics overlaid on it.
  size_t m_publish_latency_ns;
  /// Number of detections to display
  int m_filter_top_k;

  /// Measure and report receive/publish FPS - RGB Images
  RateTracker m_img_rate_tracker;
  /// Measure and report receive/publish FPS - 2D Object Detections
  RateTracker m_det_rate_tracker;
  /// Measure and report receive/publish FPS - Pose joints
  RateTracker m_joints_rate_tracker;

  std::shared_ptr< image_transport::ImageTransport > m_img_transport;

  rclcpp::Subscription< sensor_msgs::msg::Image >::SharedPtr m_sub_input_image;
  rclcpp::Subscription< ObjectDetection2dSet >::SharedPtr m_sub_input_det_2d;
  rclcpp::Subscription< HandJointPosesUpdate >::SharedPtr m_sub_input_joints;
  std::shared_ptr< image_transport::Publisher > m_pub_overlay_image;

  /// Simple counter for image messages received.
  size_t m_image_count = 0;
  /// Simple counter for 2D detection messages received.
  size_t m_detset_count = 0;
  /// Simple counter for pose joint messages received.
  size_t m_jointset_count = 0;

  /// Type associating a nanoseconds integer key to the image message.
  using frame_map_t = std::map< size_t, TimeDataBlock >;

  /// Mapping of time to the data-block of content associated with that time.
  frame_map_t m_frame_map;
  /// Mutex protecting m_frame_map
  std::mutex m_frame_map_lock;
  /// Nanosecond timestamp of the most recently published frame.
  /// This timestamp will be one that is `m_publish_latency_ns` behind the most
  /// recently received frame.
  /// We should not accept data that is at/earlier than this time.
  /// This value should be updated when we create an overlay for publishing.
  /// This value should be protected by the `m_frame_map_lock` mutex.
  size_t m_latest_frame_published{ 0 };
};

// ----------------------------------------------------------------------------
Simple2dDetectionOverlay
::Simple2dDetectionOverlay( rclcpp::NodeOptions const& options )
  : Node( "Simple2dDetectionOverlay", options ),
    m_img_rate_tracker( 10 ),
    m_det_rate_tracker( 10 )
{
  auto log = this->get_logger();

  // This two-stage declare->get allows the lack of passing a parameter to
  // throw an error with the parameter name in the error so the user has a
  // clue what is going wrong.
  declare_parameter( PARAM_TOPIC_INPUT_IMAGES );
  declare_parameter( PARAM_TOPIC_INPUT_DET_2D );
  declare_parameter( PARAM_TOPIC_INPUT_JOINTS );
  declare_parameter( PARAM_TOPIC_OUTPUT_IMAGE );
  declare_parameter( PARAM_MAX_IMAGE_HISTORY_SECONDS, 5.0 );
  declare_parameter( PARAM_PUBLISH_LATENCY_SECONDS, 1.0 );
  declare_parameter( PARAM_FILTER_TOP_K, -1 );

  auto topic_input_images =
    this->get_parameter( PARAM_TOPIC_INPUT_IMAGES ).as_string();
  auto topic_input_detections_2d =
    this->get_parameter( PARAM_TOPIC_INPUT_DET_2D ).as_string();
  auto topic_input_joints =
    this->get_parameter( PARAM_TOPIC_INPUT_JOINTS ).as_string();
  auto topic_output_image =
    this->get_parameter( PARAM_TOPIC_OUTPUT_IMAGE ).as_string();

  double tmp_double{ 0 };
  tmp_double = get_parameter( PARAM_MAX_IMAGE_HISTORY_SECONDS ).get_value< double >();
  if( tmp_double <= 0 )
  {
    std::stringstream ss;
    ss  << "Invalid max image history size, must be greater than 0, given "
        << tmp_double;
    throw std::invalid_argument( ss.str() );
  }
  m_max_image_history_ns = sec2nano( tmp_double );
  RCLCPP_INFO( log, "Max image history: %f s, %zu ns", tmp_double, m_max_image_history_ns );

  tmp_double = get_parameter( PARAM_PUBLISH_LATENCY_SECONDS ).get_value< double >();
  if( tmp_double < 0 )
  {
    std::stringstream ss;
    ss  << "Invalid publish latency, must be greater than or equal to 0, given "
        << tmp_double;
    throw std::invalid_argument( ss.str() );
  }
  m_publish_latency_ns = sec2nano( tmp_double );
  RCLCPP_INFO( log, "Latency window: %f s, %zu ns", tmp_double, m_publish_latency_ns );

  m_filter_top_k = get_parameter( PARAM_FILTER_TOP_K ).get_value< int >();
  if( m_filter_top_k < 0 )
  {
    RCLCPP_INFO(
      log,
      "Top-K filter set to a negative number (%d), no filtering will be "
      "performed.",
      m_filter_top_k );
  }

  // Create the ImageTransport instance with the empty-deleter-shared-ptr of
  // this.
  // * Create a shared pointer that references this node, but with an
  //   overridden deleter. This is to allow the shared pointer to be used as a
  //   weak pointer to be given to the ImageTransport object, and avoid a
  //   double-free of this instance when the ImageTransport instance is
  //   deallocated.
  std::shared_ptr< Simple2dDetectionOverlay > node_handle = { this, []( auto* ){} };
  m_img_transport = std::make_shared< image_transport::ImageTransport >( node_handle );

  RCLCPP_INFO( log, "Creating subscribers and publishers -- Input image" );
  // Alternative "best effort" QoS: rclcpp::SensorDataQoS().keep_last( 1 )
  m_sub_input_image = this->create_subscription< sensor_msgs::msg::Image >(
    topic_input_images, 1,
    std::bind( &Simple2dDetectionOverlay::collect_images, this, _1 )
    );
  RCLCPP_INFO( log, "Creating subscribers and publishers -- Input detections" );
  m_sub_input_det_2d = this->create_subscription< ObjectDetection2dSet >(
    topic_input_detections_2d, 1,
    std::bind( &Simple2dDetectionOverlay::collect_detections, this, _1 )
    );
  RCLCPP_INFO( log, "Creating subscribers and publishers -- Input joints" );
  m_sub_input_joints = this->create_subscription< HandJointPosesUpdate >(
    topic_input_joints, 1,
    std::bind( &Simple2dDetectionOverlay::collect_joints, this, _1 )
    );
  RCLCPP_INFO( log, "Creating subscribers and publishers -- Output image" );
  // Do we have to keep `it` around for `m_pub_overlay_image` to continue
  // working?
  m_pub_overlay_image = std::make_shared< image_transport::Publisher >(
    m_img_transport->advertise( topic_output_image, 1 )
    );
  RCLCPP_INFO( log, "Creating subscribers and publishers -- Done" );
}

// ----------------------------------------------------------------------------
void
Simple2dDetectionOverlay
::collect_images( sensor_msgs::msg::Image::SharedPtr const image_msg )
{
  auto log = this->get_logger();

  size_t image_nanosec_key = time_key_from_header( image_msg->header.stamp );
  RCLCPP_DEBUG( log, "Received image with time key: %zu", image_nanosec_key );

  // Guard against concurrent map access.
  std::lock_guard< std::mutex > guard( this->m_frame_map_lock );

  // If we have some frames, only retain this image if it newer than the last
  // frame-overlay published.
  if( image_nanosec_key <= m_latest_frame_published )
  {
    RCLCPP_WARN(
      log,
      "Received frame has time-key (%zu) at/older than the last published "
      "overlay (%zu).",
      image_nanosec_key, m_latest_frame_published );
    return;
  }

  // If this image has the same key as something in the map, something weird
  // happened? Unsure of the pathology of this, so just warning for now.
  if( m_frame_map.count( image_nanosec_key ) )
  {
    RCLCPP_WARN(
      log,
      "Incoming frame has key (%zu) already in our history map. What? "
      "Skipping for now...",
      image_nanosec_key
      );
    return;
  }

  // Add a data block to the map with the timestamp key.
  // `operator[]` will default construct a TimeDataBlock if one did not exist
  // yet for the timestamp key.
  m_frame_map[ image_nanosec_key ].image_msg = image_msg;

  // Get the most recent entry that is older than `image_ts - publish_latency`
  // and attempt overlay & publish. If there is nothing old enough, then we
  // should do nothing. When we are done finding such a frame to publish, we
  // should remove anything older than our `publish latency`.
  // Given our pattern of insertion and removal:
  // * The "latest" item in the map will always be `image_nanosec_key`
  // * The "target" item will always be older the `image_nanosec_key`
  // * Scanning starting from the beginning of the map will be the most
  //   efficient place to start on average (should probably only see
  //   one or two things at most).
  size_t target_ts = image_nanosec_key - m_publish_latency_ns;
  RCLCPP_DEBUG( log, "Finding image before %zu...", target_ts );

  // Stop looping if we've hit the end of the map, or if the current item is
  // *just* older than the target timestamp relative to the next item.
  // Removing items older than the target timestamp as we go.
  bool found_data{ false };
  for( auto it = m_frame_map.cbegin();
       it != m_frame_map.cend() && it->first <= target_ts;
       /* explicitly not incrementing here */ )
  {
    // If this is the last (but old enough) entry, or if the next entry is now
    // within the latency window, this entry should be published.
    auto next_it = std::next( it );

    RCLCPP_DEBUG( log, "--> C-TS: %zu", it->first );
    if( next_it == m_frame_map.cend() )
    {
      RCLCPP_DEBUG( log, "--> N-TS: END" );
    }
    else
    {
      RCLCPP_DEBUG( log, "--> N-TS: %zu", next_it->first );
    }

    // This should always trigger if there is *something* in the map.
    if( next_it == m_frame_map.cend() || next_it->first > target_ts )
    {
      if( ( it->second ).image_msg != nullptr )
      {
        found_data = true;
        RCLCPP_DEBUG( log, "--> :) Found image: %zu", it->first );
        this->overlay_and_publish( it->second );
        m_latest_frame_published = it->first;
      }
      else
      {
        RCLCPP_WARN( log, "No image message to overlay and publish for "
                          "buffered data with timestamp %zu", it->first );
      }
    }
    // Remove this entry, which is older than the target_ts, from the frame
    // map.
    it = m_frame_map.erase( it );
  }
  if( !found_data )
  {
    RCLCPP_INFO( log, "--> :( No image found" );
  }
  RCLCPP_DEBUG( log, "m_frame_map size: %zu", m_frame_map.size() );

  // Because we like to know how fast this is going.
  m_img_rate_tracker.tick();
  RCLCPP_INFO( log,
               "Collected Image #%lu NS=%zu (hz: %f)",
               m_image_count,
               image_nanosec_key,
               m_img_rate_tracker.get_rate_avg() );
  ++m_image_count;
}

// ----------------------------------------------------------------------------
void
Simple2dDetectionOverlay
::collect_detections( ObjectDetection2dSet::SharedPtr const det_set )
{
  auto log = this->get_logger();

  // lookup image for det_set->source_stamp
  // check that detection header frame_id matches
  size_t source_nanosec_key =
    time_key_from_header( det_set->source_stamp );
  RCLCPP_DEBUG( log, "Detection source key: %zu", source_nanosec_key );

  // Guard against concurrent map access.
  std::lock_guard< std::mutex > guard( this->m_frame_map_lock );

  // If we have some frames, only retain these detections if it is newer
  // than the last frame-overlay published.
  if( source_nanosec_key <= m_latest_frame_published )
  {
    RCLCPP_WARN(
      log,
      "Received detections has time-key (%zu) at/older than the last "
      "published overlay (%zu).",
      source_nanosec_key, m_latest_frame_published );
    return;
  }

  m_frame_map[ source_nanosec_key ].dets_msg = det_set;

  // Because we like to know how fast this is going.
  m_det_rate_tracker.tick();
  RCLCPP_INFO( log, "Collected detection set #%lu NS=%zu (hz: %f)",
               m_detset_count,
               source_nanosec_key,
               m_det_rate_tracker.get_rate_avg() );
  ++m_detset_count;
}

// ----------------------------------------------------------------------------
void
Simple2dDetectionOverlay
::collect_joints( HandJointPosesUpdate::SharedPtr const joints_msg )
{
  auto log = this->get_logger();

  // Only plot the patient skeleton
  if( joints_msg->hand != "patient" )
  {
    RCLCPP_WARN(
      log,
      "Received joints that do not belong to the patient."
      "joint source: (%s)",
      joints_msg->hand
      );
    return;
  }

  // lookup image for joints_msg->source_stamp
  // check that detection header frame_id matches
  size_t source_nanosec_key =
    time_key_from_header( joints_msg->source_stamp );
  RCLCPP_DEBUG( log, "Joint source key: %zu", source_nanosec_key );

  // Guard against concurrent map access.
  std::lock_guard< std::mutex > guard( this->m_frame_map_lock );

  // If we have some frames, only retain these joints if it is newer than the
  // last frame-overlay published.
  if( source_nanosec_key <= m_latest_frame_published )
  {
    RCLCPP_WARN(
      log,
      "Received detections has time-key (%zu) at/older than the last "
      "published overlay (%zu).",
      source_nanosec_key, m_latest_frame_published );
    return;
  }

  m_frame_map[ source_nanosec_key ].joints_msg = joints_msg;

  // Because we like to know how fast this is going.
  m_joints_rate_tracker.tick();
  RCLCPP_INFO( log, "Collected joints set #%lu NS=%zu (hz: %f)",
               m_jointset_count,
               source_nanosec_key,
               m_joints_rate_tracker.get_rate_avg() );
  ++m_jointset_count;
}

// ----------------------------------------------------------------------------
void
Simple2dDetectionOverlay
::overlay_and_publish( angel_utils::TimeDataBlock const& tdb ) const
{
  // If there is no image to overlay on, then we're done.
  if( tdb.image_msg == nullptr )
  {
    throw std::invalid_argument( "No image provided to over lay on" );
  }

  // Convert the image message to a cv::Mat to be drawn over.
  cv_bridge::CvImagePtr img_ptr = cv_bridge::toCvCopy( tdb.image_msg, "rgb8" );

  // Draw pose
  if( tdb.joints_msg != nullptr )
  {
    overlay_pose_joints( img_ptr, tdb.joints_msg );
  }

  // Draw detections
  if( tdb.dets_msg != nullptr )
  {
    overlay_object_detections( img_ptr, tdb.dets_msg );
  }

  // Publish the final image`
  auto out_img_msg = img_ptr->toImageMsg();
  m_pub_overlay_image->publish( *out_img_msg );
}

void
Simple2dDetectionOverlay
::overlay_object_detections( cv_bridge::CvImagePtr img_ptr,
                             ObjectDetection2dSet::SharedPtr det_set ) const
{
  // color constants we're using here.
  static auto const COLOR_BOX = cv::Scalar{ 255, 0, 255 }; // magenta
  static auto const COLOR_TEXT = cv::Scalar{ 255, 255, 255 }; // white

  auto log = this->get_logger();

  // Choose a line thickness to use base on the size of the image.
  int line_thickness = thickness_for_drawing( img_ptr->image );
  RCLCPP_DEBUG( log, "Using line thickness (detections): %d", line_thickness );

  // Create the matrix of per-label confidence scores for the detection set
  size_t num_detections = det_set->num_detections;
  size_t num_labels = det_set->label_vec.size();
  auto det_label_conf_mat = cv::Mat{
    (int) num_detections, (int) num_labels, CV_64F,
    det_set->label_confidences.data() };

  // Temp variables for recording the maximally confident label and the
  // confidence value for each detection.
  cv::Point max_point;  // Only x will be populated due to single row scan.
  double max_conf;

  // Labels of the most confident class per detection.
  // Shape: [n_dets]
  std::vector< std::string > max_label_vec;
  // Confidence values of the most confident class per detection.
  // Shape: [n_dets]
  std::vector< double > max_conf_vec;
  for( size_t i = 0; i < num_detections; ++i )
  {
    // Determine the label for each detection based on max confidence item.
    // Only `max_point.x` is populated b/c operating on a single row.
    cv::minMaxLoc( det_label_conf_mat.row( i ), NULL, &max_conf, NULL, &max_point );
    // Record the label and confidence value for the most confident class.
    max_label_vec.push_back( det_set->label_vec[ max_point.x ] );
    max_conf_vec.push_back( max_conf );
  }

  // Number of detections to draw.
  size_t draw_n = max_conf_vec.size();
  // Indices of the detections to draw.
  std::vector< size_t > draw_indices( max_conf_vec.size() );
  std::iota( draw_indices.begin(), draw_indices.end(), 0 );

  // Find top k results
  // If filtering by top-k, sort the beginning `k` portion of `draw_indices`
  // based on detection confidence score in descending order, reducing the
  // `draw_n` value to `k`.
  if( m_filter_top_k > -1 )
  {
    RCLCPP_DEBUG( log, "Top k: %d", m_filter_top_k );

    // Arg-sort the top-k highest confidence detections.
    draw_n = std::min( max_conf_vec.size(), (size_t) m_filter_top_k );
    std::partial_sort( draw_indices.begin(), draw_indices.begin() + draw_n, draw_indices.end(),
                       [ & ]( size_t A, size_t B ){
                         return max_conf_vec[ A ] > max_conf_vec[ B ];
                       } );
  }

  // Draw detection bounding boxes and class labels for the most confident
  // class.
  size_t idx;
  for( size_t i = 0; i < draw_n; ++i )
  {
    idx = draw_indices.at( i );

    cv::Point pt_ul = { (int) round( det_set->left[ idx ] ),
                        (int) round( det_set->top[ idx ] ) },
              pt_br = { (int) round( det_set->right[ idx ] ),
                        (int) round( det_set->bottom[ idx ] ) };
    cv::rectangle( img_ptr->image, pt_ul, pt_br,
                   COLOR_BOX, line_thickness, cv::LINE_8 );

    std::string line = max_label_vec[ idx ];
    int line_len = line.length();
    for(int i = 0; i < line_len; i += MAX_LINE_LEN)
    {
      std::string split_line = line.substr(
                                 i, std::min( MAX_LINE_LEN, line_len - i ) );
      cv::Point line_loc = { pt_ul.x, pt_ul.y + i * ( line_thickness + 1 ) };
      cv::putText( img_ptr->image, split_line, line_loc,
                   cv::FONT_HERSHEY_COMPLEX, line_thickness, COLOR_TEXT,
                   line_thickness );
    }
  }
}

void
Simple2dDetectionOverlay
::overlay_pose_joints( cv_bridge::CvImagePtr img_ptr,
                       HandJointPosesUpdate::SharedPtr joints_msg ) const
{
  // color constants we're using here.
  static auto const COLOR_PT = cv::Scalar{ 0, 255, 0 };

  auto log = this->get_logger();

  int line_thickness = thickness_for_drawing( img_ptr->image );
  RCLCPP_DEBUG( log, "Using line thickness (joints): %d", line_thickness );

  // Draw the joint points
  // Save the joint positions for later so we can draw the connecting lines.
  std::map< std::string, std::vector< double > > joint_positions = {};
  for( auto const& joint : joints_msg->joints )
  {
    double x = joint.pose.position.x;
    double y = joint.pose.position.y;
    joint_positions[ joint.joint ] = { x, y }; // save for later

    // Plot the point
    cv::Point pt = { (int) round( x ),
                     (int) round( y ) };
    cv::circle( img_ptr->image, pt, line_thickness * 3,
                COLOR_PT, cv::FILLED );
  }

  // Draw the joint connections
  cv::Point pt1;
  cv::Point pt2;
  for(auto const& connection : JOINT_CONNECTION_LINES)
  {
    std::string joint_name = connection.first;
    std::vector< std::string > joint_connections = connection.second;
    std::vector< double > first_joint = joint_positions[ joint_name ];
    pt1 = {
      (int) round( first_joint[ 0 ] ),
      (int) round( first_joint[ 1 ] ) };

    for(auto const& connecting_joint : joint_connections)
    {
      std::vector< double > connecting_pt = joint_positions[ connecting_joint ];
      pt2 = {
        (int) round( connecting_pt[ 0 ] ),
        (int) round( connecting_pt[ 1 ] ) };

      cv::line( img_ptr->image, pt1, pt2, COLOR_PT, line_thickness, cv::LINE_8 );
    }
  }
}

} // namespace angel_utils

RCLCPP_COMPONENTS_REGISTER_NODE( angel_utils::Simple2dDetectionOverlay )
