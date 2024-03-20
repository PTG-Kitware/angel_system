#include <chrono>
#include <regex>
#include <sstream>

#include <bbn_integration/ros_to_yaml.h>
#include <bbn_integration_msgs/msg/bbn_update.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <yaml-cpp/yaml.h>
#include <zmq.hpp>

using bbn_integration_msgs::msg::BBNUpdate;
using std::placeholders::_1;

namespace angel {

namespace bbn_integration {

namespace {

/// Constant TCP URL pattern expected
static std::string const&
TCP_URL_REGEX_STR()
{
  static std::string _TCP_URL_REGEX_STR{ "tcp://[^:]+:\\d+" };
  return _TCP_URL_REGEX_STR;
}

/// Constant regex object for matching the TCP URL pattern above.
static std::regex const&
TCP_URL_REGEX()
{
  static std::regex _TCP_URL_REGEX{ TCP_URL_REGEX_STR() };
  return _TCP_URL_REGEX;
}

} // namespace

/**
 * Node to communicate out YAML messages to a receiving ZMQ server operated by
 * BBN.
 */
class ZmqIntegrationClient
  : public rclcpp::Node
{
private:
  std::string m_server_address;
  // ZMQ context with a single IO thread
  zmq::context_t m_zmq_context{ 1 };
  // ZMQ socket for communication
  zmq::socket_t m_zmq_socket{ m_zmq_context, zmq::socket_type::req };

  rclcpp::Subscription< BBNUpdate >::SharedPtr m_sub_updates;

public:
  /// Constructor
  ZmqIntegrationClient( rclcpp::NodeOptions const& options )
    : rclcpp::Node( "ZmqIntegrationClient", options )
  {
    this->declare_parameter( "topic_update_msg" );
    this->declare_parameter( "server_address" );
    // ROS2 Iron version
    //this->declare_parameter< std::string >( "topic_update_msg" );
    //this->declare_parameter< std::string >( "server_address" );

    // Separating parameter declaration and getting so the error messages are
    // more (read: at all) informative.
    auto topic_update_msg = this->get_parameter( "topic_update_msg" ).as_string();
    m_server_address = this->get_parameter( "server_address" ).as_string();

    // Initialize the socket with the server address
    RCLCPP_INFO_STREAM(
      get_logger(),
      "Initializing ZMQ socket for the server address: " << m_server_address );
    if( !std::regex_match( m_server_address, TCP_URL_REGEX() ) )
    {
      throw std::invalid_argument( "Input server address was not a valid TCP url. "
                                   "It should match the form: " +
                                   TCP_URL_REGEX_STR() );
    }
    m_zmq_socket.connect( m_server_address );

    RCLCPP_INFO( this->get_logger(), "Creating subscriber" );
    this->m_sub_updates = this->create_subscription< BBNUpdate >(
      topic_update_msg, 1,
      std::bind( &ZmqIntegrationClient::handle_bbn_update, this, _1 )
      );

    RCLCPP_INFO( this->get_logger(), "Node initialization complete!" );
  }

  /// Destructor
  ~ZmqIntegrationClient() override = default;

  /**
   * Handle a received update message, forwarding it over ZMQ to the configured
   * server address.
   *
   * @param bbn_update_msg Update message to translate and forward.
   */
  void
  handle_bbn_update( BBNUpdate::SharedPtr const bbn_update_msg )
  {
    auto const& log = this->get_logger();
    RCLCPP_INFO( log, "Received ROS update message, translating "
                      "into YAML and sending to ZMQ socket." );

    YAML::Node yaml_root;
    try
    {
      yaml_root = bbn_update_to_yaml( bbn_update_msg );
    }
    catch ( std::invalid_argument const& ex )
    {
      std::stringstream ss;
      ss        << "Failed to translate input BBNUpdate message into yaml: "
                << ex.what();
      RCLCPP_ERROR_STREAM( log, ss.str() );
      return;
    }

    // Override the transmission timestamp to "now"
    auto const now = std::chrono::system_clock::now();
    auto epoch_seconds =
      std::chrono::duration< double >( now.time_since_epoch() ).count();
    yaml_root[ "header" ][ "transmit timestamp" ] = epoch_seconds;

    auto yaml_translation = YAML::Dump( yaml_root );

    // Should return the number of bytes sent. If return does not have a value,
    // then the zmq_send function returned code EAGAIN (11) "Try again".
    RCLCPP_INFO( log, "Sending ROS update message as YAML..." );
    auto send_ret = m_zmq_socket.send(
                      zmq::buffer( yaml_translation ),
                      zmq::send_flags::none
                      );
    if( !send_ret.has_value() )
    {
      RCLCPP_WARN( log, "Failed to send YAML message via ZMQ socket." );
      return;
    }
    RCLCPP_INFO( log, "Sending ROS update message as YAML... Done" );

    // Recv should return the number of bytes received. Like before, if we
    // don't
    // receive a value, the zmq_recv function returned EAGAIN (11) "Try again".
    zmq::message_t reply_msg;
    auto recv_ret = m_zmq_socket.recv( reply_msg, zmq::recv_flags::none );
    if( !recv_ret.has_value() )
    {
      RCLCPP_WARN( log, "Failed to receive server reply via ZMQ socket." );
      return;
    }
    RCLCPP_INFO_STREAM( log, "Received reply from server: \"" << reply_msg.to_string() << "\"" );
  }
};

} // namespace bbn_integration

} // namespace angel

// Register this node as a ROS2 component node.
RCLCPP_COMPONENTS_REGISTER_NODE( angel::bbn_integration::ZmqIntegrationClient )
