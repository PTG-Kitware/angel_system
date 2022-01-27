#include <memory>
#include <iostream>
#include <iterator>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/byte_multi_array.hpp"

using std::placeholders::_1;

class MinimalSubscriber : public rclcpp::Node
{
  public:
    MinimalSubscriber()
    : Node("minimal_subscriber")
    {
      subscriptionLF_ = this->create_subscription<std_msgs::msg::ByteMultiArray>(
      "LFFrames", 10, std::bind(&MinimalSubscriber::topic_callback_LF, this, _1));
      subscriptionRF_ = this->create_subscription<std_msgs::msg::ByteMultiArray>(
      "RFFrames", 10, std::bind(&MinimalSubscriber::topic_callback_RF, this, _1));
      subscriptionLL_ = this->create_subscription<std_msgs::msg::ByteMultiArray>(
      "LLFrames", 10, std::bind(&MinimalSubscriber::topic_callback_LL, this, _1));
      subscriptionRR_ = this->create_subscription<std_msgs::msg::ByteMultiArray>(
      "RRFrames", 10, std::bind(&MinimalSubscriber::topic_callback_RR, this, _1));
    }

  private:
    rclcpp::Subscription<std_msgs::msg::ByteMultiArray>::SharedPtr subscriptionLF_;
    rclcpp::Subscription<std_msgs::msg::ByteMultiArray>::SharedPtr subscriptionRF_;
    rclcpp::Subscription<std_msgs::msg::ByteMultiArray>::SharedPtr subscriptionLL_;
    rclcpp::Subscription<std_msgs::msg::ByteMultiArray>::SharedPtr subscriptionRR_;

    void topic_callback_LF(const std_msgs::msg::ByteMultiArray::SharedPtr msg) const
    {
      std::cout << "Got something LF: " << std::to_string(msg->data.size()) << std::endl;
    }

    void topic_callback_RF(const std_msgs::msg::ByteMultiArray::SharedPtr msg) const
    {
      std::cout << "Got something RF: " << std::to_string(sizeof(msg->data)) << std::endl;
    }

    void topic_callback_LL(const std_msgs::msg::ByteMultiArray::SharedPtr msg) const
    {
      std::cout << "Got something LL: " << std::to_string(sizeof(msg->data)) << std::endl;
    }

    void topic_callback_RR(const std_msgs::msg::ByteMultiArray::SharedPtr msg) const
    {
      std::cout << "Got something RR: " << std::to_string(sizeof(msg->data))  << std::endl;
    }

};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  std::cout << "Listening for messages!\n";

  rclcpp::spin(std::make_shared<MinimalSubscriber>());

  rclcpp::shutdown();
  return 0;
}