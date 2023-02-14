#ifndef BBN_INTEGRATION_ROS_TO_YAML_H
#define BBN_INTEGRATION_ROS_TO_YAML_H

#include <bbn_integration_msgs/msg/bbn_update.hpp>
#include <yaml-cpp/yaml.h>

namespace angel {

namespace bbn_integration {

/**
 * Convert an input BBNUpdate message to a YAML formatted string that abides
 * the reference BBN format.
 *
 * @param msg BBNUpdate message to convert into a YAML string.
 * @return New root node to the YAML structure created.
 *
 * @throws std::invalid_argument The input message was not formatted
 *   appropriately to construct the target YAML format.
 */
YAML::Node
bbn_update_to_yaml( bbn_integration_msgs::msg::BBNUpdate::SharedPtr const msg );

} // namespace bbn_integration

} // namespace angel

#endif //BBN_INTEGRATION_ROS_TO_YAML_H
