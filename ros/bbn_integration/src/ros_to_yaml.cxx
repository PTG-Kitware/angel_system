#include <bbn_integration/ros_to_yaml.h>
#include <bbn_integration_msgs/msg/bbn_casualties.hpp>
#include <yaml-cpp/yaml.h>

using bbn_integration_msgs::msg::BBNSkillConfidenceList;
using bbn_integration_msgs::msg::BBNUpdate;
using YAML::Node;

namespace angel {

namespace bbn_integration {

namespace {

/**
 * Create substructure that maps a casualty ID to a list of skill names and
 * confidences from the input consisting of parallel associative vectors.
 *
 * ROS2 messages don't support maps, thus the parallel associative vector
 * input.
 */
Node
create_skill_conf_map( std::vector< uint32_t > const& casualty_ids,
                       std::vector< BBNSkillConfidenceList > const& skills_lists )
{
  Node mapping{ YAML::NodeType::Map };
  for( size_t i = 0; i < casualty_ids.size(); ++i )
  {
    auto casualty_id = casualty_ids.at( i );
    Node skill_conf_list;
    for( auto const& skill_conf_msg : skills_lists.at( i ).list )
    {
      Node skill_conf;
      skill_conf[ 0 ] = skill_conf_msg.label;
      skill_conf[ 1 ] = skill_conf_msg.confidence;
      skill_conf_list.push_back( skill_conf );
    }
    mapping[ casualty_id ] = skill_conf_list;
  }
  return mapping;
}

char const*
step_state_to_str( uint8_t step_state )
{
  switch( step_state )
  {
    case 0: return "done";
    case 1: return "implied";
    case 2: return "current";
    case 3: return "unobserved";
  }

  // Unknown value;
  std::stringstream ss;
  ss    << "Unknown step state constant value, cannot translate to a string: "
        << step_state;
  throw std::invalid_argument( ss.str() );
}

Node
msg_to_yaml( bbn_integration_msgs::msg::BBNHeader const& msg )
{
  Node root;
  root[ "sender" ] = msg.sender;
  root[ "sender software version" ] = msg.sender_software_version;
  root[ "header fmt version" ] = msg.HEADER_FMT_VERSION;
  root[ "transmit timestamp" ] = msg.transmit_timestamp;
  root[ "closest hololens dataframe timestamp" ] = msg.closest_hololens_dataframe_timestamp;
  return root;
}

Node
msg_to_yaml( bbn_integration_msgs::msg::BBNCasualties const& msg )
{
  Node n;
  n[ "populated" ] = msg.populated;
  n[ "count" ] = msg.count;
  n[ "confidence" ] = msg.confidence;
  return n;
}

Node
msg_to_yaml( bbn_integration_msgs::msg::BBNSkillsOpenPerCasualty const& msg )
{
  Node n;
  n[ "populated" ] = msg.populated;
  n[ "casualty" ] = create_skill_conf_map( msg.casualty_ids, msg.skill_confidences );
  return n;
}

Node
msg_to_yaml( bbn_integration_msgs::msg::BBNSkillsDonePerCasualty const& msg )
{
  Node n;
  n[ "populated" ] = msg.populated;
  n[ "casualty" ] = create_skill_conf_map( msg.casualty_ids, msg.skill_confidences );
  return n;
}

Node
msg_to_yaml( bbn_integration_msgs::msg::BBNCasualtyCurrentlyWorkingOn const& msg )
{
  Node n;
  n[ "casualty" ] = msg.casualty;
  n[ "confidence" ] = msg.confidence;
  return n;
}

Node
msg_to_yaml( bbn_integration_msgs::msg::BBNCurrentSkill const& msg )
{
  Node n;
  n[ "number" ] = msg.number;
  n[ "confidence" ] = msg.confidence;
  return n;
}

Node
msg_to_yaml( std::vector< bbn_integration_msgs::msg::BBNStepState > const& msg_vec )
{
  Node n{ YAML::NodeType::Sequence };
  for( auto const& msg : msg_vec )
  {
    Node s;
    s[ "number" ] = msg.number;
    s[ "name" ] = msg.name;
    s[ "state" ] = step_state_to_str( msg.state );
    s[ "confidence" ] = msg.confidence;
    n.push_back( s );
  }
  return n;
}

Node
msg_to_yaml( bbn_integration_msgs::msg::BBNCurrentUserActions const& msg )
{
  Node n;
  n[ "populated" ] = msg.populated;
  n[ "casualty currently working on" ] = msg_to_yaml( msg.casualty_currently_working_on );
  n[ "current skill" ] = msg_to_yaml( msg.current_skill );
  n[ "steps" ] = msg_to_yaml( msg.steps );
  return n;
}

Node
msg_to_yaml( bbn_integration_msgs::msg::BBNNextStepProgress const& msg )
{
  Node n;
  n[ "populated" ] = msg.populated;
  n[ "velocity" ] = msg.velocity;
  return n;
}

Node
msg_to_yaml( bbn_integration_msgs::msg::BBNCurrentErrors const& msg )
{
  Node n;
  n[ "populated" ] = msg.populated;
  n[ "errors" ] = msg.errors;
  return n;
}

Node
msg_to_yaml( bbn_integration_msgs::msg::BBNCurrentUserState const& msg )
{
  Node n;
  n[ "populated" ] = msg.populated;
  return n;
}

} // namespace

Node
bbn_update_to_yaml( BBNUpdate::SharedPtr const msg )
{
  // Consistency checks
  auto num_ids = msg->skills_open_per_casualty.casualty_ids.size(),
       num_lists = msg->skills_open_per_casualty.skill_confidences.size();
  if( num_ids != num_lists )
  {
    throw std::invalid_argument( "Incongruent vector sizes between "
                                 "skills_open_per_casualty::casualty_ids and "
                                 "skills_open_per_casualty::skill_confidences." );
  }
  num_ids = msg->skills_done_per_casualty.casualty_ids.size();
  num_lists = msg->skills_done_per_casualty.skill_confidences.size();
  if( num_ids != num_lists )
  {
    throw std::invalid_argument( "Incongruent vector sizes between "
                                 "skills_done_per_casualty::casualty_ids and "
                                 "skills_done_per_casualty::skill_confidences." );
  }

  Node root;
  root[ "header" ] = msg_to_yaml( msg->bbn_header );
  root[ "casualties" ] = msg_to_yaml( msg->casualties );
  root[ "skills open per casualty" ] = msg_to_yaml( msg->skills_open_per_casualty );
  root[ "belief skill is done per casualty" ] = msg_to_yaml( msg->skills_done_per_casualty );
  root[ "users current actions right now" ] = msg_to_yaml( msg->current_user_actions );
  root[ "next step progress velocity" ] = msg_to_yaml( msg->next_step_progress );
  root[ "current errors" ] = msg_to_yaml( msg->current_errors );
  root[ "current user state" ] = msg_to_yaml( msg->current_user_state );

  return root;
}

} // namespace bbn_integration

} // namespace angel
