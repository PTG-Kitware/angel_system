#include "../include/angel_utils/rate_tracker.hpp"

#include <chrono>
#include <numeric>
#include <vector>

namespace angel_utils {

namespace {

/// Type of clock to use for tick rate tracking.
using rt_clock_t = std::chrono::high_resolution_clock;

} // namespace

// ----------------------------------------------------------------------------
struct RateTracker::PImpl
{
  //
  // Properties
  //

  /// The measurement window size for rate smoothing.
  size_t m_window_size = 10;

  /// If we have not received a tick yet.
  bool m_first_tick = true;

  /// The last a tick occurred.
  std::chrono::time_point< rt_clock_t > m_last_measure_time;

  /// Circular buffer of time measurements between ticks.
  std::vector< double > m_time_vec;

  /// When the m_time_vec circular buffer is full, this is the next index to
  /// write to.
  size_t m_time_vec_i = 0;

  //
  // Methods
  //

  /// If we have time measurements to act on or not.
  bool
  has_measurements() const
  {
    return not m_time_vec.empty();
  }
};

// ----------------------------------------------------------------------------
RateTracker
::RateTracker( size_t window_size )
  : impl( std::make_unique< PImpl >() )
{
  impl->m_window_size = window_size;
}

// ----------------------------------------------------------------------------
RateTracker::~RateTracker() = default;

// ----------------------------------------------------------------------------
void
RateTracker
::tick()
{
  auto pub_time = rt_clock_t::now();
  if( impl->m_first_tick )
  {
    impl->m_first_tick = false;
  }
  else
  {
    double time_since_last_tick = std::chrono::duration< double >(
      pub_time - impl->m_last_measure_time
      ).count();
    // Insert time measurement appropriately into window.
    if( impl->m_time_vec.size() < impl->m_window_size )
    {
      // add measurement
      impl->m_time_vec.push_back( time_since_last_tick );
    }
    else
    {
      // m_frame_time_vec is full, so now we start rotating new-measurement
      // insertion
      impl->m_time_vec[ impl->m_time_vec_i ] = time_since_last_tick;
      impl->m_time_vec_i = ( impl->m_time_vec_i + 1 ) %
                           impl->m_time_vec.size();
    }
  }
  impl->m_last_measure_time = pub_time;
}

// ----------------------------------------------------------------------------
double
RateTracker
::get_delta_avg()
{
  double avg_time = -1;
  if( impl->has_measurements() )
  {
    double window_total = std::accumulate( impl->m_time_vec.begin(),
                                           impl->m_time_vec.end(), 0.0 );
    avg_time = window_total / static_cast< double >( impl->m_time_vec.size() );
  }
  return avg_time;
}

// ----------------------------------------------------------------------------
double
RateTracker
::get_rate_avg()
{
  double avg_rate = -1;
  if( impl->has_measurements() )
  {
    auto avg_time = get_delta_avg();
    avg_rate = 1.0 / avg_time;
  }
  return avg_rate;
}

} // namespace angel_utils
