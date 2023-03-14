#ifndef RATETRACKER_HPP
#define RATETRACKER_HPP

#include <memory>

namespace angel_utils {

/// \brief Keep track of the rate at which something is cycling.
///
/// The python sibling of this utility is located [here](ros/angel_utils/python/angel_utils/rate_tracker.py).
/// Changes to this API and implementation should be reflected there.
class RateTracker
{
public:
  /// Create a new rate-tracker
  /// \param window_size Number of tick delta's to retain to compute moving
  /// averages.
  explicit RateTracker( size_t window_size = 10 );
  /// Destructor
  virtual ~RateTracker();

  /// Perform a measurement of the time since the last tick.
  void tick();

  /// Get the average time delta between ticks within our window of
  /// measurements.
  ///
  /// If there have been no measurements taken yet (by calling `tick()`) then
  /// a -1 value is returned.
  ///
  /// \return Average time in seconds between tick measurements.
  double get_delta_avg();

  /// Get the average tick rate from the window of measurements.
  ///
  /// If there have been no measurements taken yet (by calling `tick()`) then
  /// a -1 value is returned.
  ///
  /// \return Average rate in Hz between tick measurements.
  double get_rate_avg();

private:
  struct PImpl;
  std::unique_ptr< PImpl > impl;
};

} // namespace angel_utils

#endif //RATETRACKER_HPP
