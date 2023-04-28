# 1. BASIC HEADER STUFF (required)

# name / id you pick
string sender

# version of your software you pick
string sender_software_version

# 1.0 until we (BBN) decide otherwise
string HEADER_FMT_VERSION = 1.0

# your clock at xmit time in fractional seconds since 1970
float64 transmit_timestamp

# clock for the most recent hololens frame as seconds since 1970
float64 closest_hololens_dataframe_timestamp
