#
# An image metadata message
#

# Header should represent when this message was published not the image.
std_msgs/Header header

# Timestamp of when the image was created.
builtin_interfaces/Time image_source_stamp

# pixel size of the image
uint32 height
uint32 width
