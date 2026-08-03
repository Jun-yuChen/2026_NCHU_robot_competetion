std_msgs/Header header

sensor_msgs/Image rgb
sensor_msgs/Image depth
sensor_msgs/CameraInfo camera_info
float64 depth_scale   # meters per raw depth unit: depth_meters = raw_depth_value * depth_scale