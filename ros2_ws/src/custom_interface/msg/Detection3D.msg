string class_name
float32 confidence

# 3D pose of the object in the camera optical frame (meters).
# Orientation is identity (unknown) unless a downstream node estimates it.
geometry_msgs/Pose pose
