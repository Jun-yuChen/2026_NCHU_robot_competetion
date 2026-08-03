import numpy as np
from dataclasses import dataclass
import pyrealsense2 as rs

'''
Image coordinate system (for YOLO and OpenCV)
            x / u →
          0 ───────────────────→ width
          │
          │
    y / v │       (x, y)
      ↓   │          ●
          │
          │
          ↓
          height
'''

@dataclass
class CameraDataFrame:
    color_image: np.ndarray
    depth_image: np.ndarray
    intrinsics: dict
    depth_scale: float


def get_connected_serials() -> list[str]:
    """
    Enumerate serial numbers of all connected RealSense devices.
    """
    ctx = rs.context()
    return [d.get_info(rs.camera_info.serial_number) for d in ctx.query_devices()]


class RsCamera:
    def __init__(self, serial: str, width: int = 848, height: int = 480, fps: int = 30, name=None):
        """
        Initialize a single RealSense D405 camera identified by its serial number.

        Args:
            serial:  Device serial number string (e.g. "123622270732").
            width:   Stream width in pixels  (D405 supports 848, 1280, …).
            height:  Stream height in pixels (D405 supports 480, 720, …).
            fps:     Frame rate.  Color and depth must share the same fps on D405.
        """
        self.serial = serial

        if name is not None:
            self._tag = f"[camera: {name}]"
        else:
            self._tag = f"[camera: {serial}]"

        print(f"{self._tag} Initializing Intel RealSense D405...")

        self.pipeline = rs.pipeline()
        config = rs.config()

        # Bind this pipeline exclusively to one physical device.
        # Without this, the second pipeline.start() on a multi-camera system
        # would grab whichever device happens to be "first" — unreliable.
        config.enable_device(serial)

        # D405: color and depth must have identical resolution and fps.
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16,  fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        try:
            self.profile = self.pipeline.start(config)
        except Exception as e:
            print(f"{self._tag} Failed to start. Check USB 3.0 connection. Error: {e}")
            raise

        # Get camera intrinsics
        color_profile = (
            self.profile
                .get_stream(rs.stream.color)
                .as_video_stream_profile()
        )

        intr = color_profile.get_intrinsics()

        # rs.distortion model name -> ROS sensor_msgs/CameraInfo distortion_model string.
        # RealSense color sensors typically report "Brown Conrady" or "None"; ROS
        # expects the plumb_bob name for the Brown-Conrady case.
        _model_name = str(intr.model).split(".")[-1]
        distortion_model = "plumb_bob" if "brown" in _model_name.lower() else "none"

        self.intrinsics = {
            "fx": intr.fx,
            "fy": intr.fy,
            "ppx": intr.ppx,
            "ppy": intr.ppy,
            "width": intr.width,
            "height": intr.height,
            "model": distortion_model,
            "coeffs": list(intr.coeffs),  # [k1, k2, p1, p2, k3]
        }

        # Get depth scale (self.depth_scale will transform raw depth value to meters, depth_meter = depth * depth_scale)
        device = self.profile.get_device()
        depth_sensor = device.first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale() 

        # Align depth frames to the color frame coordinate space.
        # D405 has good physical alignment, but software alignment guarantees
        # pixel-exact correspondence between color and depth arrays.
        self._align = rs.align(rs.stream.color)

        # Warm up: discard early frames so auto-exposure / white-balance settle.
        print(f"{self._tag} Warming up...")
        for _ in range(30):
            # wait_for_frames takes a positional int (milliseconds), not a keyword.
            self.pipeline.wait_for_frames(5000)
        print(f"{self._tag} Ready.")


    def read(self) -> CameraDataFrame | None:
        """
        Capture one synchronized RGB-D image.

        Returns:
            CameraDataFrame if successful, otherwise None.
        """
        # wait_for_frames(timeout_ms) — positional int, default 5000 ms.
        try:
            frames = self.pipeline.wait_for_frames(5000)
        except RuntimeError:
            return None

        aligned = self._align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            print(f"{self._tag} WARNING: incomplete frameset, skipping.")
            return None

        # np.asanyarray returns a *view* into the SDK's internal buffer.
        # Copying ensures callers hold a stable, independent snapshot.
        color_image = np.asanyarray(color_frame.get_data()).copy()
        depth_image = np.asanyarray(depth_frame.get_data()).copy()

        return CameraDataFrame(
            color_image = color_image,
            depth_image = depth_image,
            intrinsics = self.intrinsics,
            depth_scale = self.depth_scale,
        )


    def stop(self):
        """
        Stop the camera to release the USB device.
        """
        if self.pipeline is not None:
            self.pipeline.stop()


#============================= Tools ==============================================================

def get_depth_at_pixel(u: float, v: float, depth_image: np.ndarray, depth_scale: float) -> float:
    """
    Calculate depth by averaging the valid depth values in the
    3x3 neighborhood centered at (u, v).

    Args:
        depth_image: Depth image of shape (H, W)
        u: x-coordinate (float)
        v: y-coordinate (float)

    Returns:
        Average depth value in meters.
    """
    h, w = depth_image.shape

    print(f"(h, w) = ({h}, {w}),  (u, v) = ({u}, {v})")

    # Convert float coordinates to nearest pixel
    u = int(round(u))  # x , column
    v = int(round(v))  # y , row

    # Clamp to image bounds
    u = max(0, min(u, w - 1))
    v = max(0, min(v, h - 1))

    # Extract 10x10 neighborhood
    u_min = max(0, u - 5)
    u_max = min(w, u + 6)

    v_min = max(0, v - 5)
    v_max = min(h, v + 6)

    depth_meter = None

    try:
        roi = depth_image[v_min:v_max, u_min:u_max]
        #print(f"[Depth image] {roi}")

        valid = roi[roi > 0]

        if len(valid) == 0:
            raise ValueError(f"No valid depth values near row={u}, col={v}")

        depth_meter = float(np.median(valid)) * depth_scale  # transform to meter
        
    except: 
        print(f"[Depth image] Depth frame fail")
        depth_meter = None

    finally:
        return depth_meter
    


def get_bbox_depth(
    bbox,
    depth_image: np.ndarray,
    depth_scale: float,
) -> float | None:
    """
    Calculate the average depth of all valid pixels inside a bounding box.

    Args:
        bbox: Bounding box in the format (u1, v1, u2, v2).
        depth_image: Depth image of shape (H, W).
        depth_scale: Scale factor for converting raw depth values to meters.

    Returns:
        Average depth value inside the bounding box in meters.
        Returns None if the bounding box is invalid or contains
        no valid depth values.
    """
    u1, v1, u2, v2 = bbox

    h, w = depth_image.shape

    # Convert float coordinates to nearest pixel
    u1 = int(round(u1))
    v1 = int(round(v1))
    u2 = int(round(u2))
    v2 = int(round(v2))

    # Clamp bounding box to image bounds
    u1 = max(0, min(u1, w))
    u2 = max(0, min(u2, w))

    v1 = max(0, min(v1, h))
    v2 = max(0, min(v2, h))

    # Check for invalid bounding box
    if u1 >= u2 or v1 >= v2:
        print(f"[Depth image] Invalid bounding box: {bbox}")
        return None

    try:
        # Extract bounding box ROI using your u, v convention
        roi = depth_image[v1:v2, u1:u2]
        #print(f"[Depth image]\n {roi}")

        # Keep only valid depth values
        valid = roi[roi > 0]

        if len(valid) == 0:
            print(f"[Depth image] No valid depth values inside bbox: {bbox}")
            return None

        # Convert raw depth values to meters
        depth_meter = float(np.median(valid)) * depth_scale

        return depth_meter

    except Exception as e:
        print(f"[Depth image] Depth frame fail: {e}")
        return None



def Transform_pixel_to_camera(
    u: float,
    v: float,
    depth_meter: float,
    intrinsics: dict,
) -> np.ndarray:
    """
    Convert a pixel coordinate to a 3D point in the camera coordinate system.

    Args:
        u: Pixel x-coordinate (column).
        v: Pixel y-coordinate (row).
        depth_meter: Depth value at (u, v) in meters.
        intrinsics: Camera intrinsics dictionary containing
                    "fx", "fy", "ppx", and "ppy".

    Returns:
        A NumPy array [x, y, z] in meters.
    """
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["ppx"]
    cy = intrinsics["ppy"]

    z = depth_meter
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return np.array([x, y, z], dtype=np.float32)
