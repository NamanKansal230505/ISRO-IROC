#!/usr/bin/env python3
"""
ISRO IROC Qualification - Task 3: ArUco Precision Landing
==========================================================
The UAV shall perform a safe and controlled landing within the take-off
zone using ArUco marker detection for precision guidance.

Uses a downward-facing camera on the Jetson Nano to detect an ArUco
marker on the landing pad, then sends LANDING_TARGET MAVLink messages
to the Pixhawk for precision landing (PrecLand).

Hardware:
    - Pixhawk FC with ArduCopter (PrecLand enabled)
    - Jetson Nano with downward-facing camera (CSI or USB)
    - ArUco marker (4x4_50 dictionary, ID 0) on the landing pad

Pixhawk Parameters (set beforehand):
    PLND_ENABLED  = 1
    PLND_TYPE     = 1  (MAVLink)
    PLND_EST_TYPE = 0  (RawSensor)

Usage:
    # Full sequence (takeoff + hover briefly + precision land):
    python3 task3_aruco_landing.py --alt 5

    # Land from current hover (drone already airborne):
    python3 task3_aruco_landing.py --alt 5 --skip-takeoff

    # Use USB camera instead of CSI:
    python3 task3_aruco_landing.py --camera usb --alt 5
"""

import argparse
import math
import time
import sys
import numpy as np
import cv2
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil


# ── Configuration ──
DEFAULT_CONNECTION = "/dev/ttyTHS1"
DEFAULT_BAUD = 921600
DEFAULT_TAKEOFF_ALT = 5.0

# ArUco marker settings
ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_50
TARGET_MARKER_ID = 0
MARKER_SIZE_CM = 40.0          # physical marker size in cm

# Camera settings (calibrate for your specific camera!)
# These are approximate values for a typical wide-angle camera
# Run camera calibration to get precise values for your setup
CAMERA_MATRIX = np.array([
    [530.0,   0.0, 320.0],
    [  0.0, 530.0, 240.0],
    [  0.0,   0.0,   1.0]
], dtype=np.float32)

DIST_COEFFS = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

# Image resolution
IMG_WIDTH = 640
IMG_HEIGHT = 480

# Landing control
DESCENT_SPEED = 0.3            # m/s — gentle descent rate
LAND_ALT_THRESHOLD = 0.8      # meters — switch to LAND mode below this
MARKER_LOST_TIMEOUT = 3.0     # seconds without marker before aborting to LAND
SEND_RATE_HZ = 10             # landing target message rate


def log(msg):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")
    sys.stdout.flush()


# ──────────────────────────────────────────────────────────
#  Camera Setup
# ──────────────────────────────────────────────────────────

def open_camera_csi(width=IMG_WIDTH, height=IMG_HEIGHT, fps=30):
    """Open Jetson Nano CSI camera via GStreamer pipeline."""
    gst_pipeline = (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width=1280, height=720, "
        f"format=NV12, framerate={fps}/1 ! "
        f"nvvidconv flip-method=0 ! "
        f"video/x-raw, width={width}, height={height}, format=BGR ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink"
    )
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("Failed to open CSI camera")
    return cap


def open_camera_usb(device=0, width=IMG_WIDTH, height=IMG_HEIGHT):
    """Open USB camera."""
    cap = cv2.VideoCapture(device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open USB camera (device {device})")
    return cap


# ──────────────────────────────────────────────────────────
#  ArUco Detection
# ──────────────────────────────────────────────────────────

class ArUcoDetector:
    def __init__(self, marker_id=TARGET_MARKER_ID, marker_size_cm=MARKER_SIZE_CM):
        self.target_id = marker_id
        self.marker_size_m = marker_size_cm / 100.0
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

    def detect(self, frame):
        """
        Detect ArUco marker and return angular offsets and distance.

        Returns:
            (found, angle_x, angle_y, distance_m)
            - angle_x: angle offset from center in radians (positive = right)
            - angle_y: angle offset from center in radians (positive = down)
            - distance_m: estimated distance to marker in meters
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is None:
            return False, 0, 0, 0

        # Find our target marker
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id == self.target_id:
                # Get marker center in pixels
                marker_corners = corners[i][0]
                cx = np.mean(marker_corners[:, 0])
                cy = np.mean(marker_corners[:, 1])

                # Estimate pose using solvePnP
                obj_points = np.array([
                    [-self.marker_size_m / 2,  self.marker_size_m / 2, 0],
                    [ self.marker_size_m / 2,  self.marker_size_m / 2, 0],
                    [ self.marker_size_m / 2, -self.marker_size_m / 2, 0],
                    [-self.marker_size_m / 2, -self.marker_size_m / 2, 0],
                ], dtype=np.float32)

                success, rvec, tvec = cv2.solvePnP(
                    obj_points, marker_corners, CAMERA_MATRIX, DIST_COEFFS
                )

                if success:
                    # tvec gives [x, y, z] in camera frame
                    # Camera points down, so z = distance to marker
                    tx, ty, tz = tvec.flatten()
                    distance = math.sqrt(tx**2 + ty**2 + tz**2)

                    # Angular offsets from camera center
                    angle_x = math.atan2(tx, tz)
                    angle_y = math.atan2(ty, tz)

                    return True, angle_x, angle_y, distance

                # Fallback: estimate from pixel position
                angle_x = math.atan2(cx - IMG_WIDTH / 2,
                                     CAMERA_MATRIX[0, 0])
                angle_y = math.atan2(cy - IMG_HEIGHT / 2,
                                     CAMERA_MATRIX[1, 1])
                # Rough distance from marker apparent size
                marker_px_size = np.linalg.norm(
                    marker_corners[0] - marker_corners[2])
                if marker_px_size > 0:
                    distance = (self.marker_size_m * CAMERA_MATRIX[0, 0]) / marker_px_size
                else:
                    distance = 5.0

                return True, angle_x, angle_y, distance

        return False, 0, 0, 0


# ──────────────────────────────────────────────────────────
#  MAVLink Landing Target
# ──────────────────────────────────────────────────────────

def send_landing_target(vehicle, angle_x, angle_y, distance):
    """
    Send LANDING_TARGET MAVLink message to Pixhawk.
    This is what PrecLand uses to correct position during descent.
    """
    msg = vehicle.message_factory.landing_target_encode(
        0,                          # time_usec (not used)
        0,                          # target_num
        mavutil.mavlink.MAV_FRAME_BODY_FRD,  # frame
        angle_x,                    # angle_x (rad)
        angle_y,                    # angle_y (rad)
        distance,                   # distance (m)
        0,                          # size_x (not used)
        0                           # size_y (not used)
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()


def send_velocity_ned(vehicle, vn, ve, vd):
    """Send velocity command in NED frame."""
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111000111,  # type_mask: velocity only
        0, 0, 0,            # position (ignored)
        vn, ve, vd,         # velocity NED
        0, 0, 0,            # acceleration (ignored)
        0, 0                 # yaw, yaw_rate (ignored)
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()


# ──────────────────────────────────────────────────────────
#  Takeoff Helpers
# ──────────────────────────────────────────────────────────

def arm_and_takeoff(vehicle, target_alt):
    log("Waiting for vehicle to be armable...")
    while not vehicle.is_armable:
        time.sleep(1)

    vehicle.mode = VehicleMode("GUIDED")
    while vehicle.mode.name != "GUIDED":
        time.sleep(0.5)

    log("Arming...")
    vehicle.armed = True
    while not vehicle.armed:
        time.sleep(0.5)

    time.sleep(2)
    log(f"Taking off to {target_alt:.1f}m...")
    vehicle.simple_takeoff(target_alt)

    while True:
        alt = vehicle.location.global_relative_frame.alt or 0
        log(f"  Climbing: {alt:.1f}m")
        if alt >= target_alt * 0.95:
            log(f"  Reached {alt:.1f}m")
            return True
        time.sleep(1)


# ──────────────────────────────────────────────────────────
#  Precision Landing
# ──────────────────────────────────────────────────────────

def precision_land(vehicle, cap, detector, target_alt):
    """
    Execute ArUco-guided precision landing.

    Strategy:
    1. Stay in GUIDED mode for position control
    2. Detect ArUco marker with downward camera
    3. Send LANDING_TARGET messages to Pixhawk PrecLand
    4. Command gentle descent velocity
    5. When below LAND_ALT_THRESHOLD, switch to LAND mode for final touchdown
    """
    log(f"\n{'='*55}")
    log("  PRECISION LANDING — ArUco Guided")
    log(f"{'='*55}")

    marker_last_seen = time.time()
    last_send_time = 0
    frame_count = 0
    detect_count = 0
    landing_complete = False

    # Ensure GUIDED mode
    if vehicle.mode.name != "GUIDED":
        vehicle.mode = VehicleMode("GUIDED")
        time.sleep(1)

    log("Starting ArUco detection loop...")
    log("  Looking for marker ID {} ({}cm, 4x4_50 dict)".format(
        TARGET_MARKER_ID, int(MARKER_SIZE_CM)))

    while not landing_complete:
        # Read camera frame
        ret, frame = cap.read()
        if not ret:
            log("WARNING: Camera frame read failed")
            time.sleep(0.1)
            continue

        frame_count += 1
        now = time.time()

        # Detect ArUco marker
        found, angle_x, angle_y, distance = detector.detect(frame)

        alt = vehicle.location.global_relative_frame.alt or 0

        if found:
            marker_last_seen = now
            detect_count += 1

            # Send landing target at specified rate
            if now - last_send_time >= 1.0 / SEND_RATE_HZ:
                send_landing_target(vehicle, angle_x, angle_y, distance)
                last_send_time = now

            # Command gentle descent while maintaining position via PrecLand
            # Slower descent as we get closer to ground
            if alt > 3.0:
                descent_rate = DESCENT_SPEED
            elif alt > 1.5:
                descent_rate = DESCENT_SPEED * 0.6
            else:
                descent_rate = DESCENT_SPEED * 0.3

            send_velocity_ned(vehicle, 0, 0, descent_rate)

            # Log every ~1 second
            if frame_count % 10 == 0:
                offset_x_deg = math.degrees(angle_x)
                offset_y_deg = math.degrees(angle_y)
                log(f"  ALT={alt:.2f}m | Marker: X={offset_x_deg:+.1f}° "
                    f"Y={offset_y_deg:+.1f}° | Dist={distance:.2f}m | "
                    f"Descent={descent_rate:.2f}m/s")

        else:
            # Marker not visible — hold position, stop descent
            time_since_seen = now - marker_last_seen

            if frame_count % 30 == 0:
                log(f"  ALT={alt:.2f}m | Marker: LOST ({time_since_seen:.1f}s)")

            if time_since_seen > MARKER_LOST_TIMEOUT:
                if alt > 2.0:
                    # At higher altitude, try to hold and search
                    send_velocity_ned(vehicle, 0, 0, 0)
                    log("  Holding position — searching for marker...")
                else:
                    # Close to ground, just land
                    log("  Marker lost near ground — switching to LAND")
                    vehicle.mode = VehicleMode("LAND")
                    landing_complete = True
                    break

        # Check if low enough for final LAND
        if alt < LAND_ALT_THRESHOLD and found:
            log(f"  Below {LAND_ALT_THRESHOLD}m with marker lock — final LAND")
            vehicle.mode = VehicleMode("LAND")
            landing_complete = True

        # Check if already landed
        if vehicle.mode.name == "LAND":
            if alt < 0.15 and not vehicle.armed:
                log("  Touchdown detected — motors disarmed")
                landing_complete = True
                break

        time.sleep(0.02)  # ~50Hz loop

    # Wait for actual touchdown if in LAND mode
    log("\nWaiting for touchdown...")
    touchdown_timeout = time.time() + 30
    while vehicle.armed and time.time() < touchdown_timeout:
        alt = vehicle.location.global_relative_frame.alt or 0
        if frame_count % 20 == 0:
            log(f"  Landing... {alt:.2f}m")
        time.sleep(0.5)

    # ── Summary ──
    detect_rate = (detect_count / frame_count * 100) if frame_count > 0 else 0
    final_loc = vehicle.location.global_relative_frame

    log(f"\n{'='*55}")
    log("  PRECISION LANDING — RESULTS")
    log(f"{'='*55}")
    log(f"  Frames processed  : {frame_count}")
    log(f"  Marker detections : {detect_count} ({detect_rate:.1f}%)")
    log(f"  Final altitude    : {final_loc.alt:.2f}m")
    log(f"  Motors armed      : {vehicle.armed}")
    log(f"  Vehicle mode      : {vehicle.mode.name}")

    if not vehicle.armed:
        log(f"\n  RESULT: LANDED SUCCESSFULLY")
    else:
        log(f"\n  RESULT: REVIEW — vehicle may still be armed")

    log(f"{'='*55}")

    return not vehicle.armed


def main():
    parser = argparse.ArgumentParser(
        description="IROC Task 3: ArUco Precision Landing")
    parser.add_argument("--connect", default=DEFAULT_CONNECTION,
                        help=f"Vehicle connection string (default: {DEFAULT_CONNECTION})")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--alt", type=float, default=DEFAULT_TAKEOFF_ALT,
                        help=f"Takeoff/hover altitude in meters (default: {DEFAULT_TAKEOFF_ALT})")
    parser.add_argument("--skip-takeoff", action="store_true",
                        help="Skip arm/takeoff if drone is already airborne")
    parser.add_argument("--camera", choices=["csi", "usb"], default="csi",
                        help="Camera type (default: csi)")
    parser.add_argument("--camera-device", type=int, default=0,
                        help="USB camera device index (default: 0)")
    parser.add_argument("--marker-id", type=int, default=TARGET_MARKER_ID,
                        help=f"ArUco marker ID to track (default: {TARGET_MARKER_ID})")
    parser.add_argument("--marker-size", type=float, default=MARKER_SIZE_CM,
                        help=f"Marker size in cm (default: {MARKER_SIZE_CM})")
    args = parser.parse_args()

    log("=" * 55)
    log("  IROC QUALIFICATION - TASK 3: ArUco PRECISION LANDING")
    log("=" * 55)

    # ── Open camera ──
    log(f"Opening {args.camera.upper()} camera...")
    if args.camera == "csi":
        cap = open_camera_csi()
    else:
        cap = open_camera_usb(args.camera_device)

    # Verify camera
    ret, test_frame = cap.read()
    if not ret:
        log("ABORT: Cannot read from camera")
        cap.release()
        return
    log(f"  Camera OK — {test_frame.shape[1]}x{test_frame.shape[0]}")

    # ── ArUco detector ──
    detector = ArUcoDetector(
        marker_id=args.marker_id,
        marker_size_cm=args.marker_size
    )

    # Quick test: check if marker visible from ground
    found, ax, ay, dist = detector.detect(test_frame)
    if found:
        log(f"  ArUco marker #{args.marker_id} detected in test frame (dist={dist:.2f}m)")
    else:
        log(f"  ArUco marker #{args.marker_id} not visible in test frame (will search in flight)")

    # ── Connect to vehicle ──
    log(f"Connecting to vehicle on {args.connect}...")
    vehicle = connect(args.connect, baud=args.baud, wait_ready=True, heartbeat_timeout=60)
    log(f"  Connected — Firmware: {vehicle.version}")

    try:
        if not args.skip_takeoff:
            if not arm_and_takeoff(vehicle, args.alt):
                log("ABORT: Takeoff failed")
                return
            # Brief hover to stabilize before descent
            log("Stabilizing at hover altitude for 5s...")
            time.sleep(5)
        else:
            alt = vehicle.location.global_relative_frame.alt or 0
            log(f"  Skip-takeoff mode — current altitude: {alt:.1f}m")
            if vehicle.mode.name != "GUIDED":
                vehicle.mode = VehicleMode("GUIDED")
                time.sleep(1)

        # ── Execute precision landing ──
        success = precision_land(vehicle, cap, detector, args.alt)

        if success:
            log("\n  TASK 3 COMPLETE: Precision landing successful")
        else:
            log("\n  TASK 3: Landing completed (check alignment)")

    except KeyboardInterrupt:
        log("\nUser interrupt — switching to LAND mode")
        vehicle.mode = VehicleMode("LAND")
    except Exception as e:
        log(f"\nERROR: {e}")
        log("Switching to LAND mode for safety")
        vehicle.mode = VehicleMode("LAND")
    finally:
        cap.release()
        vehicle.close()
        log("Camera and vehicle connection closed.")


if __name__ == "__main__":
    main()
