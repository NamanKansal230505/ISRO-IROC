#!/usr/bin/env python3
"""
ISRO IROC Qualification - Task 3: ArUco Precision Landing (GPS-Denied)
=======================================================================
The UAV shall perform a safe and controlled landing within the take-off
zone using ArUco marker detection for precision guidance.

Uses the Arducam OV9281 (120 FPS, global shutter, monochrome) downward-
facing camera on the Jetson Nano to detect an ArUco marker on the
landing pad. Sends LANDING_TARGET MAVLink messages to Pixhawk PrecLand.

Navigation: Intel RealSense D435i VIO (primary) + Matek MTF-01
            PMW3901 optical flow + VL53L1X LiDAR (backup)
Landing Camera: Arducam OV9281 (120 FPS monochrome, global shutter)

Pixhawk Parameters (set beforehand):
    PLND_ENABLED  = 1
    PLND_TYPE     = 1  (MAVLink)
    PLND_EST_TYPE = 0  (RawSensor)
    EK3_SRC1_POSXY = 6 (ExternalNav)
    EK3_SRC1_VELXY = 6 (ExternalNav)
    GPS_TYPE       = 0 (Disabled)

Usage:
    # Full sequence (takeoff + hover briefly + precision land):
    python3 task3_aruco_landing.py --alt 5

    # Land from current hover (drone already airborne):
    python3 task3_aruco_landing.py --alt 5 --skip-takeoff

    # Override camera device:
    python3 task3_aruco_landing.py --camera-device 1 --alt 5
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

# Arducam OV9281 settings
# Monochrome, global shutter, 120 FPS capable
# IMPORTANT: Replace with your actual calibration values!
# Run: python3 -c "import cv2; help(cv2.calibrateCamera)" for calibration
IMG_WIDTH = 640
IMG_HEIGHT = 480
CAMERA_FPS = 120

# Camera intrinsics (calibrate for your specific OV9281!)
# These are approximate — run proper checkerboard calibration
CAMERA_MATRIX = np.array([
    [420.0,   0.0, 320.0],
    [  0.0, 420.0, 240.0],
    [  0.0,   0.0,   1.0]
], dtype=np.float32)

DIST_COEFFS = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

# Landing control
DESCENT_SPEED = 0.3            # m/s — gentle descent rate
LAND_ALT_THRESHOLD = 0.8      # meters — switch to LAND mode below this
MARKER_LOST_TIMEOUT = 3.0     # seconds without marker before fallback
SEND_RATE_HZ = 30             # landing target msg rate (match camera FPS)
OPTICAL_FLOW_FALLBACK_ALT = 0.3  # below this, trust optical flow for final touch


def log(msg):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")
    sys.stdout.flush()


def get_rangefinder_alt(vehicle):
    """Get altitude from rangefinder (MTF-01 VL53L1X LiDAR)."""
    alt = vehicle.rangefinder.distance
    if alt is not None and alt > 0:
        return alt
    return vehicle.location.global_relative_frame.alt or 0


# ──────────────────────────────────────────────────────────
#  Camera Setup — Arducam OV9281
# ──────────────────────────────────────────────────────────

def open_arducam_v4l2(device=0):
    """
    Open Arducam OV9281 via V4L2.
    The OV9281 is monochrome — frames come as single-channel grayscale.
    """
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    # Use MJPG or raw format for high FPS
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open Arducam OV9281 (device {device})")
    return cap


def open_arducam_gstreamer(device=0):
    """
    Open Arducam OV9281 via GStreamer pipeline on Jetson Nano.
    Fallback if V4L2 doesn't work at desired FPS.
    """
    gst_pipeline = (
        f"v4l2src device=/dev/video{device} ! "
        f"video/x-raw, width={IMG_WIDTH}, height={IMG_HEIGHT}, "
        f"framerate={CAMERA_FPS}/1 ! "
        f"videoconvert ! video/x-raw, format=GRAY8 ! appsink"
    )
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("Failed to open Arducam via GStreamer")
    return cap


# ──────────────────────────────────────────────────────────
#  ArUco Detection (optimized for monochrome OV9281)
# ──────────────────────────────────────────────────────────

class ArUcoDetector:
    def __init__(self, marker_id=TARGET_MARKER_ID, marker_size_cm=MARKER_SIZE_CM):
        self.target_id = marker_id
        self.marker_size_m = marker_size_cm / 100.0
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
        self.aruco_params = cv2.aruco.DetectorParameters()
        # Tune for high-speed monochrome camera
        self.aruco_params.adaptiveThreshConstant = 7
        self.aruco_params.adaptiveThreshWinSizeMin = 3
        self.aruco_params.adaptiveThreshWinSizeMax = 23
        self.aruco_params.adaptiveThreshWinSizeStep = 10
        self.aruco_params.minMarkerPerimeterRate = 0.02
        self.aruco_params.maxMarkerPerimeterRate = 4.0
        self.aruco_params.polygonalApproxAccuracyRate = 0.03
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Object points for solvePnP
        half = self.marker_size_m / 2
        self.obj_points = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0],
        ], dtype=np.float32)

    def detect(self, frame):
        """
        Detect ArUco marker in frame (grayscale or BGR).

        Returns:
            (found, angle_x, angle_y, distance_m, cx, cy)
            - angle_x/y: angular offset from center in radians
            - distance_m: estimated distance to marker
            - cx, cy: marker center in pixels
        """
        # OV9281 is mono — frame may already be grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is None:
            return False, 0, 0, 0, 0, 0

        for i, marker_id in enumerate(ids.flatten()):
            if marker_id == self.target_id:
                marker_corners = corners[i][0]
                cx = np.mean(marker_corners[:, 0])
                cy = np.mean(marker_corners[:, 1])

                # Pose estimation via solvePnP
                success, rvec, tvec = cv2.solvePnP(
                    self.obj_points, marker_corners,
                    CAMERA_MATRIX, DIST_COEFFS
                )

                if success:
                    tx, ty, tz = tvec.flatten()
                    distance = math.sqrt(tx**2 + ty**2 + tz**2)
                    angle_x = math.atan2(tx, tz)
                    angle_y = math.atan2(ty, tz)
                    return True, angle_x, angle_y, distance, cx, cy

                # Fallback: pixel-based estimation
                angle_x = math.atan2(cx - IMG_WIDTH / 2, CAMERA_MATRIX[0, 0])
                angle_y = math.atan2(cy - IMG_HEIGHT / 2, CAMERA_MATRIX[1, 1])
                marker_px_size = np.linalg.norm(marker_corners[0] - marker_corners[2])
                distance = (self.marker_size_m * CAMERA_MATRIX[0, 0]) / max(marker_px_size, 1)
                return True, angle_x, angle_y, distance, cx, cy

        return False, 0, 0, 0, 0, 0


# ──────────────────────────────────────────────────────────
#  MAVLink Commands
# ──────────────────────────────────────────────────────────

def send_landing_target(vehicle, angle_x, angle_y, distance):
    """Send LANDING_TARGET MAVLink message for Pixhawk PrecLand."""
    msg = vehicle.message_factory.landing_target_encode(
        0,                          # time_usec
        0,                          # target_num
        mavutil.mavlink.MAV_FRAME_BODY_FRD,
        angle_x,                    # angle_x (rad)
        angle_y,                    # angle_y (rad)
        distance,                   # distance (m)
        0,                          # size_x
        0                           # size_y
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()


def send_velocity_ned(vehicle, vn, ve, vd):
    """Send velocity command in NED frame."""
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111000111,
        0, 0, 0,
        vn, ve, vd,
        0, 0, 0,
        0, 0
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()


# ──────────────────────────────────────────────────────────
#  Takeoff (GPS-Denied)
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
        alt = get_rangefinder_alt(vehicle)
        log(f"  Climbing: {alt:.1f}m")
        if alt >= target_alt * 0.95:
            log(f"  Reached {alt:.1f}m")
            return True
        time.sleep(1)


# ──────────────────────────────────────────────────────────
#  Precision Landing
# ──────────────────────────────────────────────────────────

def precision_land(vehicle, cap, detector):
    """
    Execute ArUco-guided precision landing.

    Strategy:
    1. GUIDED mode — PrecLand corrects lateral position from LANDING_TARGET
    2. Arducam OV9281 at 120 FPS detects ArUco marker
    3. Adaptive descent: slower near ground for gentle touchdown
    4. Below 0.3m: optical flow (MTF-01) handles final touchdown
    5. Below LAND_ALT_THRESHOLD with marker lock: switch to LAND
    """
    log(f"\n{'='*55}")
    log("  PRECISION LANDING — ArUco Guided (GPS-Denied)")
    log(f"  Camera: Arducam OV9281 @ {CAMERA_FPS} FPS")
    log(f"  Marker: 4x4_50 ID {TARGET_MARKER_ID}, {int(MARKER_SIZE_CM)}cm")
    log(f"{'='*55}")

    marker_last_seen = time.time()
    last_send_time = 0
    frame_count = 0
    detect_count = 0
    landing_complete = False

    if vehicle.mode.name != "GUIDED":
        vehicle.mode = VehicleMode("GUIDED")
        time.sleep(1)

    log("Starting ArUco detection loop...")

    while not landing_complete:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.005)
            continue

        frame_count += 1
        now = time.time()

        found, angle_x, angle_y, distance, cx, cy = detector.detect(frame)

        alt = get_rangefinder_alt(vehicle)

        if found:
            marker_last_seen = now
            detect_count += 1

            # Send landing target at high rate
            if now - last_send_time >= 1.0 / SEND_RATE_HZ:
                send_landing_target(vehicle, angle_x, angle_y, distance)
                last_send_time = now

            # Adaptive descent rate
            if alt > 3.0:
                descent_rate = DESCENT_SPEED
            elif alt > 1.5:
                descent_rate = DESCENT_SPEED * 0.6
            elif alt > OPTICAL_FLOW_FALLBACK_ALT:
                descent_rate = DESCENT_SPEED * 0.3
            else:
                # Very close to ground — minimal descent, let LAND handle it
                descent_rate = DESCENT_SPEED * 0.15

            send_velocity_ned(vehicle, 0, 0, descent_rate)

            # Log every ~0.5 second
            if frame_count % 60 == 0:
                offset_x_deg = math.degrees(angle_x)
                offset_y_deg = math.degrees(angle_y)
                # Pixel offset from center
                px_off_x = cx - IMG_WIDTH / 2
                px_off_y = cy - IMG_HEIGHT / 2
                log(f"  ALT={alt:.2f}m | Marker: X={offset_x_deg:+.1f}° "
                    f"Y={offset_y_deg:+.1f}° | px({px_off_x:+.0f},{px_off_y:+.0f}) | "
                    f"Dist={distance:.2f}m | Descent={descent_rate:.2f}m/s")

        else:
            time_since_seen = now - marker_last_seen

            if frame_count % 120 == 0:
                log(f"  ALT={alt:.2f}m | Marker: LOST ({time_since_seen:.1f}s)")

            if time_since_seen > MARKER_LOST_TIMEOUT:
                if alt > 2.0:
                    send_velocity_ned(vehicle, 0, 0, 0)
                    if frame_count % 120 == 0:
                        log("  Holding position — searching for marker...")
                else:
                    # Near ground without marker — optical flow handles it
                    log("  Marker lost near ground — LAND mode (optical flow fallback)")
                    vehicle.mode = VehicleMode("LAND")
                    landing_complete = True
                    break

        # Below threshold with marker lock → final LAND
        if alt < LAND_ALT_THRESHOLD and found:
            log(f"  Below {LAND_ALT_THRESHOLD}m with marker lock — switching to LAND")
            log(f"  Final offset: X={math.degrees(angle_x):+.1f}° Y={math.degrees(angle_y):+.1f}°")
            vehicle.mode = VehicleMode("LAND")
            landing_complete = True

        # Already landed check
        if vehicle.mode.name == "LAND" and alt < 0.15 and not vehicle.armed:
            log("  Touchdown detected — motors disarmed")
            landing_complete = True
            break

        time.sleep(0.005)  # ~120Hz loop to match camera

    # Wait for actual touchdown
    log("\nWaiting for touchdown (optical flow + LiDAR guiding final descent)...")
    touchdown_timeout = time.time() + 30
    while vehicle.armed and time.time() < touchdown_timeout:
        alt = get_rangefinder_alt(vehicle)
        log(f"  Landing... {alt:.2f}m")
        time.sleep(1)

    # ── Summary ──
    detect_rate = (detect_count / frame_count * 100) if frame_count > 0 else 0

    log(f"\n{'='*55}")
    log("  PRECISION LANDING — RESULTS")
    log(f"{'='*55}")
    log(f"  Camera            : Arducam OV9281 @ {CAMERA_FPS} FPS (mono)")
    log(f"  Navigation        : VIO + Optical Flow (GPS-Denied)")
    log(f"  Altitude source   : VL53L1X LiDAR rangefinder")
    log(f"  Frames processed  : {frame_count}")
    log(f"  Marker detections : {detect_count} ({detect_rate:.1f}%)")
    log(f"  Final altitude    : {get_rangefinder_alt(vehicle):.2f}m")
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
        description="IROC Task 3: ArUco Precision Landing (GPS-Denied)")
    parser.add_argument("--connect", default=DEFAULT_CONNECTION,
                        help=f"Vehicle connection string (default: {DEFAULT_CONNECTION})")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--alt", type=float, default=DEFAULT_TAKEOFF_ALT,
                        help=f"Takeoff altitude in meters (default: {DEFAULT_TAKEOFF_ALT})")
    parser.add_argument("--skip-takeoff", action="store_true",
                        help="Skip arm/takeoff if drone is already airborne")
    parser.add_argument("--camera-device", type=int, default=0,
                        help="Arducam V4L2 device index (default: 0)")
    parser.add_argument("--use-gstreamer", action="store_true",
                        help="Use GStreamer pipeline instead of V4L2")
    parser.add_argument("--marker-id", type=int, default=TARGET_MARKER_ID,
                        help=f"ArUco marker ID to track (default: {TARGET_MARKER_ID})")
    parser.add_argument("--marker-size", type=float, default=MARKER_SIZE_CM,
                        help=f"Marker size in cm (default: {MARKER_SIZE_CM})")
    args = parser.parse_args()

    log("=" * 55)
    log("  IROC QUALIFICATION - TASK 3: PRECISION LANDING")
    log("  [GPS-DENIED — ArUco + VIO + Optical Flow]")
    log("=" * 55)

    # ── Open Arducam OV9281 ──
    log(f"Opening Arducam OV9281 (device {args.camera_device})...")
    try:
        if args.use_gstreamer:
            cap = open_arducam_gstreamer(args.camera_device)
            log("  Opened via GStreamer pipeline")
        else:
            cap = open_arducam_v4l2(args.camera_device)
            log("  Opened via V4L2")
    except RuntimeError as e:
        log(f"ABORT: {e}")
        return

    # Verify camera
    ret, test_frame = cap.read()
    if not ret:
        log("ABORT: Cannot read from Arducam")
        cap.release()
        return
    h, w = test_frame.shape[:2]
    channels = 1 if len(test_frame.shape) == 2 else test_frame.shape[2]
    log(f"  Camera OK — {w}x{h}, {channels}ch {'(mono)' if channels == 1 else '(color)'}")

    # ── ArUco detector ──
    detector = ArUcoDetector(
        marker_id=args.marker_id,
        marker_size_cm=args.marker_size
    )

    # Quick marker test
    found, ax, ay, dist, cx, cy = detector.detect(test_frame)
    if found:
        log(f"  ArUco #{args.marker_id} detected in test frame (dist={dist:.2f}m)")
    else:
        log(f"  ArUco #{args.marker_id} not in test frame (will detect in flight)")

    # ── Connect to vehicle ──
    log(f"Connecting to vehicle on {args.connect}...")
    vehicle = connect(args.connect, baud=args.baud, wait_ready=True, heartbeat_timeout=60)
    log(f"  Connected — Firmware: {vehicle.version}")
    log(f"  EKF: {'OK' if vehicle.ekf_ok else 'NOT READY'}")

    try:
        if not args.skip_takeoff:
            if not arm_and_takeoff(vehicle, args.alt):
                log("ABORT: Takeoff failed")
                return
            log("Stabilizing at hover altitude for 5s...")
            time.sleep(5)
        else:
            alt = get_rangefinder_alt(vehicle)
            log(f"  Skip-takeoff mode — altitude: {alt:.1f}m")
            if vehicle.mode.name != "GUIDED":
                vehicle.mode = VehicleMode("GUIDED")
                time.sleep(1)

        # ── Execute precision landing ──
        success = precision_land(vehicle, cap, detector)

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
        log("Arducam and vehicle connection closed.")


if __name__ == "__main__":
    main()
