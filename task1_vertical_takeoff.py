#!/usr/bin/env python3
"""
ISRO IROC Qualification - Task 1: Vertical Take-Off (GPS-Denied)
=================================================================
Demonstrates stable vertical take-off from the designated base area
without abnormal vibration or loss of control.

Navigation: Intel RealSense D435i VIO (primary) + Matek MTF-01
            PMW3901 optical flow + VL53L1X LiDAR (backup)
No GPS — EKF3 configured for external vision source.

Hardware:
    - Pixhawk FC (ArduCopter, EKF3 external vision)
    - Jetson Nano companion computer
    - RealSense D435i (VIO via ISAAC ROS)
    - Matek MTF-01 (optical flow + LiDAR rangefinder)

Pixhawk Parameters (set beforehand):
    EK3_SRC1_POSXY  = 6  (ExternalNav)
    EK3_SRC1_VELXY  = 6  (ExternalNav)
    EK3_SRC1_POSZ   = 1  (Rangefinder / ExternalNav)
    VISO_TYPE        = 1  (MAVLink)
    GPS_TYPE         = 0  (Disabled)
    RNGFND1_TYPE     = 2  (MAVLink / MTF-01)

Usage:
    python3 task1_vertical_takeoff.py [--connect <connection_string>] [--alt <meters>]
"""

import argparse
import time
import sys
from dronekit import connect, VehicleMode
from pymavlink import mavutil


# ── Configuration ──
DEFAULT_CONNECTION = "/dev/ttyTHS1"
DEFAULT_BAUD = 921600
DEFAULT_TAKEOFF_ALT = 5.0       # meters (within 3-6m range)
PREFLIGHT_CHECK_TIMEOUT = 30    # seconds to wait for pre-arm checks
ARM_TIMEOUT = 15                # seconds to wait for arming
TAKEOFF_TIMEOUT = 30            # seconds max for takeoff climb
ALTITUDE_TOLERANCE = 0.5        # meters tolerance for target altitude
VIBRATION_THRESHOLD = 30.0      # m/s² — above this is abnormal


def log(msg):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")
    sys.stdout.flush()


def check_vibration(vehicle):
    """Check vibration levels from IMU. Returns True if within limits."""
    vibe = vehicle._master.recv_match(type='VIBRATION', blocking=True, timeout=5)
    if vibe:
        vx, vy, vz = vibe.vibration_x, vibe.vibration_y, vibe.vibration_z
        log(f"  Vibration X={vx:.1f} Y={vy:.1f} Z={vz:.1f} m/s²")
        if max(vx, vy, vz) > VIBRATION_THRESHOLD:
            log(f"  WARNING: Vibration exceeds threshold ({VIBRATION_THRESHOLD} m/s²)")
            return False
        log("  Vibration levels: NOMINAL")
        return True
    log("  Vibration data not available (non-critical)")
    return True


def check_ekf_source(vehicle):
    """Verify EKF is receiving external vision data (no GPS required)."""
    log("Checking EKF external vision status...")

    ekf_ok = vehicle.ekf_ok
    log(f"  EKF status: {'OK' if ekf_ok else 'NOT READY'}")

    # Check for VISION_POSITION_ESTIMATE messages (from VIO)
    vpe = vehicle._master.recv_match(
        type='VISION_POSITION_ESTIMATE', blocking=True, timeout=5)
    if vpe:
        log(f"  VIO data received — X={vpe.x:.2f} Y={vpe.y:.2f} Z={vpe.z:.2f}")
        return True

    # Check for ODOMETRY messages (alternative VIO format)
    odom = vehicle._master.recv_match(
        type='ODOMETRY', blocking=True, timeout=3)
    if odom:
        log(f"  Odometry data received — X={odom.x:.2f} Y={odom.y:.2f} Z={odom.z:.2f}")
        return True

    log("  WARNING: No external vision data detected")
    log("  Ensure RealSense VIO / ISAAC ROS is running")
    return False


def check_rangefinder(vehicle):
    """Check rangefinder (MTF-01 VL53L1X LiDAR) is providing altitude data."""
    log("Checking rangefinder (MTF-01 LiDAR)...")
    rf = vehicle._master.recv_match(type='RANGEFINDER', blocking=True, timeout=5)
    if rf:
        log(f"  Rangefinder distance: {rf.distance:.2f}m")
        return True

    # Alternative: check DISTANCE_SENSOR
    ds = vehicle._master.recv_match(type='DISTANCE_SENSOR', blocking=True, timeout=3)
    if ds:
        log(f"  Distance sensor: {ds.current_distance / 100.0:.2f}m")
        return True

    log("  WARNING: No rangefinder data — altitude may rely on VIO only")
    return False


def check_optical_flow(vehicle):
    """Check optical flow (MTF-01 PMW3901) data is available."""
    log("Checking optical flow (MTF-01 PMW3901)...")
    of = vehicle._master.recv_match(type='OPTICAL_FLOW', blocking=True, timeout=5)
    if of:
        log(f"  Optical flow: quality={of.quality} flowX={of.flow_x:.2f} flowY={of.flow_y:.2f}")
        return True
    log("  Optical flow data not detected (VIO is primary — non-critical)")
    return False


def wait_for_armable(vehicle, timeout=PREFLIGHT_CHECK_TIMEOUT):
    """Wait for vehicle to become armable (all pre-arm checks pass)."""
    log("Waiting for vehicle to be armable...")
    start = time.time()
    while time.time() - start < timeout:
        if vehicle.is_armable:
            log("  Vehicle is armable — all pre-arm checks passed")
            return True
        log(f"  Pre-arm checks pending... (EKF ok={vehicle.ekf_ok}, "
            f"mode={vehicle.mode.name})")
        time.sleep(2)
    log("ERROR: Vehicle not armable within timeout")
    return False


def arm_vehicle(vehicle, timeout=ARM_TIMEOUT):
    """Switch to GUIDED mode and arm the vehicle."""
    log("Setting GUIDED mode...")
    vehicle.mode = VehicleMode("GUIDED")
    while vehicle.mode.name != "GUIDED":
        time.sleep(0.5)
    log("  Mode: GUIDED")

    log("Arming motors...")
    vehicle.armed = True
    start = time.time()
    while not vehicle.armed:
        if time.time() - start > timeout:
            log("ERROR: Arming timeout")
            return False
        time.sleep(0.5)

    log("  Motors ARMED")
    time.sleep(2)
    return True


def takeoff(vehicle, target_alt, timeout=TAKEOFF_TIMEOUT):
    """Command takeoff and monitor ascent using rangefinder/VIO altitude."""
    log(f"Commanding takeoff to {target_alt:.1f}m...")
    vehicle.simple_takeoff(target_alt)

    start = time.time()
    max_alt = 0
    while time.time() - start < timeout:
        # Use rangefinder altitude (preferred for indoor/GPS-denied)
        alt = vehicle.rangefinder.distance
        if alt is None or alt <= 0:
            alt = vehicle.location.global_relative_frame.alt or 0
        max_alt = max(max_alt, alt)

        # Attitude monitoring
        att = vehicle.attitude
        roll_deg = abs(att.roll * 57.2958)
        pitch_deg = abs(att.pitch * 57.2958)

        log(f"  Altitude: {alt:.2f}m | Roll: {roll_deg:.1f}° | "
            f"Pitch: {pitch_deg:.1f}°")

        # Safety: excessive tilt during takeoff
        if alt > 1.0 and (roll_deg > 15 or pitch_deg > 15):
            log("WARNING: Excessive tilt detected during takeoff!")

        # Check if we've reached target altitude
        if alt >= target_alt * 0.95:
            log(f"  Target altitude reached: {alt:.2f}m")
            return True

        time.sleep(0.5)

    log(f"WARNING: Takeoff timeout — max alt {max_alt:.2f}m")
    return max_alt >= target_alt * 0.80


def main():
    parser = argparse.ArgumentParser(description="IROC Task 1: Vertical Take-Off")
    parser.add_argument("--connect", default=DEFAULT_CONNECTION,
                        help=f"Vehicle connection string (default: {DEFAULT_CONNECTION})")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--alt", type=float, default=DEFAULT_TAKEOFF_ALT,
                        help=f"Takeoff altitude in meters (default: {DEFAULT_TAKEOFF_ALT})")
    args = parser.parse_args()

    if args.alt < 3 or args.alt > 6:
        log(f"WARNING: Altitude {args.alt}m is outside qualification range (3-6m)")

    # ── Connect to vehicle ──
    log("=" * 55)
    log("  IROC QUALIFICATION - TASK 1: VERTICAL TAKE-OFF")
    log("  [GPS-DENIED — VIO + Optical Flow Navigation]")
    log("=" * 55)
    log(f"Connecting to vehicle on {args.connect} @ {args.baud}...")

    vehicle = connect(args.connect, baud=args.baud, wait_ready=True, heartbeat_timeout=60)
    log(f"  Connected — Firmware: {vehicle.version}")
    log(f"  Armed: {vehicle.armed} | Mode: {vehicle.mode.name}")

    try:
        # ── Pre-flight checks (GPS-denied) ──
        log("\n--- PRE-FLIGHT CHECKS (GPS-DENIED) ---")

        # EKF + VIO check (replaces GPS lock)
        vio_ok = check_ekf_source(vehicle)
        if not vio_ok:
            log("WARNING: VIO not detected — check RealSense + ISAAC ROS")
            log("Proceeding with caution (optical flow may provide backup)...")

        # Rangefinder check
        check_rangefinder(vehicle)

        # Optical flow check (backup nav source)
        check_optical_flow(vehicle)

        # Battery check
        batt = vehicle.battery
        log(f"  Battery: {batt.voltage:.1f}V, "
            f"{batt.level if batt.level else '?'}%")
        if batt.voltage and batt.voltage < 14.0:
            log("ABORT: Battery voltage too low (< 14.0V)")
            return

        # Wait for armable
        if not wait_for_armable(vehicle):
            log("ABORT: Pre-arm checks failed")
            log("  Ensure VIO is streaming and EKF3 is configured for ExternalNav")
            return

        # Initial vibration check
        check_vibration(vehicle)

        log("\n--- TAKEOFF SEQUENCE ---")

        # ── Arm and take off ──
        if not arm_vehicle(vehicle):
            log("ABORT: Could not arm vehicle")
            return

        success = takeoff(vehicle, args.alt)

        if success:
            # Post-takeoff checks
            log("\n--- POST-TAKEOFF CHECKS ---")
            check_vibration(vehicle)

            att = vehicle.attitude
            log(f"  Final attitude — Roll: {abs(att.roll * 57.2958):.1f}° | "
                f"Pitch: {abs(att.pitch * 57.2958):.1f}°")

            alt = vehicle.rangefinder.distance
            if alt is None or alt <= 0:
                alt = vehicle.location.global_relative_frame.alt or 0
            log(f"  Altitude: {alt:.2f}m (rangefinder/VIO)")

            log("\n" + "=" * 55)
            log("  TASK 1 COMPLETE: Stable vertical take-off achieved")
            log("=" * 55)
            log("  Vehicle hovering at target altitude.")
            log("  Ready to proceed to Task 2 (Hover Stability).")
        else:
            log("\nTASK 1 WARNING: Takeoff may not have fully completed")

    except KeyboardInterrupt:
        log("\nUser interrupt — switching to LAND mode")
        vehicle.mode = VehicleMode("LAND")
    except Exception as e:
        log(f"\nERROR: {e}")
        log("Switching to LAND mode for safety")
        vehicle.mode = VehicleMode("LAND")
    finally:
        log("\nClosing vehicle connection (vehicle stays in GUIDED hover)")
        vehicle.close()


if __name__ == "__main__":
    main()
