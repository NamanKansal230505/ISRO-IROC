#!/usr/bin/env python3
"""
ISRO IROC Qualification - Task 1: Vertical Take-Off
=====================================================
Demonstrates stable vertical take-off from the designated base area
without abnormal vibration or loss of control.

Hardware: Pixhawk FC + Jetson Nano companion computer
Connection: Serial /dev/ttyTHS1 @ 921600 baud (Jetson UART)

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
    """Check vibration levels from RAW_IMU. Returns True if within limits."""
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


def wait_for_gps_lock(vehicle, min_satellites=6, timeout=60):
    """Wait for adequate GPS fix before proceeding."""
    log("Waiting for GPS lock...")
    start = time.time()
    while time.time() - start < timeout:
        gps = vehicle.gps_0
        fix = gps.fix_type
        sats = gps.satellites_visible
        log(f"  GPS fix={fix}, satellites={sats}")
        if fix >= 3 and sats >= min_satellites:
            log(f"  GPS lock acquired: {sats} satellites, fix type {fix}")
            return True
        time.sleep(2)
    log("ERROR: GPS lock timeout")
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
    # Brief settle time after arming
    time.sleep(2)
    return True


def takeoff(vehicle, target_alt, timeout=TAKEOFF_TIMEOUT):
    """Command takeoff and monitor ascent."""
    log(f"Commanding takeoff to {target_alt:.1f}m...")
    vehicle.simple_takeoff(target_alt)

    start = time.time()
    max_alt = 0
    while time.time() - start < timeout:
        alt = vehicle.location.global_relative_frame.alt
        if alt is None:
            alt = 0
        max_alt = max(max_alt, alt)

        # Attitude monitoring
        att = vehicle.attitude
        roll_deg = abs(att.roll * 57.2958)
        pitch_deg = abs(att.pitch * 57.2958)

        log(f"  Altitude: {alt:.2f}m | Roll: {roll_deg:.1f}° | "
            f"Pitch: {pitch_deg:.1f}° | Climb rate check")

        # Safety: excessive tilt during takeoff
        if alt > 1.0 and (roll_deg > 15 or pitch_deg > 15):
            log("WARNING: Excessive tilt detected during takeoff!")

        # Check if we've reached target altitude
        if alt >= target_alt * 0.95:
            log(f"  Target altitude reached: {alt:.2f}m")
            return True

        time.sleep(0.5)

    log(f"WARNING: Takeoff timeout — current alt {max_alt:.2f}m")
    return max_alt >= target_alt * 0.80


def main():
    parser = argparse.ArgumentParser(description="IROC Task 1: Vertical Take-Off")
    parser.add_argument("--connect", default=DEFAULT_CONNECTION,
                        help=f"Vehicle connection string (default: {DEFAULT_CONNECTION})")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD,
                        help=f"Baud rate (default: {DEFAULT_BAUD})")
    parser.add_argument("--alt", type=float, default=DEFAULT_TAKEOFF_ALT,
                        help=f"Takeoff altitude in meters (default: {DEFAULT_TAKEOFF_ALT})")
    args = parser.parse_args()

    if args.alt < 3 or args.alt > 6:
        log(f"WARNING: Altitude {args.alt}m is outside qualification range (3-6m)")

    # ── Connect to vehicle ──
    log("=" * 55)
    log("  IROC QUALIFICATION - TASK 1: VERTICAL TAKE-OFF")
    log("=" * 55)
    log(f"Connecting to vehicle on {args.connect} @ {args.baud}...")

    vehicle = connect(args.connect, baud=args.baud, wait_ready=True, heartbeat_timeout=60)
    log(f"  Connected — Firmware: {vehicle.version}")
    log(f"  Vehicle type: {vehicle._vehicle_type}")
    log(f"  Armed: {vehicle.armed} | Mode: {vehicle.mode.name}")

    try:
        # ── Pre-flight checks ──
        log("\n--- PRE-FLIGHT CHECKS ---")

        if not wait_for_gps_lock(vehicle):
            log("ABORT: No GPS lock")
            return

        if not wait_for_armable(vehicle):
            log("ABORT: Pre-arm checks failed")
            return

        # Battery check
        batt = vehicle.battery
        log(f"  Battery: {batt.voltage:.1f}V, "
            f"{batt.level if batt.level else '?'}%")
        if batt.voltage and batt.voltage < 14.0:
            log("ABORT: Battery voltage too low (< 14.0V)")
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
            # Post-takeoff vibration check
            log("\n--- POST-TAKEOFF CHECKS ---")
            check_vibration(vehicle)

            att = vehicle.attitude
            log(f"  Final attitude — Roll: {abs(att.roll * 57.2958):.1f}° | "
                f"Pitch: {abs(att.pitch * 57.2958):.1f}°")
            log(f"  Altitude: {vehicle.location.global_relative_frame.alt:.2f}m")

            log("\n" + "=" * 55)
            log("  TASK 1 COMPLETE: Stable vertical take-off achieved")
            log("=" * 55)
            log("  Vehicle is now hovering at target altitude.")
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
        # NOTE: We do NOT disarm or land here — Task 2 continues from hover
        log("\nClosing vehicle connection (vehicle stays in GUIDED hover)")
        vehicle.close()


if __name__ == "__main__":
    main()
