#!/usr/bin/env python3
"""
ISRO IROC Qualification - Task 2: Hover Stability
===================================================
After take-off, the ASCEND shall hover at a fixed height (3-6m) for a
minimum of 5 minutes maintaining stable attitude and altitude.

This script monitors and logs all stability metrics during the hover
phase. It can be run standalone (arms + takes off + hovers) or after
Task 1 if the drone is already airborne in GUIDED mode.

Hardware: Pixhawk FC + Jetson Nano companion computer
Connection: Serial /dev/ttyTHS1 @ 921600 baud

Usage:
    # Standalone (full sequence):
    python3 task2_hover_stability.py --alt 5

    # Continue from Task 1 (drone already hovering):
    python3 task2_hover_stability.py --alt 5 --skip-takeoff
"""

import argparse
import time
import sys
import csv
import os
from datetime import datetime
from dronekit import connect, VehicleMode
from pymavlink import mavutil


# ── Configuration ──
DEFAULT_CONNECTION = "/dev/ttyTHS1"
DEFAULT_BAUD = 921600
DEFAULT_HOVER_ALT = 5.0             # meters
HOVER_DURATION = 300                 # 5 minutes in seconds
LOG_INTERVAL = 1.0                   # seconds between telemetry logs
ALTITUDE_TOLERANCE = 0.5             # meters drift allowed
MAX_TILT_ANGLE = 10.0                # degrees — flag if exceeded
VIBRATION_THRESHOLD = 30.0           # m/s²
POSITION_DRIFT_WARN = 1.5            # meters horizontal drift warning


def log(msg):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")
    sys.stdout.flush()


def get_vibration(vehicle):
    """Get vibration data from vehicle."""
    vibe = vehicle._master.recv_match(type='VIBRATION', blocking=True, timeout=2)
    if vibe:
        return vibe.vibration_x, vibe.vibration_y, vibe.vibration_z
    return None, None, None


def wait_for_armable(vehicle, timeout=30):
    log("Waiting for vehicle to be armable...")
    start = time.time()
    while time.time() - start < timeout:
        if vehicle.is_armable:
            log("  Vehicle is armable")
            return True
        time.sleep(2)
    return False


def arm_and_takeoff(vehicle, target_alt):
    """Full arm + takeoff sequence."""
    if not wait_for_armable(vehicle):
        log("ABORT: Vehicle not armable")
        return False

    log("Setting GUIDED mode and arming...")
    vehicle.mode = VehicleMode("GUIDED")
    while vehicle.mode.name != "GUIDED":
        time.sleep(0.5)

    vehicle.armed = True
    start = time.time()
    while not vehicle.armed:
        if time.time() - start > 15:
            log("ABORT: Arming timeout")
            return False
        time.sleep(0.5)
    log("  Armed")

    time.sleep(2)
    log(f"Taking off to {target_alt:.1f}m...")
    vehicle.simple_takeoff(target_alt)

    while True:
        alt = vehicle.location.global_relative_frame.alt or 0
        log(f"  Climbing... {alt:.1f}m")
        if alt >= target_alt * 0.95:
            log(f"  Reached target altitude: {alt:.1f}m")
            return True
        time.sleep(1)


def run_hover_test(vehicle, target_alt, duration, log_file):
    """
    Main hover stability test. Monitors altitude, attitude, position drift,
    vibration, and battery for the specified duration.
    """
    log(f"\n{'='*55}")
    log(f"  HOVER STABILITY TEST — {duration}s at {target_alt:.1f}m")
    log(f"{'='*55}")

    # Record starting position for drift calculation
    start_loc = vehicle.location.global_relative_frame
    start_lat = start_loc.lat
    start_lon = start_loc.lon

    # CSV telemetry log
    csv_writer = None
    csv_file = None
    if log_file:
        csv_file = open(log_file, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'elapsed_s', 'altitude_m', 'alt_error_m',
            'roll_deg', 'pitch_deg', 'yaw_deg',
            'vibe_x', 'vibe_y', 'vibe_z',
            'lat', 'lon', 'hdrift_m',
            'batt_v', 'batt_pct', 'mode'
        ])

    # Stats tracking
    alt_errors = []
    max_roll = 0
    max_pitch = 0
    max_vibe = 0
    max_hdrift = 0
    anomalies = 0

    start_time = time.time()
    last_log_time = 0

    log("\nTime   | Alt(m) | AltErr | Roll°  | Pitch° | Vibe   | Drift(m) | Batt")
    log("-" * 80)

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break

            now = time.time()
            if now - last_log_time < LOG_INTERVAL:
                time.sleep(0.1)
                continue
            last_log_time = now

            # ── Gather telemetry ──
            loc = vehicle.location.global_relative_frame
            alt = loc.alt if loc.alt else 0
            alt_error = alt - target_alt

            att = vehicle.attitude
            roll_deg = att.roll * 57.2958
            pitch_deg = att.pitch * 57.2958
            yaw_deg = att.yaw * 57.2958

            vx, vy, vz = get_vibration(vehicle)
            vibe_max = max(vx or 0, vy or 0, vz or 0)

            # Horizontal drift (approximate meters from lat/lon)
            dlat = (loc.lat - start_lat) * 111320
            dlon = (loc.lon - start_lon) * 111320 * 0.7  # cos(lat) approx
            hdrift = (dlat**2 + dlon**2) ** 0.5

            batt = vehicle.battery
            batt_v = batt.voltage or 0
            batt_pct = batt.level if batt.level else -1

            # ── Track stats ──
            alt_errors.append(abs(alt_error))
            max_roll = max(max_roll, abs(roll_deg))
            max_pitch = max(max_pitch, abs(pitch_deg))
            max_vibe = max(max_vibe, vibe_max)
            max_hdrift = max(max_hdrift, hdrift)

            # ── Anomaly detection ──
            flags = []
            if abs(alt_error) > ALTITUDE_TOLERANCE:
                flags.append("ALT_DRIFT")
            if abs(roll_deg) > MAX_TILT_ANGLE or abs(pitch_deg) > MAX_TILT_ANGLE:
                flags.append("TILT")
            if vibe_max > VIBRATION_THRESHOLD:
                flags.append("VIBRATION")
            if hdrift > POSITION_DRIFT_WARN:
                flags.append("POS_DRIFT")
            if vehicle.mode.name != "GUIDED":
                flags.append("MODE_CHANGE")

            if flags:
                anomalies += 1

            # ── Console output ──
            remaining = duration - elapsed
            mins, secs = divmod(int(remaining), 60)
            flag_str = f" !! {','.join(flags)}" if flags else ""
            log(f"{mins:02d}:{secs:02d} | {alt:5.2f}  | {alt_error:+.2f}  | "
                f"{roll_deg:+5.1f}  | {pitch_deg:+5.1f}  | {vibe_max:5.1f}  | "
                f"{hdrift:5.2f}    | {batt_v:.1f}V {batt_pct}%{flag_str}")

            # ── CSV log ──
            if csv_writer:
                csv_writer.writerow([
                    f"{elapsed:.1f}", f"{alt:.2f}", f"{alt_error:.2f}",
                    f"{roll_deg:.2f}", f"{pitch_deg:.2f}", f"{yaw_deg:.2f}",
                    f"{vx:.1f}" if vx else "", f"{vy:.1f}" if vy else "",
                    f"{vz:.1f}" if vz else "",
                    f"{loc.lat:.7f}", f"{loc.lon:.7f}", f"{hdrift:.2f}",
                    f"{batt_v:.2f}", str(batt_pct), vehicle.mode.name
                ])

            # ── Safety: if mode changed (e.g. failsafe), abort hover test ──
            if vehicle.mode.name not in ("GUIDED", "LOITER", "POSHOLD"):
                log(f"WARNING: Mode changed to {vehicle.mode.name} — aborting hover test")
                break

    finally:
        if csv_file:
            csv_file.close()

    # ── Generate summary report ──
    avg_alt_error = sum(alt_errors) / len(alt_errors) if alt_errors else 0
    max_alt_error = max(alt_errors) if alt_errors else 0

    log(f"\n{'='*55}")
    log("  HOVER STABILITY TEST — RESULTS")
    log(f"{'='*55}")
    log(f"  Duration          : {int(elapsed)}s / {duration}s")
    log(f"  Target altitude   : {target_alt:.1f}m")
    log(f"  Avg altitude error: {avg_alt_error:.3f}m")
    log(f"  Max altitude error: {max_alt_error:.3f}m")
    log(f"  Max roll          : {max_roll:.2f}°")
    log(f"  Max pitch         : {max_pitch:.2f}°")
    log(f"  Max vibration     : {max_vibe:.1f} m/s²")
    log(f"  Max horiz drift   : {max_hdrift:.2f}m")
    log(f"  Anomaly events    : {anomalies}")
    log(f"  Battery           : {batt_v:.1f}V")

    passed = (elapsed >= duration * 0.98 and
              max_alt_error < 1.0 and
              max_roll < 15 and max_pitch < 15)

    if passed:
        log(f"\n  RESULT: PASS")
    else:
        log(f"\n  RESULT: REVIEW NEEDED")
        if elapsed < duration * 0.98:
            log(f"    - Hover duration short ({int(elapsed)}s < {duration}s)")
        if max_alt_error >= 1.0:
            log(f"    - Altitude error too large ({max_alt_error:.2f}m)")
        if max_roll >= 15 or max_pitch >= 15:
            log(f"    - Excessive tilt (roll={max_roll:.1f}° pitch={max_pitch:.1f}°)")

    log(f"{'='*55}")

    if log_file:
        log(f"  Telemetry log saved: {log_file}")

    return passed


def main():
    parser = argparse.ArgumentParser(description="IROC Task 2: Hover Stability")
    parser.add_argument("--connect", default=DEFAULT_CONNECTION,
                        help=f"Vehicle connection string (default: {DEFAULT_CONNECTION})")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--alt", type=float, default=DEFAULT_HOVER_ALT,
                        help=f"Hover altitude in meters (default: {DEFAULT_HOVER_ALT})")
    parser.add_argument("--duration", type=int, default=HOVER_DURATION,
                        help=f"Hover duration in seconds (default: {HOVER_DURATION})")
    parser.add_argument("--skip-takeoff", action="store_true",
                        help="Skip arm/takeoff if drone is already airborne")
    parser.add_argument("--log-dir", default="/home/nano/Desktop/ISRO IROC/logs",
                        help="Directory for telemetry CSV logs")
    args = parser.parse_args()

    log("=" * 55)
    log("  IROC QUALIFICATION - TASK 2: HOVER STABILITY")
    log("=" * 55)
    log(f"Connecting to vehicle on {args.connect}...")

    vehicle = connect(args.connect, baud=args.baud, wait_ready=True, heartbeat_timeout=60)
    log(f"  Connected — Firmware: {vehicle.version}")

    # Set up log file
    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.log_dir, f"hover_test_{timestamp}.csv")

    try:
        if not args.skip_takeoff:
            if not arm_and_takeoff(vehicle, args.alt):
                log("ABORT: Takeoff failed")
                return
        else:
            alt = vehicle.location.global_relative_frame.alt or 0
            log(f"  Skip-takeoff mode — current altitude: {alt:.1f}m")
            if alt < 2.0:
                log("WARNING: Altitude below 2m — drone may not be airborne!")

            # Ensure GUIDED mode
            if vehicle.mode.name != "GUIDED":
                log(f"  Switching from {vehicle.mode.name} to GUIDED...")
                vehicle.mode = VehicleMode("GUIDED")
                time.sleep(1)

        # Hover at target altitude (re-send in case of drift)
        from dronekit import LocationGlobalRelative
        target_loc = LocationGlobalRelative(
            vehicle.location.global_relative_frame.lat,
            vehicle.location.global_relative_frame.lon,
            args.alt
        )
        vehicle.simple_goto(target_loc)
        time.sleep(3)  # Let altitude controller settle

        # Run the hover stability test
        passed = run_hover_test(vehicle, args.alt, args.duration, log_file)

        log("\n  Hover test complete. Vehicle remains in GUIDED hover.")
        log("  Ready to proceed to Task 3 (Controlled Landing).")

    except KeyboardInterrupt:
        log("\nUser interrupt — switching to LAND mode")
        vehicle.mode = VehicleMode("LAND")
    except Exception as e:
        log(f"\nERROR: {e}")
        log("Switching to LAND mode for safety")
        vehicle.mode = VehicleMode("LAND")
    finally:
        vehicle.close()


if __name__ == "__main__":
    main()
