"""ResQTrack live detector - YOLO11 tracking, incident policy and dispatch.

    python main.py                                   # default demo video
    python main.py --source data/test.mp4 --loop     # any file, replayed
    python main.py --source 0                        # webcam
    python main.py --source rtsp://...               # live CCTV feed
    python main.py --headless                        # server / no display
    python main.py --sensitivity strict --no-alerts  # tuning runs

While it runs the window shows, for every tracked road user, the YOLO11 class,
tracker id, detection confidence, estimated speed, heading, colour and motion
state.  When the incident policy confirms an accident the detector posts it to
the ResQTrack backend, which pushes it to responders in real time.

Press T at any time to inject a rehearsed test alert - useful on stage when the
demo video has not reached the crash yet.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from accident_logic import reset_vehicle_history, update_vehicle
from temporal_features_engine import (
    FEATURES,
    STRIDE_FRAMES,
    WINDOW_FRAMES,
    aggregate_window,
    extract_frame_features,
)
from vision.alert_client import AlertClient, AlertPayload
from vision.incident_engine import CONFIRMED, PERSON, REVIEW, VEHICLE, EnginePolicy, IncidentEngine
from vision import overlay
from vision.stream_server import StreamServer
from vision.vehicle_profile import (
    PERSON_CLASS,
    TRACKED_CLASSES,
    ScaleEstimator,
    VehicleRegistry,
)

WINDOW_NAME = "ResQTrack - Live Incident Detection"
SNAPSHOT_DIR = Path("snapshots")


# ============================================================
# COMMAND LINE
# ============================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ResQTrack live accident detection and emergency dispatch",
    )
    parser.add_argument("--source", default="data/accident.mp4",
                        help="video file, webcam index or RTSP url")
    parser.add_argument("--model", default="yolo11n.pt",
                        help="YOLO11 weights (yolo11n/s/m/l.pt)")
    parser.add_argument("--temporal-model", default="resqtrack_final_model.pkl",
                        help="trained temporal context model, or 'none'")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--sensitivity", default="balanced",
                        choices=("balanced", "high", "strict"),
                        help="incident policy preset")

    parser.add_argument("--camera-id", default="CAM-001")
    parser.add_argument("--lat", type=float, default=12.9719, help="camera latitude")
    parser.add_argument("--lon", type=float, default=77.5937, help="camera longitude")
    parser.add_argument("--location-name", default="Trinity Junction, MG Road")

    parser.add_argument("--backend", default="http://localhost:8000",
                        help="ResQTrack backend base url")
    parser.add_argument("--no-alerts", action="store_true", help="detect but never dispatch")
    parser.add_argument("--alert-cooldown", type=float, default=25.0,
                        help="seconds before the same camera may raise another alert")

    parser.add_argument("--stream-port", type=int, default=8001,
                        help="MJPEG port the dashboard embeds")
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--headless", action="store_true", help="no GUI window")
    parser.add_argument("--loop", action="store_true", help="replay the video forever")
    parser.add_argument("--record", default="", help="write the annotated view to this mp4")
    parser.add_argument("--meters-per-pixel", type=float, default=0.0,
                        help="fixed camera calibration; 0 = estimate from vehicle sizes")
    parser.add_argument("--max-fps", type=float, default=0.0,
                        help="throttle processing (0 = source fps for files, free-run for cameras)")
    return parser.parse_args(argv)


# ============================================================
# HELPERS
# ============================================================

def open_source(source: str) -> tuple[cv2.VideoCapture, float, bool]:
    """Open a file, a webcam index or a network stream."""
    is_live = False
    if source.isdigit():
        capture = cv2.VideoCapture(int(source))
        is_live = True
    else:
        capture = cv2.VideoCapture(source)
        is_live = source.startswith(("rtsp://", "http://", "https://", "udp://"))
    if not capture.isOpened():
        capture.release()
        raise SystemExit(f"ResQTrack: could not open video source '{source}'")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 1.0 or fps > 240.0:
        fps = 25.0
    return capture, fps, is_live


def class_name_for(model, class_id: int) -> str:
    names = getattr(model, "names", {})
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def load_temporal_model(path: str):
    """The temporal model is optional context; a missing file must not stop us."""
    if not path or path.lower() == "none":
        return None
    try:
        import joblib

        package = joblib.load(path)
        model, features = package["model"], list(package["features"])
    except Exception as error:  # noqa: BLE001 - any failure degrades to no context
        print(f"[model] temporal context disabled ({type(error).__name__}: {error})")
        return None
    if features != list(FEATURES):
        print("[model] temporal context disabled: feature order does not match the engine")
        return None
    print(f"[model] temporal context loaded ({len(features)} features)")
    return model


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_payload(args, evidence, registry, snapshot: bytes | None) -> AlertPayload:
    involved_ids = set(evidence.actor_ids)
    vehicles = [
        profile.as_dict()
        for profile in registry.profiles.values()
        if profile.actor_id in involved_ids or profile.state != "STOPPED"
    ][:20]
    return AlertPayload(
        camera_id=args.camera_id,
        latitude=args.lat,
        longitude=args.lon,
        kind=evidence.kind,
        label=evidence.label,
        severity=evidence.severity,
        confidence=float(evidence.confidence),
        reason=evidence.reason,
        frame=evidence.frame,
        evidence_types=list(evidence.evidence_types),
        signals=[signal.as_dict() for signal in evidence.signals],
        involved=[dict(item) for item in evidence.involved],
        vehicles=vehicles,
        ml_probability=float(evidence.ml_probability),
        detected_at=now_iso(),
        snapshot=snapshot,
    )


def rehearsal_evidence(frame_number: int):
    """A synthetic CONFIRMED verdict for the T key, clearly marked as a drill."""
    from vision.incident_engine import IncidentEvidence, IncidentSignal

    return IncidentEvidence(
        status=CONFIRMED,
        kind="vehicle_vehicle_collision",
        confidence=0.91,
        actor_ids=(101, 102),
        reason="operator-triggered rehearsal alert (not a real detection)",
        severity="HIGH",
        signals=(
            IncidentSignal("approach", 0.9, "rehearsal"),
            IncidentSignal("contact", 0.9, "rehearsal"),
            IncidentSignal("disruption", 0.9, "rehearsal"),
        ),
        evidence_types=("approach", "contact", "disruption"),
        frame=frame_number,
    )


# ============================================================
# MAIN
# ============================================================

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print()
    print("=" * 66)
    print("  ResQTrack - AI accident detection and emergency response")
    print("=" * 66)
    print(f"  camera     : {args.camera_id} @ {args.lat:.5f}, {args.lon:.5f}")
    print(f"  location   : {args.location_name}")
    print(f"  source     : {args.source}")
    print(f"  policy     : {args.sensitivity}")
    print("=" * 66)
    print()

    from ultralytics import YOLO

    yolo = YOLO(args.model)
    print(f"[yolo] {args.model} ready")

    temporal_model = load_temporal_model(args.temporal_model)
    capture, fps, is_live = open_source(args.source)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[video] {fps:.1f} fps, {total_frames if total_frames > 0 else 'live'} frames")

    policy = EnginePolicy.for_sensitivity(args.sensitivity)
    frame_size = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    engine = IncidentEngine(policy=policy, fps=fps, frame_size=frame_size)
    scale = ScaleEstimator(args.meters_per_pixel or None)
    registry = VehicleRegistry(scale, fps)

    alerts = AlertClient(args.backend, enabled=not args.no_alerts)
    alerts.start()
    if not args.no_alerts:
        reachable = alerts.check_backend()
        print(
            f"[backend] {args.backend} "
            + ("reachable" if reachable else "NOT reachable - alerts will spool to alerts_offline/")
        )

    stream: StreamServer | None = None
    if not args.no_stream:
        stream = StreamServer(port=args.stream_port)
        url = stream.start()
        if url:
            print(f"[stream] annotated feed at {url}")

    writer: cv2.VideoWriter | None = None
    display = not args.headless
    if display:
        try:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, 1360, 766)
        except cv2.error:
            print("[display] no GUI available, continuing headless")
            display = False

    reset_vehicle_history()
    SNAPSHOT_DIR.mkdir(exist_ok=True)

    frame_number = 0
    frame_features: list[dict] = []
    previous_vehicles: dict | None = None
    previous_previous_vehicles: dict | None = None
    ml_probability = 0.0
    peak_ml_probability = 0.0

    alert_active_until = 0.0
    last_alert_time = -1e9
    alert_evidence = None
    confirmed_count = 0
    fps_samples: deque[float] = deque(maxlen=30)
    show_cards = True
    show_roster = True
    fullscreen = False
    target_period = 0.0
    if args.max_fps > 0:
        target_period = 1.0 / args.max_fps
    elif not is_live:
        target_period = 1.0 / fps

    try:
        while True:
            loop_start = time.time()
            ok, frame = capture.read()
            if not ok:
                if args.loop and not is_live:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    # A replay is a new scene: stale tracks would create ghost events.
                    engine.reset()
                    registry.profiles.clear()
                    frame_features.clear()
                    previous_vehicles = previous_previous_vehicles = None
                    continue
                print("[video] source finished")
                break

            frame_number += 1

            # ----------------------------------------------------------
            # YOLO11 detection + ByteTrack identities
            # ----------------------------------------------------------
            results = yolo.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=args.conf,
                classes=list(TRACKED_CLASSES),
                verbose=False,
            )
            if not results:
                continue
            result = results[0]

            detections: list[dict] = []
            actors: dict[int, dict] = {}
            vehicles: dict[int, dict] = {}

            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                ids = boxes.id.cpu().numpy().astype(int)
                classes = boxes.cls.cpu().numpy().astype(int)
                confidences = boxes.conf.cpu().numpy()

                for box, actor_id, class_id, confidence in zip(xyxy, ids, classes, confidences):
                    actor_id = int(actor_id)
                    class_id = int(class_id)
                    box = tuple(float(value) for value in box)
                    kind = PERSON if class_id == PERSON_CLASS else VEHICLE
                    name = class_name_for(yolo, class_id)

                    record = {
                        "id": actor_id,
                        "box": box,
                        "class_id": class_id,
                        "class_name": name,
                        "kind": kind,
                        "confidence": float(confidence),
                    }
                    detections.append(record)
                    actors[actor_id] = record

                    if kind == VEHICLE:
                        centre = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
                        vehicles[actor_id] = {"center": centre, "box": box}
                        update_vehicle(actor_id, centre)

            # ----------------------------------------------------------
            # Temporal context model (corroboration only)
            # ----------------------------------------------------------
            if temporal_model is not None:
                frame_features.append(
                    extract_frame_features(vehicles, previous_vehicles, previous_previous_vehicles)
                )
                del frame_features[:-WINDOW_FRAMES]
                window_ready = len(frame_features) >= WINDOW_FRAMES
                stride_ready = (
                    frame_number >= WINDOW_FRAMES
                    and (frame_number - WINDOW_FRAMES) % STRIDE_FRAMES == 0
                )
                if window_ready and stride_ready:
                    aggregated = aggregate_window(frame_features)
                    if aggregated is not None:
                        model_input = pd.DataFrame(
                            [[aggregated[name] for name in FEATURES]], columns=list(FEATURES)
                        )
                        ml_probability = float(temporal_model.predict_proba(model_input)[0][1])
                        peak_ml_probability = max(peak_ml_probability, ml_probability)

            previous_previous_vehicles = previous_vehicles
            previous_vehicles = vehicles

            # ----------------------------------------------------------
            # Incident policy
            # ----------------------------------------------------------
            evidence = engine.update(
                actors,
                frame_number=frame_number,
                timestamp=frame_number / fps,
                ml_probability=ml_probability,
            )

            profiles = registry.update(
                frame,
                detections,
                frame_number,
                involved_ids=set(evidence.actor_ids) if evidence.status != "NORMAL" else set(),
                sample_colour=frame_number % 5 == 1,
            )

            # ----------------------------------------------------------
            # Dispatch
            # ----------------------------------------------------------
            now = time.time()
            if evidence.confirmed and (now - last_alert_time) >= args.alert_cooldown:
                last_alert_time = now
                alert_active_until = now + 12.0
                alert_evidence = evidence
                confirmed_count += 1
                _log_confirmation(args, evidence, frame_number)

                annotated_for_snapshot = _render(
                    frame.copy(), profiles, evidence, engine, args, scale,
                    True, show_cards, show_roster, fps_samples, alerts.state,
                )
                ok_encode, encoded = cv2.imencode(".jpg", annotated_for_snapshot)
                snapshot = encoded.tobytes() if ok_encode else None
                if snapshot:
                    path = SNAPSHOT_DIR / f"{args.camera_id}-frame{frame_number}.jpg"
                    path.write_bytes(snapshot)
                alerts.send(build_payload(args, evidence, registry, snapshot))

            alert_active = now < alert_active_until
            if not alert_active:
                alert_evidence = None

            # ----------------------------------------------------------
            # Render
            # ----------------------------------------------------------
            fps_samples.append(1.0 / max(1e-3, time.time() - loop_start))
            annotated = _render(
                frame, profiles, alert_evidence or evidence, engine, args, scale,
                alert_active, show_cards, show_roster, fps_samples, alerts.state,
            )

            if stream is not None:
                stream.publish(annotated)
            if args.record:
                if writer is None:
                    writer = cv2.VideoWriter(
                        args.record,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (annotated.shape[1], annotated.shape[0]),
                    )
                writer.write(annotated)

            if display:
                cv2.imshow(WINDOW_NAME, annotated)
                wait = max(1, int((target_period - (time.time() - loop_start)) * 1000)) if target_period else 1
                key = cv2.waitKey(wait) & 0xFF
                if key in (ord("q"), 27):
                    print("[input] stopped by operator")
                    break
                if key == ord("f"):
                    fullscreen = not fullscreen
                    cv2.setWindowProperty(
                        WINDOW_NAME,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
                    )
                if key == ord("i"):
                    show_cards = not show_cards
                if key == ord("r"):
                    show_roster = not show_roster
                if key == ord("t"):
                    drill = rehearsal_evidence(frame_number)
                    alert_evidence = drill
                    alert_active_until = time.time() + 12.0
                    last_alert_time = time.time()
                    print("[drill] rehearsal alert dispatched")
                    alerts.send(build_payload(args, drill, registry, None))
            elif target_period:
                remaining = target_period - (time.time() - loop_start)
                if remaining > 0:
                    time.sleep(remaining)

    except KeyboardInterrupt:
        print("\n[input] interrupted")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        if stream is not None:
            stream.stop()
        alerts.stop()
        reset_vehicle_history()

        print()
        print("=" * 66)
        print("  ResQTrack detector stopped")
        print(f"  frames processed      : {frame_number}")
        print(f"  incidents confirmed   : {confirmed_count}")
        print(f"  alerts delivered      : {alerts.sent_count}")
        print(f"  alerts spooled offline: {alerts.failed_count}")
        print(f"  peak model context    : {peak_ml_probability:.4f}")
        print("=" * 66)
    return 0


def _render(
    frame: np.ndarray,
    profiles,
    evidence,
    engine: IncidentEngine,
    args,
    scale: ScaleEstimator,
    alert_active: bool,
    show_cards: bool,
    show_roster: bool,
    fps_samples,
    dispatch_state: str,
) -> np.ndarray:
    values = list(profiles.values()) if isinstance(profiles, dict) else list(profiles)
    overlay.draw_detection_boxes(frame, values)

    reserved: list[tuple[int, int, int, int]] = []
    status_rect = overlay.draw_status(
        frame,
        evidence,
        alert_active,
        engine.scene_summary(),
        args.camera_id,
        dispatch_state,
        sum(fps_samples) / len(fps_samples) if fps_samples else 0.0,
    )
    reserved.append(status_rect)

    if show_roster:
        roster_rect = overlay.draw_roster(
            frame, values, scale.calibrated, scale.metres_per_pixel, reserved
        )
        if roster_rect:
            reserved.append(roster_rect)
    if show_cards:
        overlay.draw_vehicle_cards(frame, values, scale.calibrated, reserved)
    if alert_active:
        overlay.draw_alert_border(frame, evidence, abs((time.time() * 2) % 2 - 1))
    overlay.draw_help(frame)
    return frame


def _log_confirmation(args, evidence, frame_number: int) -> None:
    print()
    print("!" * 66)
    print(f"  ACCIDENT CONFIRMED  -  {evidence.label}  ({evidence.severity})")
    print("!" * 66)
    print(f"  camera     : {args.camera_id}  ({args.location_name})")
    print(f"  location   : {args.lat:.6f}, {args.lon:.6f}")
    print(f"  frame      : {frame_number}")
    print(f"  confidence : {evidence.confidence:.2f}")
    print(f"  evidence   : {', '.join(evidence.evidence_types)}")
    print(f"  reason     : {evidence.reason}")
    for item in evidence.involved:
        print(
            f"    - {item.get('class_name', '?'):<12} #{item.get('id')}  "
            f"heading {item.get('heading', '-')}"
        )
    print(f"  ML context : {evidence.ml_probability:.3f} (corroboration only)")
    print("!" * 66)
    print()


if __name__ == "__main__":
    sys.exit(main())
