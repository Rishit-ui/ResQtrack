"""ResQTrack live detector - YOLO11 tracking, incident policy and dispatch.

    python main.py                                   # default demo video
    python main.py --source data/test.mp4 --loop     # any file, replayed
    python main.py --source 0                        # webcam
    python main.py --source rtsp://...               # live CCTV feed
    python main.py --headless                        # server / no display
    python main.py --sensitivity strict --no-alerts  # tuning runs

While it runs the window shows, for every tracked road user, the YOLO11 class,
tracker id, detection confidence, estimated speed, heading, colour and motion
state. When the incident policy confirms an accident the detector posts it to
the ResQTrack backend, which pushes it to responders in real time.

The detector also records model runs, vehicle observations and system events
in the ResQTrack SQLite database.

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
from uuid import uuid4

import cv2
import numpy as np
import pandas as pd

from accident_logic import reset_vehicle_history, update_vehicle

from backend.database import (
    connect,
    finish_model_run,
    initialise as initialise_database,
    log_system,
    register_camera,
    save_vehicle_observation,
    start_model_run,
    update_model_run,
)

from temporal_features_engine import (
    FEATURES,
    STRIDE_FRAMES,
    WINDOW_FRAMES,
    aggregate_window,
    extract_frame_features,
)

from vision.alert_client import AlertClient, AlertPayload

from vision.incident_engine import (
    CONFIRMED,
    PERSON,
    VEHICLE,
    EnginePolicy,
    IncidentEngine,
)

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

def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "ResQTrack live accident detection "
            "and emergency dispatch"
        ),
    )

    parser.add_argument(
        "--source",
        default="data/accident.mp4",
        help="video file, webcam index or RTSP url",
    )

    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="YOLO11 weights (yolo11n/s/m/l.pt)",
    )

    parser.add_argument(
        "--temporal-model",
        default="resqtrack_final_model.pkl",
        help="trained temporal context model, or 'none'",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="YOLO confidence threshold",
    )

    parser.add_argument(
        "--sensitivity",
        default="balanced",
        choices=(
            "balanced",
            "high",
            "strict",
        ),
        help="incident policy preset",
    )

    parser.add_argument(
        "--camera-id",
        default="CAM-001",
    )

    parser.add_argument(
        "--lat",
        type=float,
        default=12.9719,
        help="camera latitude",
    )

    parser.add_argument(
        "--lon",
        type=float,
        default=77.5937,
        help="camera longitude",
    )

    parser.add_argument(
        "--location-name",
        default="Trinity Junction, MG Road",
    )

    parser.add_argument(
        "--backend",
        default="http://localhost:8000",
        help="ResQTrack backend base url",
    )

    parser.add_argument(
        "--no-alerts",
        action="store_true",
        help="detect but never dispatch",
    )

    parser.add_argument(
        "--alert-cooldown",
        type=float,
        default=25.0,
        help="seconds before the same camera may raise another alert",
    )

    parser.add_argument(
        "--stream-port",
        type=int,
        default=8001,
        help="MJPEG port the dashboard embeds",
    )

    parser.add_argument(
        "--no-stream",
        action="store_true",
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="no GUI window",
    )

    parser.add_argument(
        "--loop",
        action="store_true",
        help="replay the video forever",
    )

    parser.add_argument(
        "--record",
        default="",
        help="write the annotated view to this mp4",
    )

    parser.add_argument(
        "--meters-per-pixel",
        type=float,
        default=0.0,
        help=(
            "fixed camera calibration; "
            "0 = estimate from vehicle sizes"
        ),
    )

    parser.add_argument(
        "--max-fps",
        type=float,
        default=0.0,
        help=(
            "throttle processing "
            "(0 = source fps for files, "
            "free-run for cameras)"
        ),
    )

    parser.add_argument(
        "--db-sample-every",
        type=int,
        default=1,
        help="store vehicle observations every N frames",
    )

    return parser.parse_args(argv)


# ============================================================
# VIDEO SOURCE
# ============================================================

def open_source(
    source: str,
) -> tuple[cv2.VideoCapture, float, bool]:

    """Open a file, webcam index or network stream."""

    is_live = False

    if source.isdigit():

        capture = cv2.VideoCapture(
            int(source)
        )

        is_live = True

    else:

        capture = cv2.VideoCapture(
            source
        )

        is_live = source.startswith(
            (
                "rtsp://",
                "http://",
                "https://",
                "udp://",
            )
        )

    if not capture.isOpened():

        capture.release()

        raise SystemExit(
            "ResQTrack: could not open "
            f"video source '{source}'"
        )

    fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
    )

    if fps <= 1.0 or fps > 240.0:
        fps = 25.0

    return capture, fps, is_live


# ============================================================
# CLASS NAME
# ============================================================

def class_name_for(
    model,
    class_id: int,
) -> str:

    names = getattr(
        model,
        "names",
        {},
    )

    if isinstance(names, dict):

        return str(
            names.get(
                class_id,
                class_id,
            )
        )

    if (
        isinstance(
            names,
            (list, tuple),
        )
        and 0 <= class_id < len(names)
    ):

        return str(
            names[class_id]
        )

    return str(class_id)


# ============================================================
# TEMPORAL MODEL
# ============================================================

def load_temporal_model(path: str):
    """Load the trained temporal context model."""

    if (
        not path
        or path.lower() == "none"
    ):

        return None

    try:

        import joblib

        package = joblib.load(
            path
        )

        model = package["model"]

        features = list(
            package["features"]
        )

    except Exception as error:

        print(
            "[model] temporal context disabled "
            f"({type(error).__name__}: {error})"
        )

        return None

    if features != list(FEATURES):

        print(
            "[model] temporal context disabled: "
            "feature order does not match the engine"
        )

        return None

    print(
        f"[model] temporal context loaded "
        f"({len(features)} features)"
    )

    return model


# ============================================================
# TIME
# ============================================================

def now_iso() -> str:

    return datetime.now(
        timezone.utc
    ).isoformat()


# ============================================================
# SPEED
# ============================================================

def pixel_speed(
    previous_center: tuple[float, float] | None,
    current_center: tuple[float, float],
    fps: float,
) -> float:

    """
    Image-space speed in pixels/second.

    This is NOT km/h.
    """

    if (
        previous_center is None
        or fps <= 0
    ):

        return 0.0

    dx = (
        current_center[0]
        - previous_center[0]
    )

    dy = (
        current_center[1]
        - previous_center[1]
    )

    return float(
        np.hypot(dx, dy) * fps
    )


# ============================================================
# HEADING
# ============================================================

def heading_label(
    previous_center: tuple[float, float] | None,
    current_center: tuple[float, float],
) -> str:

    if previous_center is None:

        return "--"

    dx = (
        current_center[0]
        - previous_center[0]
    )

    dy = (
        current_center[1]
        - previous_center[1]
    )

    if np.hypot(dx, dy) < 1.5:

        return "still"

    angle = (
        np.degrees(
            np.arctan2(
                dx,
                -dy,
            )
        )
        + 360.0
    ) % 360.0

    labels = [
        "N",
        "NE",
        "E",
        "SE",
        "S",
        "SW",
        "W",
        "NW",
    ]

    return labels[
        int(
            (angle + 22.5)
            / 45.0
        )
        % 8
    ]


# ============================================================
# ALERT PAYLOAD
# ============================================================

def build_payload(
    args,
    evidence,
    registry,
    snapshot: bytes | None,
) -> AlertPayload:

    involved_ids = set(
        evidence.actor_ids
    )

    vehicles = [
        profile.as_dict()
        for profile in registry.profiles.values()
        if (
            profile.actor_id
            in involved_ids
            or profile.state != "STOPPED"
        )
    ][:20]

    return AlertPayload(
        camera_id=args.camera_id,

        latitude=args.lat,
        longitude=args.lon,

        kind=evidence.kind,

        label=evidence.label,

        severity=evidence.severity,

        confidence=float(
            evidence.confidence
        ),

        reason=evidence.reason,

        frame=evidence.frame,

        evidence_types=list(
            evidence.evidence_types
        ),

        signals=[
            signal.as_dict()
            for signal
            in evidence.signals
        ],

        involved=[
            dict(item)
            for item
            in evidence.involved
        ],

        vehicles=vehicles,

        ml_probability=float(
            evidence.ml_probability
        ),

        detected_at=now_iso(),

        snapshot=snapshot,
    )


# ============================================================
# REHEARSAL EVENT
# ============================================================

def rehearsal_evidence(
    frame_number: int,
):

    """
    Synthetic CONFIRMED event for T key.

    Clearly marked as rehearsal.
    """

    from vision.incident_engine import (
        IncidentEvidence,
        IncidentSignal,
    )

    return IncidentEvidence(
        status=CONFIRMED,

        kind="vehicle_vehicle_collision",

        confidence=0.91,

        actor_ids=(101, 102),

        reason=(
            "operator-triggered rehearsal alert "
            "(not a real detection)"
        ),

        severity="HIGH",

        signals=(
            IncidentSignal(
                "approach",
                0.9,
                "rehearsal",
            ),

            IncidentSignal(
                "contact",
                0.9,
                "rehearsal",
            ),

            IncidentSignal(
                "disruption",
                0.9,
                "rehearsal",
            ),
        ),

        evidence_types=(
            "approach",
            "contact",
            "disruption",
        ),

        frame=frame_number,
    )


# ============================================================
# MAIN
# ============================================================

def main(
    argv: list[str] | None = None,
) -> int:

    args = parse_args(argv)

    print()
    print("=" * 66)

    print(
        "  ResQTrack - AI accident detection "
        "and emergency response"
    )

    print("=" * 66)

    print(
        f"  camera     : {args.camera_id} "
        f"@ {args.lat:.5f}, {args.lon:.5f}"
    )

    print(
        f"  location   : {args.location_name}"
    )

    print(
        f"  source     : {args.source}"
    )

    print(
        f"  policy     : {args.sensitivity}"
    )

    print("=" * 66)
    print()

    # ========================================================
    # LOAD YOLO
    # ========================================================

    from ultralytics import YOLO

    yolo = YOLO(
        args.model
    )

    print(
        f"[yolo] {args.model} ready"
    )

    # ========================================================
    # LOAD TEMPORAL MODEL
    # ========================================================

    temporal_model = load_temporal_model(
        args.temporal_model
    )

    # ========================================================
    # OPEN VIDEO
    # ========================================================

    capture, fps, is_live = open_source(
        args.source
    )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print(
        f"[video] {fps:.1f} fps, "
        f"{total_frames if total_frames > 0 else 'live'} frames"
    )

    # ========================================================
    # INCIDENT ENGINE
    # ========================================================

    policy = EnginePolicy.for_sensitivity(
        args.sensitivity
    )

    frame_size = (
        int(
            capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        ),
        int(
            capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        ),
    )

    engine = IncidentEngine(
        policy=policy,
        fps=fps,
        frame_size=frame_size,
    )

    scale = ScaleEstimator(
        args.meters_per_pixel or None
    )

    registry = VehicleRegistry(
        scale,
        fps,
    )

    # ========================================================
    # ALERT CLIENT
    # ========================================================

    alerts = AlertClient(
        args.backend,
        enabled=not args.no_alerts,
    )

    alerts.start()

    if not args.no_alerts:

        reachable = (
            alerts.check_backend()
        )

        print(
            f"[backend] {args.backend} "
            + (
                "reachable"
                if reachable
                else
                "NOT reachable - alerts will spool "
                "to alerts_offline/"
            )
        )

    # ========================================================
    # STREAM
    # ========================================================

    stream: StreamServer | None = None

    if not args.no_stream:

        stream = StreamServer(
            port=args.stream_port
        )

        url = stream.start()

        if url:

            print(
                f"[stream] annotated feed at {url}"
            )

    # ========================================================
    # VIDEO RECORDING
    # ========================================================

    writer: cv2.VideoWriter | None = None

    # ========================================================
    # DISPLAY
    # ========================================================

    display = not args.headless

    if display:

        try:

            cv2.namedWindow(
                WINDOW_NAME,
                cv2.WINDOW_NORMAL,
            )

            cv2.resizeWindow(
                WINDOW_NAME,
                1360,
                766,
            )

        except cv2.error:

            print(
                "[display] no GUI available, "
                "continuing headless"
            )

            display = False

    # ========================================================
    # STATE
    # ========================================================

    reset_vehicle_history()

    SNAPSHOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame_number = 0

    frame_features: list[dict] = []

    previous_vehicles: dict | None = None

    previous_previous_vehicles: dict | None = None

    previous_actor_centers: dict[
        int,
        tuple[float, float],
    ] = {}

    ml_probability = 0.0

    peak_ml_probability = 0.0

    alert_active_until = 0.0

    last_alert_time = -1e9

    alert_evidence = None

    confirmed_count = 0

    fps_samples: deque[float] = deque(
        maxlen=30
    )

    show_cards = True

    show_roster = True

    fullscreen = False

    target_period = 0.0

    if args.max_fps > 0:

        target_period = (
            1.0 / args.max_fps
        )

    elif not is_live:

        target_period = (
            1.0 / fps
        )

    # ========================================================
    # DATABASE
    # ========================================================

    db_connection = None

    db_enabled = False

    db_status = "COMPLETED"

    run_id = (
        "RUN-"
        + datetime.now(
            timezone.utc
        ).strftime(
            "%Y%m%d%H%M%S"
        )
        + "-"
        + uuid4().hex[:8].upper()
    )

    try:

        # ----------------------------------------------------
        # INITIALISE DB
        # ----------------------------------------------------

        initialise_database()

        db_connection = connect()

        # ----------------------------------------------------
        # REGISTER CAMERA
        # ----------------------------------------------------

        register_camera(
            db_connection,

            camera_id=args.camera_id,

            name=(
                f"ResQTrack Camera "
                f"{args.camera_id}"
            ),

            location_name=args.location_name,

            latitude=args.lat,

            longitude=args.lon,

            source=args.source,

            status="ONLINE",
        )

        # ----------------------------------------------------
        # REGISTER MODEL RUN
        # ----------------------------------------------------

        start_model_run(
            db_connection,

            run_id=run_id,

            camera_id=args.camera_id,

            source=args.source,

            model_name="YOLO11 + Temporal ML",

            model_version=(
                f"{Path(args.model).name}"
                " + "
                f"{Path(args.temporal_model).name}"
            ),
        )

        # ----------------------------------------------------
        # START LOG
        # ----------------------------------------------------

        log_system(
            db_connection,

            level="INFO",

            component="detector",

            message=(
                f"Detector run started: "
                f"{run_id}"
            ),
        )

        db_connection.commit()

        db_enabled = True

        print(
            f"[database] run registered: "
            f"{run_id}"
        )

    except Exception as exc:

        print(
            "[database] logging disabled: "
            f"{type(exc).__name__}: {exc}"
        )

        if db_connection is not None:

            db_connection.close()

        db_connection = None

    # ========================================================
    # PROCESSING
    # ========================================================

    try:

        while True:

            loop_start = time.time()

            # ------------------------------------------------
            # READ FRAME
            # ------------------------------------------------

            ok, frame = (
                capture.read()
            )

            if not ok:

                if (
                    args.loop
                    and not is_live
                ):

                    capture.set(
                        cv2.CAP_PROP_POS_FRAMES,
                        0,
                    )

                    # New replay scene
                    engine.reset()

                    registry.profiles.clear()

                    frame_features.clear()

                    previous_vehicles = None

                    previous_previous_vehicles = None

                    previous_actor_centers.clear()

                    continue

                print(
                    "[video] source finished"
                )

                break

            frame_number += 1

            # ------------------------------------------------
            # PERIODIC MODEL-RUN UPDATE
            # ------------------------------------------------

            if (
                db_enabled
                and db_connection is not None
                and frame_number % 25 == 0
            ):

                try:

                    update_model_run(
                        db_connection,

                        run_id=run_id,

                        frames_processed=frame_number,

                        incidents_detected=confirmed_count,

                        alerts_sent=alerts.sent_count,
                    )

                    db_connection.commit()

                except Exception as exc:

                    print(
                        "[database] periodic update failed: "
                        f"{exc}"
                    )

                    db_enabled = False

            # =================================================
            # YOLO11 + BYTETRACK
            # =================================================

            results = yolo.track(
                frame,

                persist=True,

                tracker="bytetrack.yaml",

                conf=args.conf,

                classes=list(
                    TRACKED_CLASSES
                ),

                verbose=False,
            )

            if not results:

                continue

            result = results[0]

            detections: list[
                dict
            ] = []

            actors: dict[
                int,
                dict
            ] = {}

            vehicles: dict[
                int,
                dict
            ] = {}

            boxes = result.boxes

            # =================================================
            # PROCESS DETECTIONS
            # =================================================

            if (
                boxes is not None
                and boxes.id is not None
            ):

                xyxy = (
                    boxes.xyxy
                    .cpu()
                    .numpy()
                )

                ids = (
                    boxes.id
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                classes = (
                    boxes.cls
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                confidences = (
                    boxes.conf
                    .cpu()
                    .numpy()
                )

                for (
                    box,
                    actor_id,
                    class_id,
                    confidence,
                ) in zip(
                    xyxy,
                    ids,
                    classes,
                    confidences,
                ):

                    actor_id = int(
                        actor_id
                    )

                    class_id = int(
                        class_id
                    )

                    box = tuple(
                        float(value)
                        for value in box
                    )

                    kind = (
                        PERSON
                        if class_id
                        == PERSON_CLASS
                        else VEHICLE
                    )

                    name = class_name_for(
                        yolo,
                        class_id,
                    )

                    # -----------------------------------------
                    # CENTER
                    # -----------------------------------------

                    center = (
                        (
                            box[0]
                            + box[2]
                        ) / 2.0,

                        (
                            box[1]
                            + box[3]
                        ) / 2.0,
                    )

                    # -----------------------------------------
                    # MOVEMENT
                    # -----------------------------------------

                    previous_center = (
                        previous_actor_centers.get(
                            actor_id
                        )
                    )

                    speed_pixels_per_second = (
                        pixel_speed(
                            previous_center,
                            center,
                            fps,
                        )
                    )

                    heading = heading_label(
                        previous_center,
                        center,
                    )

                    # -----------------------------------------
                    # RECORD
                    # -----------------------------------------

                    record = {
                        "id": actor_id,

                        "box": box,

                        "center": center,

                        "class_id": class_id,

                        "class_name": name,

                        "kind": kind,

                        "confidence": float(
                            confidence
                        ),

                        "speed_pixels_per_second":
                            speed_pixels_per_second,

                        "heading":
                            heading,
                    }

                    detections.append(
                        record
                    )

                    actors[
                        actor_id
                    ] = record

                    # -----------------------------------------
                    # VEHICLE
                    # -----------------------------------------

                    if kind == VEHICLE:

                        vehicles[
                            actor_id
                        ] = {
                            "center": center,
                            "box": box,
                        }

                        update_vehicle(
                            actor_id,
                            center,
                        )

                        # =====================================
                        # DATABASE VEHICLE OBSERVATION
                        # =====================================

                        if (
                            db_enabled
                            and db_connection
                            is not None
                            and (
                                frame_number
                                % max(
                                    1,
                                    args.db_sample_every,
                                )
                                == 0
                            )
                        ):

                            try:

                                save_vehicle_observation(
                                    db_connection,

                                    camera_id=(
                                        args.camera_id
                                    ),

                                    tracker_id=(
                                        actor_id
                                    ),

                                    vehicle_class=(
                                        name
                                    ),

                                    confidence=(
                                        float(
                                            confidence
                                        )
                                    ),

                                    box=box,

                                    center=center,

                                    speed=(
                                        speed_pixels_per_second
                                    ),

                                    heading=heading,
                                )

                            except Exception as exc:

                                print(
                                    "[database] "
                                    "vehicle observation failed: "
                                    f"{exc}"
                                )

                                # Database failure must
                                # never stop AI detection.

                                db_enabled = False

            # =================================================
            # TEMPORAL MODEL
            # =================================================

            if temporal_model is not None:

                frame_features.append(
                    extract_frame_features(
                        vehicles,

                        previous_vehicles,

                        previous_previous_vehicles,
                    )
                )

                del frame_features[
                    :-WINDOW_FRAMES
                ]

                window_ready = (
                    len(frame_features)
                    >= WINDOW_FRAMES
                )

                stride_ready = (
                    frame_number
                    >= WINDOW_FRAMES

                    and (
                        frame_number
                        - WINDOW_FRAMES
                    )
                    % STRIDE_FRAMES
                    == 0
                )

                if (
                    window_ready
                    and stride_ready
                ):

                    aggregated = (
                        aggregate_window(
                            frame_features
                        )
                    )

                    if aggregated is not None:

                        model_input = (
                            pd.DataFrame(
                                [
                                    [
                                        aggregated[
                                            name
                                        ]
                                        for name
                                        in FEATURES
                                    ]
                                ],

                                columns=list(
                                    FEATURES
                                ),
                            )
                        )

                        ml_probability = float(
                            temporal_model
                            .predict_proba(
                                model_input
                            )[0][1]
                        )

                        peak_ml_probability = max(
                            peak_ml_probability,
                            ml_probability,
                        )

            # =================================================
            # PREVIOUS FRAME
            # =================================================

            previous_previous_vehicles = (
                previous_vehicles
            )

            previous_vehicles = (
                vehicles
            )

            previous_actor_centers = {
                actor_id: actor["center"]
                for actor_id, actor
                in actors.items()
            }

            # =================================================
            # INCIDENT ENGINE
            # =================================================

            evidence = engine.update(
                actors,

                frame_number=frame_number,

                timestamp=(
                    frame_number / fps
                ),

                ml_probability=(
                    ml_probability
                ),

                fps=fps
                
            )

            # =================================================
            # VEHICLE REGISTRY
            # =================================================

            profiles = registry.update(
                frame,

                detections,

                frame_number,

                involved_ids=set(
                    evidence.actor_ids
                )
                if evidence.status != "NORMAL"
                else set(),

                sample_colour=(
                    frame_number % 5
                    == 1
                ),
            )

            # =================================================
            # INCIDENT CONFIRMED
            # =================================================

            now = time.time()

            if (
                evidence.confirmed
                and (
                    now
                    - last_alert_time
                )
                >= args.alert_cooldown
            ):

                last_alert_time = now

                alert_active_until = (
                    now + 12.0
                )

                alert_evidence = (
                    evidence
                )

                confirmed_count += 1

                # ---------------------------------------------
                # CONSOLE
                # ---------------------------------------------

                _log_confirmation(
                    args,
                    evidence,
                    frame_number,
                )

                # ---------------------------------------------
                # SNAPSHOT
                # ---------------------------------------------

                annotated_for_snapshot = (
                    _render(
                        frame.copy(),

                        profiles,

                        evidence,

                        engine,

                        args,

                        scale,

                        True,

                        show_cards,

                        show_roster,

                        fps_samples,

                        alerts.state,
                    )
                )

                ok_encode, encoded = (
                    cv2.imencode(
                        ".jpg",
                        annotated_for_snapshot,
                    )
                )

                snapshot = (
                    encoded.tobytes()
                    if ok_encode
                    else None
                )

                if snapshot:

                    path = (
                        SNAPSHOT_DIR
                        / (
                            f"{args.camera_id}"
                            f"-frame"
                            f"{frame_number}.jpg"
                        )
                    )

                    path.write_bytes(
                        snapshot
                    )

                # ---------------------------------------------
                # ALERT
                # ---------------------------------------------

                alerts.send(
                    build_payload(
                        args,
                        evidence,
                        registry,
                        snapshot,
                    )
                )

                # ---------------------------------------------
                # DATABASE INCIDENT LOG
                # ---------------------------------------------

                if (
                    db_enabled
                    and db_connection
                    is not None
                ):

                    try:

                        log_system(
                            db_connection,

                            level="INFO",

                            component=(
                                "incident_engine"
                            ),

                            message=(
                                "ACCIDENT CONFIRMED "
                                f"frame={frame_number} "
                                f"kind={evidence.kind} "
                                f"confidence="
                                f"{evidence.confidence:.3f}"
                            ),
                        )

                        update_model_run(
                            db_connection,

                            run_id=run_id,

                            frames_processed=(
                                frame_number
                            ),

                            incidents_detected=(
                                confirmed_count
                            ),

                            alerts_sent=(
                                alerts.sent_count
                            ),
                        )

                        db_connection.commit()

                    except Exception as exc:

                        print(
                            "[database] incident logging failed: "
                            f"{exc}"
                        )

            # =================================================
            # ALERT DISPLAY
            # =================================================

            alert_active = (
                now
                < alert_active_until
            )

            if not alert_active:

                alert_evidence = None

            # =================================================
            # RENDER
            # =================================================

            fps_samples.append(
                1.0
                / max(
                    1e-3,
                    time.time()
                    - loop_start,
                )
            )

            annotated = _render(
                frame,

                profiles,

                (
                    alert_evidence
                    or evidence
                ),

                engine,

                args,

                scale,

                alert_active,

                show_cards,

                show_roster,

                fps_samples,

                alerts.state,
            )

            # =================================================
            # STREAM
            # =================================================

            if stream is not None:

                stream.publish(
                    annotated
                )

            # =================================================
            # RECORD
            # =================================================

            if args.record:

                if writer is None:

                    writer = cv2.VideoWriter(
                        args.record,

                        cv2.VideoWriter_fourcc(
                            *"mp4v"
                        ),

                        fps,

                        (
                            annotated.shape[1],
                            annotated.shape[0],
                        ),
                    )

                writer.write(
                    annotated
                )

            # =================================================
            # DISPLAY
            # =================================================

            if display:

                cv2.imshow(
                    WINDOW_NAME,
                    annotated,
                )

                if target_period:

                    wait = max(
                        1,
                        int(
                            (
                                target_period
                                - (
                                    time.time()
                                    - loop_start
                                )
                            )
                            * 1000
                        ),
                    )

                else:

                    wait = 1

                key = (
                    cv2.waitKey(
                        wait
                    )
                    & 0xFF
                )

                # ---------------------------------------------
                # Q / ESC
                # ---------------------------------------------

                if key in (
                    ord("q"),
                    27,
                ):

                    print(
                        "[input] "
                        "stopped by operator"
                    )

                    break

                # ---------------------------------------------
                # F = FULLSCREEN
                # ---------------------------------------------

                if key == ord("f"):

                    fullscreen = (
                        not fullscreen
                    )

                    cv2.setWindowProperty(
                        WINDOW_NAME,

                        cv2.WND_PROP_FULLSCREEN,

                        (
                            cv2.WINDOW_FULLSCREEN
                            if fullscreen
                            else cv2.WINDOW_NORMAL
                        ),
                    )

                # ---------------------------------------------
                # I = VEHICLE CARDS
                # ---------------------------------------------

                if key == ord("i"):

                    show_cards = (
                        not show_cards
                    )

                # ---------------------------------------------
                # R = RESPONDER ROSTER
                # ---------------------------------------------

                if key == ord("r"):

                    show_roster = (
                        not show_roster
                    )

                # ---------------------------------------------
                # T = REHEARSAL ALERT
                # ---------------------------------------------

                if key == ord("t"):

                    drill = (
                        rehearsal_evidence(
                            frame_number
                        )
                    )

                    alert_evidence = (
                        drill
                    )

                    alert_active_until = (
                        time.time()
                        + 12.0
                    )

                    last_alert_time = (
                        time.time()
                    )

                    confirmed_count += 1

                    print(
                        "[drill] rehearsal "
                        "alert dispatched"
                    )

                    alerts.send(
                        build_payload(
                            args,
                            drill,
                            registry,
                            None,
                        )
                    )

                    if (
                        db_enabled
                        and db_connection
                        is not None
                    ):

                        try:

                            log_system(
                                db_connection,

                                level="INFO",

                                component=(
                                    "detector"
                                ),

                                message=(
                                    "Rehearsal alert "
                                    "triggered at "
                                    f"frame {frame_number}"
                                ),
                            )

                            update_model_run(
                                db_connection,

                                run_id=run_id,

                                frames_processed=(
                                    frame_number
                                ),

                                incidents_detected=(
                                    confirmed_count
                                ),

                                alerts_sent=(
                                    alerts.sent_count
                                ),
                            )

                            db_connection.commit()

                        except Exception as exc:

                            print(
                                "[database] "
                                "rehearsal log failed: "
                                f"{exc}"
                            )

            # =================================================
            # FPS LIMIT
            # =================================================

            elif target_period:

                remaining = (
                    target_period
                    - (
                        time.time()
                        - loop_start
                    )
                )

                if remaining > 0:

                    time.sleep(
                        remaining
                    )

    # ========================================================
    # INTERRUPTED
    # ========================================================

    except KeyboardInterrupt:

        db_status = "INTERRUPTED"

        print(
            "\n[input] interrupted"
        )

    # ========================================================
    # FAILED
    # ========================================================

    except Exception as exc:

        db_status = "FAILED"

        if (
            db_connection is not None
            and db_enabled
        ):

            try:

                log_system(
                    db_connection,

                    level="ERROR",

                    component="detector",

                    message=(
                        "Detector run failed: "
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                )

                db_connection.commit()

            except Exception:
                pass

        raise

    # ========================================================
    # CLEANUP
    # ========================================================

    finally:

        # ----------------------------------------------------
        # FINISH DATABASE RUN
        # ----------------------------------------------------

        if db_connection is not None:

            try:

                finish_model_run(
                    db_connection,

                    run_id=run_id,

                    frames_processed=frame_number,

                    incidents_detected=confirmed_count,

                    alerts_sent=alerts.sent_count,

                    status=db_status,
                )

                log_system(
                    db_connection,

                    level="INFO",

                    component="detector",

                    message=(
                        f"Detector run finished: "
                        f"{run_id} "
                        f"status={db_status}"
                    ),
                )

                db_connection.commit()

            except Exception as exc:

                print(
                    "[database] finalization failed: "
                    f"{exc}"
                )

            finally:

                db_connection.close()

        # ----------------------------------------------------
        # RELEASE VIDEO
        # ----------------------------------------------------

        capture.release()

        # ----------------------------------------------------
        # RELEASE WRITER
        # ----------------------------------------------------

        if writer is not None:

            writer.release()

        # ----------------------------------------------------
        # CLOSE WINDOW
        # ----------------------------------------------------

        if display:

            cv2.destroyAllWindows()

        # ----------------------------------------------------
        # STOP STREAM
        # ----------------------------------------------------

        if stream is not None:

            stream.stop()

        # ----------------------------------------------------
        # STOP ALERT CLIENT
        # ----------------------------------------------------

        alerts.stop()

        # ----------------------------------------------------
        # RESET VEHICLE HISTORY
        # ----------------------------------------------------

        reset_vehicle_history()

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        print()

        print(
            "=" * 66
        )

        print(
            "  ResQTrack detector stopped"
        )

        print(
            f"  frames processed      : "
            f"{frame_number}"
        )

        print(
            f"  incidents confirmed   : "
            f"{confirmed_count}"
        )

        print(
            f"  alerts delivered      : "
            f"{alerts.sent_count}"
        )

        print(
            f"  alerts spooled offline: "
            f"{alerts.failed_count}"
        )

        print(
            f"  peak model context    : "
            f"{peak_ml_probability:.4f}"
        )

        print(
            f"  database run          : "
            f"{run_id}"
        )

        print(
            f"  database status       : "
            f"{db_status}"
        )

        print(
            "=" * 66
        )

    return 0


# ============================================================
# RENDER
# ============================================================

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

    values = (
        list(
            profiles.values()
        )
        if isinstance(
            profiles,
            dict,
        )
        else list(
            profiles
        )
    )

    overlay.draw_detection_boxes(
        frame,
        values,
    )

    reserved: list[
        tuple[
            int,
            int,
            int,
            int,
        ]
    ] = []

    status_rect = (
        overlay.draw_status(
            frame,

            evidence,

            alert_active,

            engine.scene_summary(),

            args.camera_id,

            dispatch_state,

            (
                sum(fps_samples)
                / len(fps_samples)
                if fps_samples
                else 0.0
            ),
        )
    )

    reserved.append(
        status_rect
    )

    if show_roster:

        roster_rect = (
            overlay.draw_roster(
                frame,

                values,

                scale.calibrated,

                scale.metres_per_pixel,

                reserved,
            )
        )

        if roster_rect:

            reserved.append(
                roster_rect
            )

    if show_cards:

        overlay.draw_vehicle_cards(
            frame,

            values,

            scale.calibrated,

            reserved,
        )

    if alert_active:

        overlay.draw_alert_border(
            frame,

            evidence,

            abs(
                (
                    time.time()
                    * 2
                )
                % 2
                - 1
            ),
        )

    overlay.draw_help(
        frame
    )

    return frame


# ============================================================
# CONSOLE CONFIRMATION
# ============================================================

def _log_confirmation(
    args,
    evidence,
    frame_number: int,
) -> None:

    print()

    print(
        "!" * 66
    )

    print(
        f"  ACCIDENT CONFIRMED  -  "
        f"{evidence.label}  "
        f"({evidence.severity})"
    )

    print(
        "!" * 66
    )

    print(
        f"  camera     : "
        f"{args.camera_id}  "
        f"({args.location_name})"
    )

    print(
        f"  location   : "
        f"{args.lat:.6f}, "
        f"{args.lon:.6f}"
    )

    print(
        f"  frame      : "
        f"{frame_number}"
    )

    print(
        f"  confidence : "
        f"{evidence.confidence:.2f}"
    )

    print(
        f"  evidence   : "
        f"{', '.join(evidence.evidence_types)}"
    )

    print(
        f"  reason     : "
        f"{evidence.reason}"
    )

    for item in evidence.involved:

        print(
            f"    - "
            f"{item.get('class_name', '?'):<12} "
            f"#{item.get('id')}  "
            f"heading "
            f"{item.get('heading', '-')}"
        )

    print(
        f"  ML context : "
        f"{evidence.ml_probability:.3f} "
        f"(corroboration only)"
    )

    print(
        "!" * 66
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    sys.exit(
        main()
    )