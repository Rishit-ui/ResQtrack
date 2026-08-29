"""Offline evaluation of the ResQTrack incident policy.

Runs the full YOLO11 + ByteTrack + incident-engine pipeline over a folder tree
of labelled clips and reports, per clip, every CONFIRMED and REVIEW event with
the frame, the incident type and the evidence that produced it.

    python evaluate_detector.py                        # dataset/ + data/
    python evaluate_detector.py --sensitivity strict
    python evaluate_detector.py --videos data/test.mp4 --verbose

Expected layout (already in this repository):

    dataset/accident/*.mp4   clips that contain a crash
    dataset/normal/*.mp4     clips that must stay NORMAL

The summary reports detection rate on the accident clips and the false-alarm
rate on the normal clips, which is the number to quote in a review.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from vision.incident_engine import CONFIRMED, PERSON, REVIEW, VEHICLE, EnginePolicy, IncidentEngine
from vision.vehicle_profile import PERSON_CLASS, TRACKED_CLASSES


@dataclass
class ClipResult:
    path: Path
    label: str
    frames: int
    seconds: float
    confirmed: list[dict] = field(default_factory=list)
    reviews: list[dict] = field(default_factory=list)

    @property
    def detected(self) -> bool:
        return bool(self.confirmed)


def analyse_clip(
    yolo,
    path: Path,
    label: str,
    policy: EnginePolicy,
    confidence: float,
    verbose: bool,
) -> ClipResult:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"could not open {path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
    size = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    engine = IncidentEngine(policy=policy, fps=fps, frame_size=size)
    result = ClipResult(path=path, label=label, frames=0, seconds=0.0)
    started = time.time()
    seen_review_kinds: set[str] = set()

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        result.frames += 1

        detections = yolo.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=confidence,
            classes=list(TRACKED_CLASSES),
            verbose=False,
        )
        actors: dict[int, dict] = {}
        if detections:
            boxes = detections[0].boxes
            if boxes is not None and boxes.id is not None:
                for box, actor_id, class_id, score in zip(
                    boxes.xyxy.cpu().numpy(),
                    boxes.id.cpu().numpy().astype(int),
                    boxes.cls.cpu().numpy().astype(int),
                    boxes.conf.cpu().numpy(),
                ):
                    actors[int(actor_id)] = {
                        "box": tuple(float(value) for value in box),
                        "kind": PERSON if int(class_id) == PERSON_CLASS else VEHICLE,
                        "class_name": yolo.names.get(int(class_id), str(class_id)),
                        "confidence": float(score),
                    }

        evidence = engine.update(actors, frame_number=result.frames)
        if evidence.status == CONFIRMED:
            record = {
                "frame": result.frames,
                "second": round(result.frames / fps, 2),
                "kind": evidence.kind,
                "severity": evidence.severity,
                "confidence": evidence.confidence,
                "evidence": list(evidence.evidence_types),
                "reason": evidence.reason,
            }
            result.confirmed.append(record)
            if verbose:
                print(
                    f"    CONFIRMED @ {record['second']:6.2f}s  {record['kind']:<30}"
                    f" {record['confidence']:.2f}  [{', '.join(record['evidence'])}]"
                )
        elif evidence.status == REVIEW and evidence.kind not in seen_review_kinds:
            seen_review_kinds.add(evidence.kind)
            result.reviews.append(
                {
                    "frame": result.frames,
                    "second": round(result.frames / fps, 2),
                    "kind": evidence.kind,
                    "confidence": evidence.confidence,
                }
            )

    capture.release()
    result.seconds = time.time() - started
    return result


def collect_clips(args) -> list[tuple[Path, str]]:
    if args.videos:
        return [(Path(item), "unlabelled") for item in args.videos]
    clips: list[tuple[Path, str]] = []
    for folder, label in ((Path("dataset/accident"), "accident"), (Path("dataset/normal"), "normal")):
        if folder.is_dir():
            clips.extend((path, label) for path in sorted(folder.glob("*.mp4")))
    for path in sorted(Path("data").glob("*.mp4")) if Path("data").is_dir() else []:
        clips.append((path, "accident" if "accident" in path.stem else "unlabelled"))
    return clips


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the ResQTrack incident policy")
    parser.add_argument("--videos", nargs="*", help="explicit clips instead of the dataset tree")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--sensitivity", default="balanced", choices=("balanced", "high", "strict"))
    parser.add_argument("--csv", default="detector_evaluation.csv")
    parser.add_argument("--json", default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    from ultralytics import YOLO

    yolo = YOLO(args.model)
    policy = EnginePolicy.for_sensitivity(args.sensitivity)
    clips = collect_clips(args)
    if not clips:
        print("No clips found.")
        return 1

    print(f"\nResQTrack policy evaluation  ({args.sensitivity})")
    print("=" * 78)

    results: list[ClipResult] = []
    for path, label in clips:
        print(f"  {label:<12} {path}")
        result = analyse_clip(yolo, path, label, policy, args.conf, args.verbose)
        results.append(result)
        verdict = "DETECTED" if result.detected else "no incident"
        kinds = ", ".join(sorted({item["kind"] for item in result.confirmed})) or "-"
        print(
            f"      {verdict:<12} frames={result.frames:<5} events={len(result.confirmed):<3}"
            f" reviews={len(result.reviews):<3} kinds={kinds}"
        )

    accidents = [item for item in results if item.label == "accident"]
    normals = [item for item in results if item.label == "normal"]
    detected = sum(1 for item in accidents if item.detected)
    false_alarms = sum(1 for item in normals if item.detected)

    print("=" * 78)
    if accidents:
        print(f"  accident clips detected : {detected}/{len(accidents)}"
              f"  ({100.0 * detected / len(accidents):.0f}%)")
    if normals:
        print(f"  normal clips false-alarm: {false_alarms}/{len(normals)}"
              f"  ({100.0 * false_alarms / len(normals):.0f}%)")
    print("=" * 78 + "\n")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["clip", "label", "frames", "detected", "event_count", "first_second", "kinds", "reasons"]
            )
            for item in results:
                writer.writerow(
                    [
                        item.path,
                        item.label,
                        item.frames,
                        int(item.detected),
                        len(item.confirmed),
                        item.confirmed[0]["second"] if item.confirmed else "",
                        "|".join(sorted({event["kind"] for event in item.confirmed})),
                        " || ".join(event["reason"] for event in item.confirmed[:3]),
                    ]
                )
        print(f"  per-clip results written to {args.csv}")

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                [
                    {
                        "clip": str(item.path),
                        "label": item.label,
                        "frames": item.frames,
                        "confirmed": item.confirmed,
                        "reviews": item.reviews,
                    }
                    for item in results
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  detailed events written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
