# ResQTrack

ResQTrack is a single-camera traffic-incident detector and emergency-response
prototype. It uses YOLO/ByteTrack for road-user tracking, a temporal model for
scene context, and an explicit incident-evidence policy for dispatch decisions.

## What changed in the live detector

`main.py` no longer confirms an accident from repeated model probabilities.
That was the source of the parked-vehicle false positive: a dense static scene
can look unusual to a model trained on traffic windows, but it is not proof of
an impact.

An **ACCIDENT CONFIRMED** alert now requires time-ordered physical evidence:

1. A road user was moving.
2. Actors approached a contact zone.
3. An abrupt trajectory disruption occurred after that interaction.

The policy supports vehicle-to-vehicle collisions and vehicle-to-pedestrian
collisions. It also detects a possible hit-and-run when a pedestrian vanishes
after a contact-zone interaction while the vehicle continues away. A close,
overlapping, or parked vehicle alone stays **NORMAL**. A dynamic near-miss is
shown as **REVIEW**, never as a confirmed accident.

The existing temporal-model probability is still visible as context, but is
explicitly not a confirmation trigger. This preserves useful anomaly context
without allowing a model bias to dispatch an emergency.

## Run the live view

Install the required packages in your Python environment, then run:

```bash
pip install ultralytics opencv-python numpy pandas joblib scikit-learn
python main.py
```

The app opens in full screen by default. Press `F` to toggle full screen,
`I` to hide/show actor labels, and `Q` or `Esc` to exit. Labels are rendered by
one layout routine rather than on top of the YOLO labels, so nearby vehicles
do not produce an unreadable stack of text.

## Verification

The policy has deterministic tests for parked/overlapping vehicles,
vehicle-to-vehicle impact, vehicle-to-pedestrian impact, hit-and-run behaviour,
and the legacy collision helper:

```bash
python -m unittest -v tests/test_incident_policy.py
```

## Important deployment boundary

No monocular camera system can reliably infer every crash from a single frame.
The current detector is deliberately conservative: it avoids an automatic
emergency confirmation if the required motion evidence is absent. Before
production dispatch, train and validate an action model on representative,
labelled footage for each deployed camera, including parked traffic, stop-and-
go queues, pedestrians, motorcycles, night/rain, and occlusion. Review and
confirmed thresholds should then be calibrated on a held-out camera-specific
set with a safety owner responsible for false positives and false negatives.
