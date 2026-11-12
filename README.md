# ResQTrack

**AI accident detection and end-to-end emergency response platform — SIH 2026**

A traffic surveillance system watches the road. When it sees an incident it confirms the collision through physical evidence and temporal ML, pushes real-time WebSocket alerts to the nearest ambulances, navigates crews directly to the scene, discovers specialized trauma facilities ranked by live traffic, and logs every transition into an immutable audit trail.

```
 CAMERA ──► YOLO11 + ByteTrack ──► incident engine ──► CONFIRMED
                                                          │
                                              real-time WebSocket
                                                          │
                         ┌────────────────────────────────┼─────────────────┐
                         ▼                                ▼                 ▼
                 control room dashboard          responder mobile app      SQLite
                 (live map + camera feed)        (accept / navigate)    (full history)
                                                          │
                                    accept ─► route to scene ─► arrival geofence
                                           ─► nearby hospitals ranked by road time
                                           ─► least-traffic route to the chosen one
                                           ─► case closed
```

---

## Three Integrated Role Experiences

ResQTrack provides three unified portals sharing one secure backend, role-based JWT authentication, and database:

1. **Citizen Portal (`/user/dashboard`)**:
   - Public account registration (`/register`) and login (`/login`).
   - Incident submission with GPS location acquisition, crash category selection, and photo attachments.
   - Private report tracking with status badges (`SUBMITTED`, `APPROVED`, `ESCALATED`, `REJECTED`).
   - Strict data isolation: Citizens can only inspect their own reports.
   - Profile management and password reset lifecycle (`/forgot-password`, `/reset-password`).

2. **Control Room & Emergency Command Centre (`/admin/dashboard`)**:
   - Overview KPI telemetry (active incidents, dispatched units, hospital handovers, mean response time).
   - Live AI Detector stream HUD (port 8001).
   - Active Incidents command table with snapshot inspection and evidence breakdown.
   - Full-screen tactical interactive map (Leaflet) with real-time incident markers, responder units, and trauma centers.
   - Citizen report review queue with 1-click **Escalate & Dispatch** creating linked active incidents.
   - Fleet dispatch control & manual standing-down actions.
   - User administration (role status toggling, staff provisioning).
   - Immutable security and operational audit trail.
   - Real-time system health diagnostics (`/api/system/health`).

3. **Ambulance & Responder Unit (`/responder`)**:
   - Mobile-first crew interface with tactile emergency buttons.
   - Persistent WebSocket alert chimes synthesized via the Web Audio API.
   - Turn-by-turn routing to accident scene.
   - Automated 75m geofencing or manual scene arrival verification.
   - Dynamic trauma facility ranking by real road driving duration (OSRM + live traffic).
   - One-click navigation and patient hospital handover completion.

---

## Default Seed Credentials

All accounts are pre-seeded in the database for instant evaluation:

| Role | Email | Password | Access Route |
|---|---|---|---|
| **Admin** | `admin@resqtrack.gov` | `Admin@ResQTrack2026!` | `/admin/dashboard` |
| **Responder** | `responder@resqtrack.gov` | `Responder@2026!` | `/responder` |
| **Citizen** | `citizen@resqtrack.org` | `Citizen@2026!` | `/user/dashboard` |

*Quick-fill demo buttons are provided on the login page for rapid 1-click access.*

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build frontend single-page application
cd frontend && npm install && npm run build && cd ..

# 3. Seed demo units, hospital facilities, and default accounts
python seed_demo.py

# 4. Start backend server
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Then navigate to:

| What | Where |
|---|---|
| Landing Page | <http://localhost:8000/> |
| Login & Demo Portal | <http://localhost:8000/login> |
| Citizen Registration | <http://localhost:8000/register> |
| Control Room Command Centre | <http://localhost:8000/admin/dashboard> |
| Responder Mobile App | <http://localhost:8000/responder> |
| Live Vision Detector Feed | <http://localhost:8001/stream> |
| OpenAPI / Swagger Docs | <http://localhost:8000/docs> |

To run the AI computer-vision accident detector:

```bash
python main.py --source data/test.mp4 --loop                    # video file
python main.py --source 0                                       # webcam
python main.py --source rtsp://user:pass@camera/stream          # live CCTV
```

**Demo tip:** press **T** in the detector window to fire a rehearsal alert at
any moment. It runs the whole dispatch chain without waiting for the video to
reach its crash.

### Opening the responder app on a real phone

Browsers only hand out GPS over `https://` or `localhost`. Over plain
`http://192.168.x.x` the phone will refuse, and the app falls back to a demo
position near the camera — fine for a laptop, not for a live navigation demo.

For real GPS, serve the backend over TLS on your LAN:

```bash
# once, on the laptop running the backend
mkcert -install
mkcert 192.168.1.42            # use your machine's actual LAN IP
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 \
       --ssl-certfile ./192.168.1.42.pem --ssl-keyfile ./192.168.1.42-key.pem
```

Install the mkcert root certificate on the phone, then open
`https://192.168.1.42:8000/responder`. The app upgrades its WebSocket to `wss://`
automatically.

> Certificates and keys are gitignored. If you generated a pair earlier and
> committed it, treat that key as compromised and regenerate it — a private key
> in a public repository is readable by anyone.

---

## What the detector shows

Every tracked road user gets a card drawn next to it with the YOLO11 class, the
tracker ID, detection confidence, estimated speed in km/h, heading, dominant
colour and motion state (`MOVING` / `SLOW` / `BRAKING` / `STOPPED` /
`INCIDENT`). A roster panel lists everything on screen, and the incident panel
shows the verdict together with the evidence chips that produced it.

**About the km/h figure.** A single camera has no depth, so pixels cannot become
metres without a reference. Rather than invent a number, ResQTrack estimates the
scale from the objects themselves — a car is about 1.8 m wide, a bus 2.55 m — and
takes a robust median of `real_size / observed_box_width` across every confident
detection. The panel reports the resulting `m/px`. For a surveyed camera, pass
the real value with `--meters-per-pixel` and the estimate is bypassed.

Keys: `F` full screen · `I` vehicle cards · `R` roster · `T` rehearsal alert ·
`Q` quit.

---

## How an accident is confirmed

The old policy judged one frame at a time and knew three situations. It was both
blind (a rollover, a pile-up, or a person collapsing were all "NORMAL") and
jumpy — one noisy frame could dispatch an ambulance.

The engine in `vision/incident_engine.py` works in two layers.

### Layer 1 — evidence detectors

Each frame, independent detectors emit *findings*, every one carrying an
evidence type:

| Evidence | Meaning |
|---|---|
| `approach` | the actors were converging beforehand |
| `contact` | their boxes actually met |
| `disruption` | a trajectory broke — hard braking, a violent turn, a jolt |
| `pose_change` | a box flipped shape: a rollover, or a person going down |
| `immobility` | something that was moving fast is now stopped in the road |
| `departure` | an actor left the contact point at speed |
| `multi_actor` | several road users were disrupted together |
| `crowd` | bystanders converged on a disrupted vehicle |
| `vanished` | a track died right after a violent event *(supporting only)* |
| `model_context` | the trained temporal model finds the window anomalous |

Every motion quantity is normalised by the actor's own bounding-box diagonal, so
one set of thresholds works at any resolution and any distance from the camera.

### Layer 2 — evidence accumulation

Findings merge into hypotheses that persist across frames. A hypothesis is
**CONFIRMED** only when it has:

1. an **impact** pattern — contact *and* disruption, or a pose change, or a
   violent disruption that ends with the vehicle stopped; **and**
2. an **aftermath** — immobility, a body down, or a party leaving the scene —
   unless the impact was violent and unambiguous enough to stand alone; **and**
3. support across several frames.

That second requirement is the heart of it. *A near miss has approach and
contact but the road goes back to normal; after a real crash it does not.*

Two refinements matter in dense traffic:

- **A stopped vehicle only counts as a wreck when the traffic around it is still
  flowing.** If everything is stopped, that is congestion, not a crash.
- **A track's first frames are ignored.** A new track starts at zero speed, so
  its second frame always looks like a violent jolt — and in dense traffic the
  tracker creates identities constantly. This was the single largest source of
  false alarms.

### Incident types covered

vehicle-to-vehicle collision (rear-end, head-on, side-swipe, T-bone) · multi-vehicle
pile-up · vehicle-to-pedestrian collision · pedestrian knocked down or collapsed ·
vehicle rollover · single-vehicle loss of control · fixed-object impact ·
possible hit-and-run · post-crash immobilisation. Near misses and pedestrians
exposed in live traffic are reported as **REVIEW** and never dispatch.

### The trained model's role

`resqtrack_final_model.pkl` still runs, and its probability is shown on screen —
but it participates only as `model_context`. It can be the *second* corroborating
signal, never the first, and a supporting signal can never push a hypothesis over
the "violent impact confirms alone" line. A model bias therefore cannot dispatch
an ambulance on its own; the physical evidence has to be there.

---

## Measured performance

`evaluate_detector.py` runs the full pipeline over the labelled clips in this
repository and reports detections against false alarms.

```bash
python evaluate_detector.py                    # balanced preset
python evaluate_detector.py --sensitivity high --verbose
```

On the 12 clips in `dataset/` and `data/`:

| Preset | Accident clips detected | Normal clips with a false alarm |
|---|---|---|
| `strict` | fewest detections, fewest false alarms | — |
| `balanced` *(default)* | **4 / 7 (57%)** | **1 / 5 (20%)** |
| `high` | **5 / 7 (71%)** | **2 / 5 (40%)** |

Choose with `--sensitivity`. Use `high` where a human reviews alerts before
dispatch, `strict` for unattended dispatch where a false ambulance is expensive.

**Be straight about this in the presentation.** The misses are a night-time
clip where the crash is small and far from the camera, and two low-speed
courtyard incidents. Twelve clips is a small sample; the numbers show the method
works and is honestly measured, not that it is production-validated. Per-clip
results and the reasoning behind every event land in `detector_evaluation.csv`
and `detector_events.json`.

---

## The response flow

| Step | What happens | Where |
|---|---|---|
| 1 | Detector confirms an accident and POSTs it with evidence, involved road users and an annotated snapshot | `vision/alert_client.py` |
| 2 | Backend picks responders by road distance, unit type and severity, and pushes an alert over WebSocket | `backend/services/dispatch.py` |
| 3 | The responder's phone takes over the screen with a siren, vibration, the scene image and the evidence | `backend/static/responder.html` |
| 4 | **Accept** assigns the case; every other crew is stood down automatically | `POST /api/incidents/{id}/accept` |
| 5 | Live GPS streams to the server; the map draws the route with turn-by-turn guidance | `POST /api/responders/{id}/location` |
| 6 | Arrival is detected automatically inside a 75 m geofence | same endpoint |
| 7 | Nearby hospitals are found and ranked by **road** travel time, A&E first | `GET /api/incidents/{id}/hospitals` |
| 8 | Choosing one returns the **least-traffic** route of several alternatives | `POST /api/incidents/{id}/hospital` |
| 9 | The case is closed and the crew released | `POST /api/incidents/{id}/close` |

Every step writes to `incident_events`, which the dashboard replays as a
timeline. Nothing polls: the responder and the dashboard both hold a WebSocket,
so a confirmed accident reaches a phone in the time one JSON frame takes to
cross the network.

### Traffic and maps

- **Routes** come from OSRM, requested *with alternatives*. Each alternative is
  scored and the one with the lowest predicted travel time wins.
- **Traffic** comes from TomTom when `TOMTOM_API_KEY` is set. Without a key,
  a documented congestion model is used (time of day, the road classes the route
  actually uses, junction density). The API always reports `traffic_source`, so
  nothing on screen claims to be live data when it is not.
- **Hospitals** come from Overpass (OpenStreetMap), falling back to Nominatim,
  then to the local cache seeded by `seed_demo.py`.

**If the venue's internet fails**, routes fall back to a clearly-labelled
straight-line estimate and hospitals come from the seeded cache — the demo still
runs end to end. Map *tiles* need internet; the turn-by-turn list and all ETAs
do not.

---

## Project layout

```
main.py                     live detector: YOLO11 → policy → dispatch
run_resqtrack.py            starts backend + detector together
evaluate_detector.py        measures detection vs false alarms on the clips
seed_demo.py                demo crews + offline hospital data

vision/
  incident_engine.py        the incident policy (evidence + accumulation)
  kinematics.py             scale-invariant motion and pair geometry
  vehicle_profile.py        per-vehicle detail, colour, km/h calibration
  overlay.py                the control-room HUD
  alert_client.py           background dispatch with retry + offline spool
  stream_server.py          MJPEG feed the dashboard embeds

backend/
  app.py                    FastAPI: REST + WebSocket hub
  database.py               schema, lifecycle state machine, audit trail
  realtime.py               role-targeted event fan-out
  services/
    dispatch.py             which unit to send, and why
    routing_service.py      OSRM + alternatives + traffic + offline fallback
    hospital_service.py     Overpass → Nominatim → local cache
    traffic.py              live (TomTom) or modelled congestion
    geo.py                  distances, bearings, bounding boxes
  static/
    dashboard.html          control room
    responder.html          ambulance crew app

tests/                      31 tests: policy behaviour + backend lifecycle
dataset/, data/             labelled clips used by evaluate_detector.py
```

The original training and validation scripts (`train_temporal_*.py`,
`extract_temporal_features_*.py`, `validate_*.py`) are unchanged, and
`event_engine.py`, `hospital_service.py` and `routing_service.py` remain as
shims so they keep importing.

---

## Tests

```bash
python -m unittest discover -s tests -v
```

31 tests. The negative ones matter most — an emergency system that cries wolf on
parked cars is worse than none:

- parked and overlapping vehicles never confirm
- a whole queue stopping at a red light is not a pile-up
- a newly appearing track is not an impact
- a car leaving through the frame edge is not a hit-and-run
- the model's probability alone never confirms
- a near miss stays REVIEW

and the positive ones: rear-end collision with aftermath, pedestrian struck,
person collapsed, and one crash producing exactly one alert.

---

## Configuration

| Variable | Purpose |
|---|---|
| `TOMTOM_API_KEY` | enables live traffic; otherwise the modelled provider is used |
| `OSRM_BASE_URL` | point at your own OSRM instance |
| `RESQTRACK_DB` | database path (tests use a temporary file) |
| `RESQTRACK_SNAPSHOTS` | where scene images are stored |

Detector flags: `--source --model --conf --sensitivity --camera-id --lat --lon
--location-name --backend --no-alerts --stream-port --headless --loop --record
--meters-per-pixel --max-fps`.

---

## Deployment boundary

Read this before claiming more than the system does.

A monocular camera cannot see every crash, and this detector is deliberately
conservative: with no motion evidence it stays silent rather than guessing. The
numbers above come from twelve clips — enough to show the method is sound, not
enough to certify it. Before any real dispatch:

- train and validate on footage from the cameras you will actually deploy on,
  including parked traffic, stop-and-go queues, motorcycles, night, rain and
  occlusion;
- calibrate the `confirm` and `review` thresholds on a held-out, camera-specific
  set;
- keep a human in the loop, and name an owner accountable for both false
  positives and false negatives.

The `strict` preset and the REVIEW tier exist for exactly this: they let a
control room see near-misses and borderline events without an ambulance being
sent for them.
