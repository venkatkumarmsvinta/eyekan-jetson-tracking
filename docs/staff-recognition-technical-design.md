# Staff Recognition: Technical Design

| | |
|---|---|
| Component | `jetson-multi-tracking`, branch `multicamera-tracking` |
| Version | 0.1 (prototype) |
| Date | 2026-11-10 |
| Status | Implemented and tested on one video; not production-ready (see §13, §14) |

---

## 1. Purpose and scope

This document describes the staff recognition feature of the people tracker: how staff face photos are
distributed from Supabase to the edge devices, how they are turned into face embeddings, and how tracked people
in the camera video are matched against them to separate staff from customers.

In scope: staff photo synchronisation, face embedding, track-level matching, the related Supabase schema, and
the admin upload tool. Out of scope: the detection, pose, tracking and re-identification pipeline (described in
`README.md`), attendance logic, and the web application.

## 2. Background

The tracker (RF-DETR detection → ViTPose keypoints → BoT-SORT tracking → ReLink re-identification) produces one
track ID per person, but cannot tell whether a person is an employee or a customer. Store analytics (customer
counts, dwell times, staff coverage) need that distinction. Staff are known in advance, have consented to
recognition, and have reference photos in Supabase; customers are unknown and must remain anonymous.

## 3. Requirements

### 3.1 Functional

| ID | Requirement |
|---|---|
| F1 | Every device keeps a local copy of the photos of active, consenting staff of its organisation. |
| F2 | New and replaced photos are downloaded; photos of people who left, became inactive or withdrew consent are deleted. |
| F3 | Each device computes face embeddings for its local staff photos, only for new or changed photos. |
| F4 | Every person track is labelled with the matching staff member's `emp_id`, or left unlabelled (not staff). |
| F5 | People not in the staff list (customers) must not be labelled as staff (open-set recognition). |
| F6 | Each sync run is recorded centrally per device and per sync type. |
| F7 | Results are visible in the output video and in the per-frame CSV. |

### 3.2 Non-functional

| ID | Requirement |
|---|---|
| N1 | Runs on the edge device (NVIDIA Jetson) without a central inference server. |
| N2 | Tracking keeps running if Supabase, the face model or the embedding step is unavailable. |
| N3 | Devices use a read-only account; admin credentials never leave the admin machine. |
| N4 | Biometric data is processed under GDPR art. 9: explicit consent, removal on withdrawal, no customer biometrics stored. |
| N5 | Configuration in one place (`tracking/config.py`); no hard-coded thresholds in the modules. |

## 4. Architecture overview

```
┌──────────────────────────── Supabase ─────────────────────────────┐
│ staff ──< staff_employment (staff_code, gender, consent)          │
│   │                 │                                             │
│   └──< staff_photos ┘ (storage_path, staff_code)                  │
│ view staff_current (active, consent, photo_paths, staff_code)     │
│ bucket staff-photos/{org_id}/{staff_id}/{n}.jpg                   │
│ sync_log (device, sync_type, status, counts)                      │
└──────────▲─────────────────────────────┬──────────────────────────┘
           │ upload (admin, secret key)  │ read (device account)
┌──────────┴────────────┐     ┌──────────▼───────────────────────────────────────────┐
│ tools/                │     │ Edge device                                          │
│ upload_staff_photos.py│     │  tracking/db.py     sync: staff list, photos, log    │
└───────────────────────┘     │       │                                              │
                              │       ▼                                              │
                              │  photo/<emp_id>/*.jpg, photo/.versions.json          │
                              │       │                                              │
                              │  tracking/faces.py  embeddings → photo/embeddings.json│
                              │       │                                              │
                              │  tracking/identify.py  StaffMatcher                  │
                              │       ▲ frames, boxes, keypoints, track IDs          │
                              │  tracking/pipeline.py  detection → pose → tracking   │
                              │       │                                              │
                              │       ▼ output/<video>/rfdetr.mp4, rfdetr.csv        │
                              └──────────────────────────────────────────────────────┘
```

Two phases:

1. **Sync phase** (`python -m tracking.db`; at tracker start and daily by cron): staff list → photos →
   embeddings → `sync_log`.
2. **Tracking phase** (`python track.py <video>`): per frame, `StaffMatcher` collects face embeddings per track
   and labels each track.

## 5. Components

### 5.1 Sync: `tracking/db.py`

| Function | Responsibility |
|---|---|
| `Supabase.headers()` | Authentication. Device account (email/password → JWT, renewed 60 s before expiry); if none is configured and `SUPABASE_SECRET_KEY` is set, the secret key (admin machine only); otherwise anonymous (sees no staff). |
| `Supabase.staff()` | `GET /rest/v1/staff_current?active=eq.true&consent=eq.true` → id, emp_id, names, photo_paths. |
| `Supabase.remote_versions(folder)` | `POST /storage/v1/object/list/staff-photos` → `{storage path: updated_at}` per staff folder. |
| `Supabase.sync_photos(staff)` | Downloads new/changed photos, deletes removed ones, maintains `photo/.versions.json`. Returns (added, updated, removed). |
| `Supabase.log(...)` | Inserts one `sync_log` row. Best effort: a failure prints a warning and never hides the sync result. |
| `sync()` | Orchestrates: staff → photos → log → `sync_embeddings()`. Network and auth errors are caught; the device keeps its existing photos. |
| `sync_embeddings(db)` | Calls `faces.embed_photos()`, logs `sync_type = 'embedding'`. Any exception is caught (N2). |

### 5.2 Embedding: `tracking/faces.py`

- `FaceEmbedder`: wraps InsightFace `FaceAnalysis` with only the `detection` (SCRFD) and `recognition`
  (ArcFace R50) modules of the `buffalo_l` pack. ONNX Runtime providers: CUDA if available, else CPU.
  Input is scaled so its longer side is `FACE_DET_SIZE`; returns the 512-d, L2-normalised embedding of the most
  confident face and its detector score.
- `embed_photos()`: keeps `photo/embeddings.json` in step with `photo/*/*.jpg` (§7.3). The model is loaded
  lazily, only if at least one photo needs embedding.

### 5.3 Matching: `tracking/identify.py`

- `load_gallery()`: per `emp_id`, the mean of that person's photo embeddings, re-normalised (one centroid per
  staff member). Photos without a detected face are skipped.
- `StaffMatcher.update(frame_idx, frame, tid, box, kp_xy, kp_conf)`: rate-limited face sampling per track (§7.4).
- `StaffMatcher.label(tid)`: the decision rule (§7.5). Returns `(emp_id, score)` or `(None, best score)`.
- If no gallery exists, the face model is not loaded and every call is a no-op.

### 5.4 Pipeline integration: `tracking/pipeline.py`, `tracking/drawing.py`

- `StaffMatcher(fps)` is created next to `ReLink`.
- Per frame, after `ReLink` assigns final IDs and **before any drawing** (the frame must be clean), `update()`
  is called for every track.
- When writing each CSV row, `label(tid)` provides the `staff` and `staff_score` columns; `draw_person()` draws
  `STAFF <emp_id>` in green above the box.
- At the end of a run, one summary line per track with collected faces.

### 5.5 Admin tool: `tools/upload_staff_photos.py`

`upload_staff_photos.py <emp_id> <jpg...> [--consent]`: looks up the staff row, uploads each file to
`staff-photos/{org_id}/{staff_id}/{n}.jpg` (numbering continues after existing photos), inserts one
`staff_photos` row per file (the person's first photo is `is_primary`), and with `--consent` sets
`consent = true, consent_at = now()` on the open employment (`end_date is null`). Uses `SUPABASE_SECRET_KEY`.

## 6. Data model

### 6.1 Supabase tables (relevant columns)

| Table | Columns | Notes |
|---|---|---|
| `staff` | `id` (uuid, PK), `org_id`, `emp_id`, `first_name`, `last_name`, `active`, `consent`*, `ref` | The person. *`consent` is superseded by `staff_employment.consent`; kept until the web app writes there. |
| `staff_employment` | `id`, `org_id`, `staff_id` → staff, `staff_code` (unique, not null), `gender`, `consent`, `consent_at`, `start_date`, `end_date`, department, designation, floor | One row per employment period. |
| `staff_photos` | `id`, `org_id`, `staff_id` → staff (cascade), `staff_code` → staff_employment, `storage_path`, `is_primary`, `captured_at` | One row per photo. |
| `staff_attendance`, `staff_entries` | … `staff_id` → staff, `staff_code` → staff_employment | `staff_code` added alongside `staff_id`. |
| `sync_log` | `id`, `created_at`, `device`, `sync_type` ∈ {photo, name, text, embedding}, `status` ∈ {ok, failed}, `added`, `updated`, `removed`, `message` | RLS: authenticated insert and select. |

Foreign keys on `staff_code`: `on update cascade on delete set null`.

### 6.2 `staff_code`

Format `<organizations.code>-STF-<n>`, e.g. `DTS-STF-002`, numbered per organisation with at least 3 digits
(no truncation above 999).

- Existing rows were numbered by `staff.seq` within each org.
- New `staff_employment` rows: trigger `staff_employment_set_code` → `set_staff_code()`, which takes a
  per-org advisory lock and assigns max + 1. Numbers are not reused unless the highest one is deleted.
- `staff_photos`, `staff_attendance`, `staff_entries`: trigger `*_fill_code` → `fill_staff_code()` fills
  `staff_code` from `staff_id` via `current_staff_code()` (open employment first, then latest start date).

### 6.3 View `staff_current` (`security_invoker = true`)

One row per staff member joined with the current employment (`start_date` null or ≤ today; open or not yet
ended first) and the aggregated photos:

- `photo_paths` (ordered: primary first, then by `created_at`), `primary_photo_path`, `photo_count`
- `consent` from the current employment (`false` if none)
- `staff_code`, `gender` (appended as the last two columns)

### 6.4 Storage

Private bucket `staff-photos`, object path `{org_id}/{staff_id}/{n}.jpg`. Read: org members; write: admins
(storage policies). The object's `updated_at` changes on re-upload and drives change detection.

### 6.5 Local files (on each device; `photo/` is git-ignored)

| File | Content |
|---|---|
| `photo/<emp_id>/<n>.jpg` | Local copies; folder per `emp_id`, file name from the storage path. |
| `photo/.versions.json` | `{"<emp_id>/<file>": "<storage updated_at>"}`: the version of each local copy. |
| `photo/embeddings.json` | `{"model": "buffalo_l", "photos": {"<emp_id>/<file>": {"stamp": "<mtime_ns>-<size>", "score": <detector score>, "embedding": [512 floats] or null}}}` |

### 6.6 Tracker CSV

`output/<video>/rfdetr.csv` gains two columns at the end: `staff` (`emp_id` or empty) and `staff_score` (best
cosine similarity of the track so far, 3 decimals). The label of a track can change while faces are collected.

## 7. Algorithms

### 7.1 Photo change detection

For each listed photo of each staff member: download if the local file is missing, or if the stored version in
`.versions.json` differs from the storage object's `updated_at`. Then delete every local file not in the list,
and folders of people no longer listed.

### 7.2 Face embedding

ArcFace maps an aligned 112×112 face to a 512-d vector on the unit sphere. It is trained with an additive angular
margin loss, so the cosine similarity between two embeddings measures identity: about 1 for the same person,
about 0 for different people.

### 7.3 Incremental embedding

A photo is (re-)embedded when its key is new or its `stamp` (file mtime in ns + size) differs; removed photos
are dropped. A change of `FACE_MODEL` discards all embeddings, because embeddings from different models are not
comparable. A photo without a detected face is stored as `null`, so it is not retried every run.

### 7.4 Face sampling per track

For track `tid` in frame `f`, a face is attempted only if all of these hold:

1. fewer than `FACE_MAX_PER_TRACK` faces are collected for `tid`;
2. at least `round(FACE_EVERY_S × fps)` frames have passed since its last attempt;
3. ViTPose says `facing = front` and face confidence ≥ `FACE_MIN_CONF`;
4. the box is at least `FACE_MIN_BOX_H` px tall, and a head point (mean of the visible face keypoints) exists.

The crop is a square of side 0.4 × box height around the head point, scaled to `FACE_DET_SIZE`. Of the faces
detected in the crop, only those whose box **contains the head point** are kept, and the most confident is
used. This rejects the face of another person in the crop: in the test, a staff member turned away from the
camera otherwise received the face of the person behind her, because ViTPose reported "front" for a head seen
from behind.

### 7.5 Decision rule

With the collected embeddings `e_i` and detector scores `s_i` of a track:

```
v = Σ s_i · e_i / ‖Σ s_i · e_i‖               (quality-weighted mean)
score_k = v · c_k                            (cosine similarity with staff centroid c_k)
best, second = two highest score_k
staff = argmax_k  if best ≥ STAFF_THRESHOLD and best − second ≥ STAFF_MARGIN, else none
```

The margin condition prevents a label when two staff members are similarly close.

## 8. Configuration (`tracking/config.py`)

| Parameter | Value | Purpose |
|---|---|---|
| `STAFF_PHOTO_BUCKET` | `staff-photos` | Storage bucket |
| `STAFF_PHOTO_DIR` | `<project>/photo` | Local photo folder (resolved from the package, independent of the working directory) |
| `STAFF_EMBED_FILE` | `<project>/photo/embeddings.json` | Local embeddings |
| `FACE_MODEL` | `buffalo_l` | InsightFace model pack |
| `FACE_DET_SIZE` | 320 | Detector input size (px, longer side) |
| `FACE_EVERY_S` | 0.33 | Minimum interval between face attempts per track (s) |
| `FACE_MAX_PER_TRACK` | 30 | Faces collected per track |
| `FACE_MIN_CONF` | 0.7 | Minimum ViTPose face confidence |
| `FACE_MIN_BOX_H` | 250 | Minimum person height (px) |
| `STAFF_THRESHOLD` | 0.25 | Minimum cosine similarity for a staff label |
| `STAFF_MARGIN` | 0.10 | Minimum lead over the second-best staff member |

Environment (`.env`): `SUPABASE_URL`, `SUPABASE_PUBLISHABLE_KEY`, `EDGE_ACCOUNT_EMAIL`,
`EDGE_ACCOUNT_PASSWORD`, `DEVICE_NAME` (devices); `SUPABASE_SECRET_KEY`, `SUPABASE_DB_URL` (admin machine only).

## 9. Interfaces

| Command | Effect |
|---|---|
| `python -m tracking.db` | Sync staff photos and embeddings; writes 2 `sync_log` rows |
| `python track.py <video> [--no-sync] [--max-frames N] [--show]` | Sync (unless `--no-sync`), then track with staff matching |
| `python tools/upload_staff_photos.py <emp_id> <jpg...> [--consent]` | Admin: add staff photos |

Use the `venv-rfdetr` interpreter (`venv-rfdetr/bin/python`).

## 10. Security and privacy

- **Credentials:** devices authenticate with a per-device account, which can only read through row-level
  security (`is_member(org_id)`). The secret key bypasses RLS and sees all organisations; it is for the admin
  machine only and must never be deployed to a device. `.env` is git-ignored.
- **Consent:** only staff with `consent = true` on the current employment are synced. Withdrawing consent or
  deactivating removes the person's photos and embeddings from every device at the next sync (daily cron).
- **Customers:** customer face embeddings exist only in memory for the duration of a run and are never written
  to disk or the database. Note: the output video and CSV still contain all tracked people.
- **Local copies:** the development project is inside OneDrive, so `photo/` is synced to OneDrive.
  `.gitignore` does not prevent this.
- **Compliance:** a DPIA is expected to be required for biometric identification of staff.

## 11. Error handling

| Failure | Behaviour |
|---|---|
| Supabase unreachable, sign-in failed, HTTP error | Sync prints "using the photos already on this device"; `photo` sync logged as `failed` if possible; tracking continues. |
| `sync_log` write fails | Warning only; the sync result is unaffected. |
| Face model missing or broken | Embedding logged as `failed`; photos stay synced; tracking continues. |
| No `embeddings.json` | `StaffMatcher` is disabled (no model loaded); CSV `staff` column empty. |
| Photo without a detectable face | Stored as `null`; that photo is ignored in the gallery. |

## 12. Performance

Measured on the development Mac (PyTorch MPS for detection and pose, ONNX Runtime CPU for faces):

| Measurement | Result |
|---|---|
| Embedding 15 staff photos (first run, including one-time model download) | 35 s |
| Embedding with nothing changed | < 0.1 s (model not loaded) |
| Tracking with staff matching, camera1, first 2,000 frames | 206 s, 9.7 fps |

Face checks are bounded per track: at most 3 per second and 30 in total. On a Jetson, ONNX Runtime GPU
(`onnxruntime-gpu`) or TensorRT is expected to reduce the per-face cost substantially; this has not been
measured yet.

## 13. Evaluation

**Method.** Camera1 test video (store, 2026-10-27). Gallery: 5 face crops each of 3 people (A, C, D) cut from the
same video; 1 further person (B) as a customer. Offline evaluation excluded frames within ±1 s of the gallery
frames; the in-pipeline test did not.

**Results (in-pipeline run, first 2,000 frames):**

| Person | Track IDs | Faces | Result | Score |
|---|---|---|---|---|
| A (DTS-EMP-002) | 1, 5, 124 | 8 / 30 / 1 | staff ✓ | 0.59 / 0.84 / 0.80 |
| C (DTS-EMP-003) | 43, 52 | 30 / 1 | staff ✓ | 0.79 / 0.47 |
| D (DTS-EMP-013) | 57 | 13 | staff ✓ | 0.31 |
| B (customer) | 6, 20 | 30 / 3 | not staff ✓ | 0.23 / 0.11 |
| Wall poster (false detection) | 104 | 3 | not staff ✓ | 0.03 |

Offline evaluation (full video, gallery frames excluded): staff tracks 0.30–0.78, customer 0.09, poster ≤ 0.04.

**Interpretation.** All tracks were labelled correctly. Staff scores are optimistic because the gallery and the
probes come from the same recording (same day, clothing, lighting). The customer reached 0.23 against a
threshold of 0.25, so the margin is thin. Two labels were based on a single face.

## 14. Limitations and risks

| # | Limitation / risk | Mitigation |
|---|---|---|
| L1 | `buffalo_l` weights are licensed for non-commercial research only. | Commercial InsightFace licence, or another model, before production. |
| L2 | Small evaluation with gallery taken from the test video; threshold not validated. | Evaluate on other days and cameras; enrol clear photos. |
| L3 | Customer score close to the threshold (0.23 vs 0.25). | Raise the threshold (e.g. 0.28) and require a minimum number of faces per label (e.g. 3). |
| L4 | Single-face labels possible. | Minimum-faces rule (L3). |
| L5 | ViTPose "front" is unreliable for heads seen from behind. | Head-point containment check (§7.4); a dedicated head-pose estimate could replace it. |
| L6 | Track hand-overs (`apply_handovers`) merge IDs after the run; per-row labels reflect the ID at that time. | Re-label after hand-overs if needed. |
| L7 | With the secret key, the sync sees all organisations. | Admin machine only; add an org filter. |
| L8 | `current_staff_code()` and `staff_current` choose the current employment slightly differently (future start dates). | Align once people have multiple employments. |
| L9 | `sync_log` embedding message counts faceless photos of the current run only. | Report the total. |
| L10 | Local folders use `emp_id`, not `staff_code`. | Switch when `staff_code` is the primary human-readable key. |
| L11 | Labels use real employee records for test persons (DTS-EMP-002, 003, 013). | Remove the test photos and consent flags after testing. |

## 15. Future work

1. Minimum-faces rule and threshold tuning (L3, L4).
2. Evaluation set from additional recordings; report true-accept and false-accept rates at several thresholds.
3. Jetson deployment: `onnxruntime-gpu`/TensorRT, timing, per-device accounts.
4. Central embeddings in Supabase (`staff_embeddings`, pgvector, model version) using `sync_type = 'embedding'`.
5. Cross-camera identity: share staff labels between cameras of a `CAMERA_GROUPS` group.
6. Licensing decision and possible model change (AdaFace for low-resolution faces, or a commercial SDK).

## 16. Change history

| Date | Commit | Change |
|---|---|---|
| 2026-11-09 | `04cb743` | Staff photo sync from Supabase |
| 2026-11-09 | `c579941` | Changed-photo detection, `sync_log` logging |
| 2026-11-09 | `cab5be7` | Upload tool; SQL: `sync_log`, `staff_code`, consent/gender on `staff_employment`, view |
| 2026-11-09 | `0aad4f3` | Device name in `sync_log`; photos in `photo/` |
| 2026-11-09 | `a4f6864` | Embeddings with `buffalo_l` after sync; secret-key fallback |
| 2026-11-10 | (this commit) | Track-level staff matching (`tracking/identify.py`); this document |

## Appendix A: database migrations

| File | Content |
|---|---|
| `sql/2026-11-09_staff_code.sql` | `staff_code` on `staff_employment`; triggers; `staff_code` FKs in photos, attendance, entries |
| `sql/2026-11-09_staff_current_code_gender.sql` | View: `staff_code`, `gender` |
| `sql/2026-11-09_sync_log_consent_gender.sql` | `sync_log`; `gender`, `consent`, `consent_at` on `staff_employment`; view reads consent from employment |

Apply in the order listed: the last file's view definition includes `staff_code`.
