"""Upload face photos for one staff member: each photo goes to the staff-photos bucket as
{org_id}/{staff_id}/{n}.jpg and gets a staff_photos row (the first photo of a person is primary).
Admin tool for this Mac only: needs SUPABASE_SECRET_KEY in .env. Never put that key on a Jetson.
    python tools/upload_staff_photos.py DTS-EMP-002 photos/A_*.jpg --consent
"""
import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

BUCKET = "staff-photos"   # same as STAFF_PHOTO_BUCKET in tracking/config.py


def main():
    ap = argparse.ArgumentParser(description="Upload staff face photos to Supabase")
    ap.add_argument("emp_id", help="staff emp_id, e.g. DTS-EMP-002")
    ap.add_argument("photos", nargs="+", help="JPEG files")
    ap.add_argument("--consent", action="store_true", help="also set consent = true (only if the person agreed)")
    args = ap.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")  # the repository's .env, from any folder
    db = httpx.Client(base_url=os.environ["SUPABASE_URL"].rstrip("/"), timeout=30,
                      headers={"apikey": os.environ["SUPABASE_SECRET_KEY"]})

    r = db.get("/rest/v1/staff", params={"select": "id,org_id,first_name", "emp_id": "eq." + args.emp_id})
    r.raise_for_status()
    if len(r.json()) != 1:
        raise SystemExit("emp_id {} matches {} staff rows, expected 1".format(args.emp_id, len(r.json())))
    s = r.json()[0]
    r = db.get("/rest/v1/staff_photos", params={"select": "id", "staff_id": "eq." + s["id"]})
    r.raise_for_status()
    first = len(r.json()) + 1   # continue numbering after photos already stored

    rows = []
    for n, photo in enumerate(args.photos, first):
        path = "{}/{}/{}.jpg".format(s["org_id"], s["id"], n)
        r = db.post("/storage/v1/object/{}/{}".format(BUCKET, path), content=Path(photo).read_bytes(),
                    headers={"content-type": "image/jpeg"})
        r.raise_for_status()
        rows.append({"org_id": s["org_id"], "staff_id": s["id"], "storage_path": path, "is_primary": n == 1})
        print("uploaded {} -> {}".format(photo, path))
    r = db.post("/rest/v1/staff_photos", json=rows, headers={"Prefer": "return=minimal"})
    r.raise_for_status()
    if args.consent:
        r = db.patch("/rest/v1/staff_employment", params={"staff_id": "eq." + s["id"], "end_date": "is.null"},
                     json={"consent": True, "consent_at": datetime.now(timezone.utc).isoformat()},
                     headers={"Prefer": "return=minimal"})
        r.raise_for_status()
    print("{} ({}): {} photos added{}".format(s["first_name"], args.emp_id, len(rows),
                                              ", consent set" if args.consent else ""))


if __name__ == "__main__":
    main()
