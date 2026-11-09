"""Supabase: read the staff list and their face photos (read-only).

Credentials come from .env in the repository folder (copy .env.example). SUPABASE_URL and
SUPABASE_PUBLISHABLE_KEY are required. Without EDGE_ACCOUNT_EMAIL/PASSWORD the calls are anonymous:
the connection works, but row-level security returns no staff.
Check the connection:  python -m tracking.db
"""
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

from .config import STAFF_PHOTO_BUCKET, STAFF_PHOTO_DIR

load_dotenv()  # reads .env from the current folder; variables already set in the shell win


class Supabase:
    def __init__(self):
        try:
            self.url = os.environ["SUPABASE_URL"].rstrip("/")
            self.key = os.environ["SUPABASE_PUBLISHABLE_KEY"]
        except KeyError as e:
            raise SystemExit("Set {} in .env (copy .env.example)".format(e.args[0]))
        self.email = os.environ.get("EDGE_ACCOUNT_EMAIL")      # optional for now
        self.password = os.environ.get("EDGE_ACCOUNT_PASSWORD")
        self.http = httpx.Client(base_url=self.url, timeout=10)
        self.token, self.expires = None, 0.0

    def headers(self):
        if not (self.email and self.password):  # anonymous: the publishable key only
            return {"apikey": self.key, "Authorization": "Bearer " + self.key}
        if time.time() > self.expires - 60:  # sign in (again) as the device account
            r = self.http.post("/auth/v1/token", params={"grant_type": "password"}, headers={"apikey": self.key},
                               json={"email": self.email, "password": self.password})
            if r.status_code != 200:
                raise SystemExit("Supabase sign-in failed ({}): {}".format(r.status_code, r.text[:200]))
            body = r.json()
            self.token, self.expires = body["access_token"], time.time() + float(body.get("expires_in", 3600))
        return {"apikey": self.key, "Authorization": "Bearer " + self.token}

    def staff(self):
        """Active staff who gave consent: [{id, emp_id, first_name, last_name, photo_paths}, ...]."""
        r = self.http.get("/rest/v1/staff_current", headers=self.headers(), params={
            "select": "id,emp_id,first_name,last_name,photo_paths",
            "active": "eq.true", "consent": "eq.true", "order": "emp_id"})
        r.raise_for_status()
        return r.json()

    def sync_photos(self, staff):
        """Make STAFF_PHOTO_DIR match `staff`: download new photos, delete the photos (and folders) of people
        no longer in the list (left, inactive, consent withdrawn). Returns (downloaded, removed)."""
        root = Path(STAFF_PHOTO_DIR)
        keep = {s["emp_id"]: {Path(p).name for p in s["photo_paths"]} for s in staff}
        added = 0
        for s in staff:
            for path in s["photo_paths"]:
                dest = root / s["emp_id"] / Path(path).name
                if dest.exists():
                    continue
                r = self.http.get("/storage/v1/object/authenticated/{}/{}".format(STAFF_PHOTO_BUCKET, path),
                                  headers=self.headers())
                r.raise_for_status()
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(r.content)
                added += 1
        removed = 0
        for folder in (root.iterdir() if root.exists() else []):
            if not folder.is_dir():
                continue
            names = keep.get(folder.name)  # None: this person is no longer in the list
            for f in folder.iterdir():
                if names is None or f.name not in names:
                    f.unlink()
                    removed += 1
            if names is None:
                folder.rmdir()
        return added, removed


def sync():
    """Refresh the local staff photos from Supabase. Called at tracker start and by the daily cron job."""
    try:
        db = Supabase()
        if not db.email:
            print("Staff sync skipped: no EDGE_ACCOUNT_EMAIL in .env (anonymous sees no staff)")
            return
        staff = db.staff()
        added, removed = db.sync_photos(staff)
        print("Staff sync: {} staff, {} photos downloaded, {} removed -> {}".format(
            len(staff), added, removed, STAFF_PHOTO_DIR))
    except (httpx.HTTPError, SystemExit) as e:
        print("Staff sync failed ({}): using the photos already on this device".format(e))


if __name__ == "__main__":
    sync()  # python -m tracking.db  (by hand, or daily from cron)
