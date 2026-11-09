"""Supabase: read the staff list and their face photos, and write one sync_log row per sync.

Credentials come from .env in the repository folder (copy .env.example). SUPABASE_URL and
SUPABASE_PUBLISHABLE_KEY are required. Without EDGE_ACCOUNT_EMAIL/PASSWORD the calls are anonymous:
the connection works, but row-level security returns no staff.
Check the connection:  python -m tracking.db
"""
import json
import os
import socket
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

    def remote_versions(self, folder):
        """{storage path: updated_at} of the photos in one bucket folder; updated_at changes when a photo is re-uploaded."""
        r = self.http.post("/storage/v1/object/list/" + STAFF_PHOTO_BUCKET, headers=self.headers(),
                           json={"prefix": folder, "limit": 1000})
        r.raise_for_status()
        return {folder + "/" + o["name"]: o.get("updated_at") for o in r.json() if o.get("id")}  # id None = sub-folder

    def sync_photos(self, staff):
        """Make STAFF_PHOTO_DIR match `staff`: download new and changed photos, delete the photos (and folders)
        of people no longer in the list (left, inactive, consent withdrawn). Returns (added, updated, removed)."""
        root = Path(STAFF_PHOTO_DIR)
        version_file = root / ".versions.json"   # {emp_id/file: updated_at} of the local copies
        try:
            versions = json.loads(version_file.read_text())
        except (OSError, ValueError):
            versions = {}
        keep = {s["emp_id"]: {Path(p).name for p in s["photo_paths"]} for s in staff}
        added = updated = 0
        for s in staff:
            remote = {}
            for folder in {str(Path(p).parent) for p in s["photo_paths"]}:
                remote.update(self.remote_versions(folder))
            for path in s["photo_paths"]:
                dest = root / s["emp_id"] / Path(path).name
                key = "{}/{}".format(s["emp_id"], dest.name)
                version = remote.get(path)
                if dest.exists() and versions.get(key) == version:
                    continue
                r = self.http.get("/storage/v1/object/authenticated/{}/{}".format(STAFF_PHOTO_BUCKET, path),
                                  headers=self.headers())
                r.raise_for_status()
                existed = dest.exists()
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(r.content)
                versions[key] = version
                if existed:
                    updated += 1
                else:
                    added += 1
        removed = 0
        for folder in (root.iterdir() if root.exists() else []):
            if not folder.is_dir():
                continue
            names = keep.get(folder.name)  # None: this person is no longer in the list
            for f in folder.iterdir():
                if names is None or f.name not in names:
                    f.unlink()
                    versions.pop("{}/{}".format(folder.name, f.name), None)
                    removed += 1
            if names is None:
                folder.rmdir()
        if root.exists():
            version_file.write_text(json.dumps(versions, indent=1))
        return added, updated, removed

    def log(self, sync_type, status, added=0, updated=0, removed=0, message=None):
        """Write one row to sync_log (sync_type: photo / name / text / embedding). Best-effort: a failed
        log write only prints a warning, so it never hides the result of the sync itself."""
        try:
            r = self.http.post("/rest/v1/sync_log", headers={**self.headers(), "Prefer": "return=minimal"},
                               json={"device": socket.gethostname(), "sync_type": sync_type, "status": status,
                                     "added": added, "updated": updated, "removed": removed, "message": message})
            r.raise_for_status()
        except (httpx.HTTPError, SystemExit) as e:
            detail = e.response.text[:200] if isinstance(e, httpx.HTTPStatusError) else e
            print("Sync log not written ({})".format(detail))


def sync():
    """Refresh the local staff photos from Supabase. Called at tracker start and by the daily cron job."""
    db = None
    try:
        db = Supabase()
        if not db.email:
            print("Staff sync skipped: no EDGE_ACCOUNT_EMAIL in .env (anonymous sees no staff)")
            return
        staff = db.staff()
        added, updated, removed = db.sync_photos(staff)
        print("Staff sync: {} staff, {} new, {} updated, {} removed -> {}".format(
            len(staff), added, updated, removed, STAFF_PHOTO_DIR))
        db.log("photo", "ok", added=added, updated=updated, removed=removed, message="{} staff".format(len(staff)))
    except (httpx.HTTPError, SystemExit) as e:
        print("Staff sync failed ({}): using the photos already on this device".format(e))
        if db and db.email:
            db.log("photo", "failed", message=str(e)[:500])


if __name__ == "__main__":
    sync()  # python -m tracking.db  (by hand, or daily from cron)
