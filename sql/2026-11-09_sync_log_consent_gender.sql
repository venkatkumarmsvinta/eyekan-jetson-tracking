-- 2026-11-09: sync log per device; consent + gender on staff_employment; staff_current reads consent from there.
-- Run after 2026-11-09_staff_code.sql (the view below includes staff_code). Safe to run again.

-- 1. Sync log: one row per sync run, per Jetson
create table if not exists public.sync_log (
  id          bigint generated always as identity primary key,
  created_at  timestamptz not null default now(),
  device      text not null,                 -- DEVICE_NAME from .env, or the hostname
  sync_type   text not null check (sync_type in ('photo', 'name', 'text', 'embedding')),
  status      text not null check (status in ('ok', 'failed')),
  added       int  not null default 0,
  updated     int  not null default 0,
  removed     int  not null default 0,
  message     text
);
create index if not exists sync_log_device_time on public.sync_log (device, created_at desc);
alter table public.sync_log enable row level security;
drop policy if exists "devices insert sync log" on public.sync_log;
create policy "devices insert sync log" on public.sync_log for insert to authenticated with check (true);
drop policy if exists "users read sync log" on public.sync_log;
create policy "users read sync log" on public.sync_log for select to authenticated using (true);

-- 2. Consent and gender on staff_employment
alter table public.staff_employment
  add column if not exists gender      text check (gender in ('male', 'female', 'other', 'not_stated')),
  add column if not exists consent     boolean not null default false,
  add column if not exists consent_at  timestamptz;   -- when consent was given; null = not recorded

-- copy today's consent from staff (staff.consent stays until the web app writes to staff_employment)
update public.staff_employment e set consent = true
from public.staff s where s.id = e.staff_id and s.consent and not e.consent;

-- 3. staff_current reads consent from the current employment
create or replace view public.staff_current with (security_invoker = true) as
select s.id,
       s.seq,
       s.ref,
       s.org_id,
       s.emp_id,
       s.first_name,
       s.last_name,
       s.avatar,
       s.active,
       coalesce(e.consent, false) as consent,
       s.created_at,
       s.updated_at,
       e.id as employment_id,
       e.employment_type,
       e.site_id,
       e.department,
       e.floor,
       e.designation,
       e.start_date as contract_start,
       e.end_date as contract_end,
       coalesce(ph.paths, '{}'::text[]) as photo_paths,
       ph.paths[1] as primary_photo_path,
       coalesce(cardinality(ph.paths), 0) as photo_count,
       e.staff_code,
       e.gender
from public.staff s
left join lateral (
  select e_1.id, e_1.org_id, e_1.staff_id, e_1.employment_type, e_1.site_id, e_1.department, e_1.floor,
         e_1.designation, e_1.start_date, e_1.end_date, e_1.created_at, e_1.updated_at,
         e_1.staff_code, e_1.gender, e_1.consent
  from public.staff_employment e_1
  where e_1.staff_id = s.id and (e_1.start_date is null or e_1.start_date <= current_date)
  order by (e_1.end_date is null or e_1.end_date >= current_date) desc, e_1.start_date desc nulls last
  limit 1
) e on true
left join lateral (
  select array_agg(p.storage_path order by p.is_primary desc, p.created_at) as paths
  from public.staff_photos p
  where p.staff_id = s.id
) ph on true;
