-- 2026-11-09: staff_current shows staff_code and gender from the current employment.
-- Same view as before; the two new columns are added at the end (required by create or replace view).

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
       s.consent,
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
         e_1.staff_code, e_1.gender
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
