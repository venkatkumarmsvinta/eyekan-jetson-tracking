-- 2026-11-09: staff_code (e.g. DTS-STF-001) on staff_employment, numbered per org.
-- staff_photos, staff_attendance and staff_entries point to it, next to staff_id (which stays).

-- 1. staff_employment.staff_code, filled for existing rows (per org, in staff order)
alter table public.staff_employment add column if not exists staff_code text;

with numbered as (
  select e.id, o.code, row_number() over (partition by e.org_id order by s.seq, e.created_at) as n
  from public.staff_employment e
  join public.staff s on s.id = e.staff_id
  join public.organizations o on o.id = e.org_id
)
update public.staff_employment e
set staff_code = n.code || '-STF-' || lpad(n.n::text, greatest(3, length(n.n::text)), '0')
from numbered n where n.id = e.id and e.staff_code is null;

alter table public.staff_employment alter column staff_code set not null;
alter table public.staff_employment add constraint staff_employment_staff_code_key unique (staff_code);

-- 2. New employment rows get the next number of their org
create or replace function public.set_staff_code() returns trigger
language plpgsql security definer set search_path to '' as $$
declare
  org_code text;
  n int;
begin
  if new.staff_code is not null then
    return new;
  end if;
  perform pg_advisory_xact_lock(hashtext('staff_code:' || new.org_id));  -- two inserts at once can't get the same number
  select code into org_code from public.organizations where id = new.org_id;
  if org_code is null then
    raise exception 'Organization % has no code', new.org_id;
  end if;
  select coalesce(max(substring(staff_code from '-STF-(\d+)$')::int), 0) + 1 into n
  from public.staff_employment where org_id = new.org_id;
  new.staff_code := org_code || '-STF-' || lpad(n::text, greatest(3, length(n::text)), '0');
  return new;
end $$;

drop trigger if exists staff_employment_set_code on public.staff_employment;
create trigger staff_employment_set_code before insert on public.staff_employment
for each row execute function public.set_staff_code();

-- 3. The current staff_code of a person (open employment first, then the latest)
create or replace function public.current_staff_code(p_staff_id uuid) returns text
language sql stable security definer set search_path to '' as $$
  select staff_code from public.staff_employment
  where staff_id = p_staff_id
  order by (end_date is null) desc, start_date desc nulls last, created_at desc
  limit 1
$$;

create or replace function public.fill_staff_code() returns trigger
language plpgsql set search_path to '' as $$
begin
  if new.staff_code is null and new.staff_id is not null then
    new.staff_code := public.current_staff_code(new.staff_id);
  end if;
  return new;
end $$;

-- 4. staff_code + foreign key in the tables that point to a staff member
alter table public.staff_photos add column if not exists staff_code text
  references public.staff_employment (staff_code) on update cascade on delete set null;
alter table public.staff_attendance add column if not exists staff_code text
  references public.staff_employment (staff_code) on update cascade on delete set null;
alter table public.staff_entries add column if not exists staff_code text
  references public.staff_employment (staff_code) on update cascade on delete set null;

create index if not exists staff_photos_staff_code_idx on public.staff_photos (staff_code);
create index if not exists staff_attendance_staff_code_idx on public.staff_attendance (staff_code);
create index if not exists staff_entries_staff_code_idx on public.staff_entries (staff_code);

update public.staff_photos set staff_code = public.current_staff_code(staff_id)
where staff_code is null and staff_id is not null;
update public.staff_attendance set staff_code = public.current_staff_code(staff_id)
where staff_code is null and staff_id is not null;
update public.staff_entries set staff_code = public.current_staff_code(staff_id)
where staff_code is null and staff_id is not null;

drop trigger if exists staff_photos_fill_code on public.staff_photos;
create trigger staff_photos_fill_code before insert on public.staff_photos
for each row execute function public.fill_staff_code();
drop trigger if exists staff_attendance_fill_code on public.staff_attendance;
create trigger staff_attendance_fill_code before insert on public.staff_attendance
for each row execute function public.fill_staff_code();
drop trigger if exists staff_entries_fill_code on public.staff_entries;
create trigger staff_entries_fill_code before insert on public.staff_entries
for each row execute function public.fill_staff_code();
