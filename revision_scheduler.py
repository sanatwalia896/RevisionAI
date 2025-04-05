# revision_scheduler.py
import json
import datetime
from pathlib import Path

SCHEDULE_FILE = Path("revision_schedule.json")
REVISION_INTERVAL_DAYS = 3  # Remind if not revised in last 3 days


def load_schedule():
    if not SCHEDULE_FILE.exists():
        return []
    with open(SCHEDULE_FILE, "r") as f:
        return json.load(f)


def save_schedule(schedule):
    with open(SCHEDULE_FILE, "w") as f:
        json.dump(schedule, f, indent=2)


def check_due_revisions():
    schedule = load_schedule()
    now = datetime.datetime.now()

    due_pages = []
    for entry in schedule:
        last_revised = datetime.datetime.fromisoformat(entry["last_revised"])
        days_since = (now - last_revised).days
        if days_since >= REVISION_INTERVAL_DAYS:
            due_pages.append((entry["page_title"], days_since))

    if due_pages:
        print("\n🔔 Revision Reminders:")
        for title, days in due_pages:
            print(f"- {title}: Last revised {days} day(s) ago")
    else:
        print("✅ All pages are up-to-date with revisions!")


def mark_page_revised(page_title):
    schedule = load_schedule()
    now = datetime.datetime.now().isoformat()
    found = False
    for entry in schedule:
        if entry["page_title"] == page_title:
            entry["last_revised"] = now
            found = True
            break
    if not found:
        schedule.append({"page_title": page_title, "last_revised": now})
    save_schedule(schedule)


def ensure_pages_in_schedule(pages):
    schedule = load_schedule()
    existing_titles = {entry["page_title"] for entry in schedule}
    now = datetime.datetime.now().isoformat()

    for page in pages:
        if page["title"] not in existing_titles:
            schedule.append({"page_title": page["title"], "last_revised": now})

    save_schedule(schedule)


if __name__ == "__main__":
    check_due_revisions()
