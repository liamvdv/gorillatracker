"""
- A run is always kept if 'keep', 'proof' or 'baseline' is in the tags. 
- All runs that match a filter condition are scheduled for deletion. Deletion is scheduled 7 days after the run was started.
- All runs before 2024-04-07 are ignored. One may lower this date to sort through 
  older runs and either fix the naming or delete unnecessary runs.

Usage:
    # Generate Plan for deletion
    python3 scripts/clean_wandb.py process --output deletion_targets.json

    # Apply deletion plan. You will get a preview and will need to confirm your actions twice.
    python3 scripts/clean_wandb.py apply deletion_targets.json
    
"""

import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Generator, List, Optional

import typer
import wandb
from pydantic import BaseModel, field_serializer

Run = Any

ENTITY = "gorillas"

ENFORCE_AFTER = "2024-04-07"


def keep(run) -> bool:
    return run.created_at <= ENFORCE_AFTER or "keep" in run.tags or "proof" in run.tags or "baseline" in run.tags

# Initialize the WandB API
if "WANDB_API_KEY" in os.environ:
    wandb.login(key=os.environ["WANDB_API_KEY"])
api = wandb.Api()


class DeletionTarget(BaseModel):
    at: datetime
    path: str  # entity/project/run
    reasons: List[str]

    name: str  # for human reference
    url: str  # for human reference

    def execute(self, dry: bool) -> None:
        print(f"Deleting {self.name} because {', '.join(self.reasons)} ({self.url})")
        run = api.run(self.path)
        if dry:
            print(run.url, "would be deleted!")
        else:
            run.delete()
            print(run, "deleted!")

    @field_serializer("at")
    def serialize_dt(self, at: datetime, _info) -> str:
        return at.isoformat()


class DeletionSchedule(BaseModel):
    targets: List[DeletionTarget]

    def execute(self, dry: bool = True) -> None:
        for target in self.targets:
            if target.at < datetime.now(timezone.utc):
                target.execute(dry)


def is_older_than_(run: Run, days: int) -> bool:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    ran = datetime.fromisoformat(run.created_at).replace(tzinfo=timezone.utc)
    return ran < cutoff


def clear_failed(run: Run) -> Optional[str]:
    if run.state in ["crashed", "failed", "cancelled"] and is_older_than_(run, 3):
        return "failed"


def clear_untraceable_names(run: Run) -> Optional[str]:
    # {Issue Number}-{purpose}
    pattern = r"^\d+-.+"
    match = re.search(pattern, run.name)
    if not match:
        return "name_untraceable"

# https://docs.wandb.ai/ref/python/run
run_filters = [clear_failed, clear_untraceable_names]

class WandbCleaner:
    def __init__(self, rules, entity=ENTITY):
        self.rules = rules
        self.entity = entity
        self.delete_after = timedelta(days=7)
        self.targets = []

    def iter_runs(self) -> Generator[Run, None, None]:
        for project in api.projects(entity=self.entity):
            print("Project:", project.name, project.url)
            for run in api.runs("/".join(project.path)):
                yield run

    def process(self) -> None:
        for run in self.iter_runs():
            self.process_run(run)

    def process_run(self, run) -> None:
        print("Processing run:", run.name, run.url)
        evaluations = [rule(run) for rule in self.rules]
        reasons = [r for r in evaluations if r is not None]
        if reasons and not keep(run):
            self.targets.append(
                DeletionTarget(
                    at=datetime.fromisoformat(run.created_at).replace(tzinfo=timezone.utc) + self.delete_after,
                    path=f"{self.entity}/{run.project}/{run.id}",
                    reasons=reasons,
                    name=run.name,
                    url=run.url,
                )
            )

    def save(self, outpath: str) -> None:
        schedule = DeletionSchedule(targets=self.targets)
        raw = json.dumps(schedule.model_dump(), indent=4)
        if outpath == "-":
            print(raw)
        else:
            Path(outpath).write_text(raw)


def DANGER_apply_deletion_schedule(filepath: str) -> None:
    with open(filepath, "rb") as f:
        schedule = DeletionSchedule.model_validate_json(f.read())
    schedule.execute(dry=True)
    if input("Are you sure you want to delete the all listed runs? (yes/N) ") != "yes":
        print("Aborted")
        return
    if input("Are you really really sure? We cannot revert this. (yes/N) ") != "yes":
        print("Aborted")
        return
    print("Deleting runs...")
    schedule.execute(dry=False)
    print("Done!")


app = typer.Typer()


@app.command()
def process(output: str = "-"):
    cleaner = WandbCleaner(run_filters)
    cleaner.process()
    cleaner.save(output)


@app.command()
def apply(filepath: str):
    DANGER_apply_deletion_schedule(filepath)


if __name__ == "__main__":
    app()
