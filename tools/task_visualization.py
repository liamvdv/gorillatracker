import time
from typing import Tuple

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeRemainingColumn
from rich.text import Text
from sqlalchemy.orm.session import Session
from sqlalchemy.sql import func
from sqlalchemy.sql.expression import select

from gorillatracker.ssl_pipeline.dataset import GorillaDatasetKISZ
from gorillatracker.ssl_pipeline.models import Task, TaskStatus, TaskType, Video

Progress_Task = Tuple[TaskType, str]  # TaskType and TaskSubType


class TaskVisualizer:
    # viszalizer
    console: Console = Console()
    layout: Layout = Layout()
    progress: Progress
    tasks: dict[TaskType, TaskID] = {}
    sub_tasks: dict[Progress_Task, TaskID] = {}

    # db
    session: Session
    version: str = "2024-04-18"

    def __init__(self, session: Session) -> None:
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main"),  # for generar information  # for progress bars
        )
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            TimeRemainingColumn(),
            BarColumn(bar_width=None),
            TextColumn("{task.completed}/{task.total}"),
            console=self.console,
            expand=True,
        )
        self.session = session
        self._initialize()

    def _initialize(self) -> None:
        self._update_header()
        self._initialize_tasks()
        self.layout["main"].update(self.progress)

    def _initialize_tasks(self) -> None:
        get_tasks = (
            select(Task.task_type, Task.task_subtype)
            .join(Video, Task.video_id == Video.video_id)
            .where(Video.version == self.version)
            .distinct()
        )
        tasks = self.session.execute(get_tasks).all()
        for task_type, task_subtype in tasks:
            if task_type not in self.tasks:
                self.tasks[task_type] = self.progress.add_task(f"[red][bold]{task_type.value}")
            self.sub_tasks[(task_type, task_subtype)] = self.progress.add_task(
                f"[green]{task_type.value} {task_subtype}"
            )

    def update_loop(self) -> None:
        with Live(self.layout, refresh_per_second=2, console=self.console):
            while not self.progress.finished:
                self._update_header()
                self._update_tasks_from_db()
                time.sleep(0.5)

    def _update_header(self) -> None:
        start_time = self.session.execute(
            select(func.min(Task.updated_at))
            .join(Video, Task.video_id == Video.video_id)
            .where(Video.version == self.version)
        ).scalar_one()
        header_content = Text(
            f"start time: {start_time.strftime('%Y-%m-%d %H:%M')} | version: {self.version}", justify="center"
        )
        self.layout["header"].update(header_content)

    def _update_tasks_from_db(self) -> None:
        for task_type, task_subtype in self.sub_tasks.keys():
            self._update_subtask_from_db(task_type, task_subtype)
        for task_type in self.tasks.keys():
            self._update_task(task_type)

    def _update_subtask_from_db(self, task_type: TaskType, task_subtype: str) -> None:
        task_query = (
            select(func.count())
            .select_from(Task)
            .join(Video, Task.video_id == Video.video_id)
            .where(Task.task_type == task_type, Task.task_subtype == task_subtype, Video.version == self.version)
        )
        finished_task_query = task_query.where(Task.status == TaskStatus.COMPLETED)
        task_count = self.session.execute(task_query).scalar_one()
        finished_task_count = self.session.execute(finished_task_query).scalar_one()
        self.progress.update(self.sub_tasks[(task_type, task_subtype)], completed=finished_task_count, total=task_count)

    def _update_task(self, task_type: TaskType) -> None:
        completed = 0
        total = 0
        sub_tasks = [task_subtype for task_type_, task_subtype in self.sub_tasks.keys() if task_type_ == task_type]
        for sub_task in sub_tasks:
            sub_task_progress = self.progress.tasks[self.sub_tasks[(task_type, sub_task)]]
            completed += int(sub_task_progress.completed)
            assert sub_task_progress.total is not None, "make mypy happy"
            total += int(sub_task_progress.total)
        self.progress.update(self.tasks[task_type], completed=completed, total=total)


if __name__ == "__main__":
    visualizer = TaskVisualizer(Session(GorillaDatasetKISZ(GorillaDatasetKISZ.DB_URI).engine))
    visualizer.update_loop()
