import time
from datetime import datetime
from gorillatracker.ssl_pipeline.models import Task, TaskType, Video, TaskStatus 
from gorillatracker.ssl_pipeline.dataset import GorillaDataset

from rich.console import Console
from rich.layout import Layout
from rich.progress import Progress, BarColumn, TextColumn, TaskID, TimeRemainingColumn
from rich.text import Text
from rich.live import Live

from sqlalchemy.sql import func
from sqlalchemy.orm.session import Session
from sqlalchemy.sql.expression import select

from typing import Tuple

Progress_Task = Tuple[TaskType, str] # TaskType and TaskSubType

class task_visualizer():
    #viszalizer
    console:Console = Console()
    layout:Layout = Layout()
    progress:Progress
    
    #task progress
    tracking_task:TaskID
    predicting_task:TaskID
    visualizing_task:TaskID
    
    tasks:dict[Progress_Task, TaskID] = {}
    
    #db
    session:Session
    version:str = "2024-04-09"
    
    def __init__(self, session:Session):
        self.layout.split(
            Layout(name="header", size=3), # for generar information
            Layout(name="main") # for progress bars
        )
        self.progress = Progress(TextColumn("[progress.description]{task.description}"),
                        TimeRemainingColumn(),
                        BarColumn(bar_width=None),
                        TextColumn("{task.completed}/{task.total}"),
                        console=self.console,
                        expand=True)
        self.session = session
        self._initialize()
        
    def _initialize(self):
        self._update_header()
        self._initialize_tasks()
        self.layout["main"].update(self.progress)
        
    def _initialize_tasks(self):
        get_tasks = select(Task.task_type, Task.task_subtype).select_from(Task).join(Video, Task.video_id == Video.video_id).where(Video.version == self.version).distinct()
        tasks = self.session.execute(get_tasks).all()
        for task in tasks:
            print(task)
            if((task[0], "") not in self.tasks):
                self.tasks[(task[0], "")] = self.progress.add_task(f"[red]{task[0].value}")
            self.tasks[task] = self.progress.add_task(f"[green]{task[0].value} {task[1]}")
            
    def update_loop(self):
        with Live(self.layout, refresh_per_second=10, console=self.console):
            while not self.progress.finished:
                self._update_header()
                self._update_tasks_from_db()
                time.sleep(0.1)                
        
    def _update_header(self):
        startime:datetime = self.session.execute(select(func.min(Task.updated_at)).select_from(Task).join(Video, Task.video_id == Video.video_id).where(Video.version == self.version)).scalar()
        header_content = Text(f"startime: {startime.strftime('%Y-%m-%d %H:%M')} | version: {self.version}", justify="center")
        self.layout["header"].update(header_content)
    
    def _update_tasks_from_db(self):
        for(task_type, task_subtype) in self.tasks.keys():
            if task_subtype == "":
                continue
            self._update_subtask_from_db(task_type, task_subtype)
        self._update_tasks()
        
    def _update_subtask_from_db(self, task_type:TaskType, task_subtype:str):
        task_query = select(func.count()).select_from(Task).join(Video, Task.video_id == Video.video_id).where(Task.task_type == task_type, Task.task_subtype == task_subtype, Video.version == self.version)
        finished_task_query = task_query.where(Task.status == TaskStatus.COMPLETED)
        task_count = self.session.execute(task_query).scalar()
        finished_task_count = self.session.execute(finished_task_query).scalar()
        self.progress.update(self.tasks[(task_type, task_subtype)], completed=finished_task_count, total=task_count)
        
    def _update_tasks(self):
        tasks = {}
        for task_type, task_subtype in self.tasks.keys():
            if task_subtype == "":
                continue
            if task_type in tasks:
                completed = tasks[task_type][0] + self.progress.tasks[self.tasks[(task_type, task_subtype)]].completed
                total = tasks[task_type][1] + self.progress.tasks[self.tasks[(task_type, task_subtype)]].total
                tasks[task_type] = (completed, total)
            else:
                completed = self.progress.tasks[self.tasks[(task_type, task_subtype)]].completed
                total = self.progress.tasks[self.tasks[(task_type, task_subtype)]].total
                tasks[task_type] = (completed, total)
        for task_type, (completed, total) in tasks.items():
            self.progress.update(self.tasks[(task_type, "")], completed=completed, total=total)       

        
if __name__ == "__main__":
    visualizer = task_visualizer(Session(GorillaDataset("sqlite:///test.db").engine))
    visualizer.update_loop()