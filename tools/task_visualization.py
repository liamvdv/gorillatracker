import time
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


class task_visualizer():
    #viszalizer
    console:Console = Console()
    layout:Layout = Layout()
    progress:Progress
    
    #task progress
    tracking_task:TaskID
    predicting_task:TaskID
    visualizing_task:TaskID
    
    #db
    session:Session
    version:str = "2024-04-09"
    
    def __init__(self, session:Session):
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main")
        )
        self.progress = Progress(TextColumn("[progress.description]{task.description}"),
                        TimeRemainingColumn(),
                        BarColumn(bar_width=None),
                        TextColumn("{task.completed}/{task.total}"),
                        console=self.console,
                        expand=True)
        self.tracking_task = self.progress.add_task("[green]tracking...")
        self.predicting_task = self.progress.add_task("[green]predicting...")
        self.visualizing_task = self.progress.add_task("[green]visiualizing...")
        self.layout["main"].update(self.progress)
        self.session = session
        self.initialize_progress()
        
    def update_loop(self):
        with Live(self.layout, refresh_per_second=10, console=self.console):
            while not self.progress.finished:
                self.update_tasks_from_db()
                time.sleep(0.1)                
        
    def initialize_progress(self):
        self.update_tasks_from_db()
    
    def update_tasks_from_db(self):
        self.update_tracking_progress()
        self.update_predicting_progress()
        self.update_visualizing_progress()

    def update_tracking_progress(self):
        tracking_tasks_query = select(func.count()).select_from(Task).join(Video, Task.video_id == Video.video_id).where(Task.task_type == TaskType.TRACK, Video.version == self.version)
        finished_tracking_tasks_query = tracking_tasks_query.where(Task.status == TaskStatus.COMPLETED)
        tasks_count = self.session.execute(tracking_tasks_query).scalar()
        finished_tasks_count = self.session.execute(finished_tracking_tasks_query).scalar()
        self.progress.update(self.tracking_task, completed=finished_tasks_count, total=tasks_count)
        
    def update_predicting_progress(self):
        predicting_tasks_query = select(func.count()).select_from(Task).join(Video, Task.video_id == Video.video_id).where(Task.task_type == TaskType.PREDICT, Video.version == self.version)
        finished_predicting_tasks_query = predicting_tasks_query.where(Task.status == TaskStatus.COMPLETED)
        tasks_count = self.session.execute(predicting_tasks_query).scalar()
        finished_tasks_count = self.session.execute(finished_predicting_tasks_query).scalar()
        self.progress.update(self.predicting_task, completed=finished_tasks_count, total=tasks_count)
        
    def update_visualizing_progress(self):
        visualizing_tasks_query = select(func.count()).select_from(Task).join(Video, Task.video_id == Video.video_id).where(Task.task_type == TaskType.VISUALIZE, Video.version == self.version)
        finished_visualizing_tasks_query = visualizing_tasks_query.where(Task.status == TaskStatus.COMPLETED)
        tasks_count = self.session.execute(visualizing_tasks_query).scalar()
        finished_tasks_count = self.session.execute(finished_visualizing_tasks_query).scalar()
        self.progress.update(self.visualizing_task, completed=finished_tasks_count, total=tasks_count)

        
def main():
    console = Console()
    layout = Layout()

    # Create top panel for text stats
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main")
    )

    # Prepare the progress display
    progress = Progress(TextColumn("[progress.description]{task.description}"),
                        BarColumn(bar_width=None),
                        TextColumn("{task.completed}/{task.total}"),
                        console=console,
                        expand=True)

    task1 = progress.add_task("[red]Tracking...", total=100)
    task2 = progress.add_task("[green]Predicting...", total=200)
    task3 = progress.add_task("[blue]Visiualizing...", total=300)

    layout["main"].update(progress)

    # Use Live to manage the layout and updates smoothly
    with Live(layout, refresh_per_second=10, console=console):
        while not progress.finished:
            # Update tasks
            progress.update(task1, completed= 20)
            progress.update(task2, advance=2)
            progress.update(task3, advance=3)

            # Update the header with dynamic information
            current_time = time.strftime("%X")
            layout["header"].update(Text(f"Time: {current_time}", justify="center"))

            # Dynamically adjust the total if needed
            if progress.tasks[task1].completed > 50:
                progress.update(task1, total=150)
            if progress.tasks[task2].completed > 100:
                progress.update(task2, total=250)

            time.sleep(0.1)

if __name__ == "__main__":
    #main()
    visualizer = task_visualizer(Session(GorillaDataset("sqlite:///test.db").engine))
    visualizer.update_loop()
