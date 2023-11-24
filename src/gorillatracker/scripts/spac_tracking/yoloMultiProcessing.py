import multiprocessing
from generateBoundingBoxes import bodyModel, faceModel, convertLabelsToJson, joinLabels, createLabels
import os

class SingletonMeta(type):
    """
    This is a thread-safe implementation of Singleton. # TODO UNTESTED
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if multiprocessing.current_process().pid not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[multiprocessing.current_process().pid] = instance
        return cls._instances[multiprocessing.current_process().pid]

class Singleton(metaclass=SingletonMeta):
    def __init__(self):
        self.value = None

    def get_yolo(self, value):
        if self.value is None:
            self.value = (bodyModel, faceModel)
        return self.value

def worker_function(i):
    singleton = Singleton()
    body, face = singleton.get_yolo(i) 
    

if __name__ == "__main__":
    if not os.path.exists("./mTrack"):
        os.mkdir("./mTrack")
        os.mkdir("./mTrack/output")
    if os.path.exists("./mTrack/tmp"):
        os.system("rm -rf ./mTrack/tmp")
    os.mkdir("./mTrack/tmp")
    pool = multiprocessing.Pool(3)
    pool.map(worker_function, range(10))
    pool.close()
    pool.join()