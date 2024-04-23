from typing import List

class Singleton(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class LabelEncoder(metaclass=Singleton):
    def __init__(self) -> None:
        self.mapping = {}
        
    def transform(self, label: str) -> int:
        if label not in self.mapping:
            self.mapping[label] = len(self.mapping)
        return self.mapping[label]
    
    def transform_list(self, labels: List[str]):
        return [self.transform(label) for label in labels]
    
if __name__ == "__main__":
    le = LabelEncoder()
    print(le.transform("a"))
    print(le.transform("b"))
    print(le.transform("a"))
    print(le.transform_list(["a", "b", "a"]))
    le2 = LabelEncoder()
    print(le2.mapping.keys())
    