from typing import Any, Dict, List, Tuple, Union


class Singleton(type):
    _instances: Dict[type, object] = {}

    # (cls would be of type Singleton but at this point Singleton is not defined)
    def __call__(cls, *args: List[Any], **kwargs: Dict[Any, Any]) -> object:  # type: ignore
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class LabelEncoder_Global(metaclass=Singleton):
    def __init__(self) -> None:
        self.mapping: Dict[str, int] = {}

    def transform(self, label: str) -> int:
        if label not in self.mapping:
            self.mapping[label] = len(self.mapping)
        return self.mapping[label]

    def transform_list(self, labels: List[str]) -> List[int]:
        return [self.transform(label) for label in labels]


class LabelEncoder_Local:
    def __init__(self) -> None:
        self.mapping: Dict[str, int] = {}

    def transform(self, label: str) -> int:
        if label not in self.mapping:
            self.mapping[label] = len(self.mapping)
        return self.mapping[label]

    def transform_list(self, labels: Union[List[str], Tuple[str]]) -> List[int]:
        return [self.transform(label) for label in labels]


if __name__ == "__main__":
    le = LabelEncoder_Global()
    print(le.transform("a"))
    print(le.transform("b"))
    print(le.transform("a"))
    print(le.transform_list(["a", "b", "a"]))
    le2 = LabelEncoder_Global()
    print(le2.mapping.keys())
    print(LabelEncoder_Global._instances)
