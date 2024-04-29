from typing import Any, Dict, List, Tuple, Union

mapping: Dict[str, int] = {}
class LabelEncoder:
    def encode(label: str) -> int:
        if label not in mapping:
            mapping[label] = len(mapping)
        return mapping[label]

    def encode_list(labels: List[str]) -> List[int]:
        return [LabelEncoder.encode(label) for label in labels]
    
    def decode(index: int) -> str:
        for label, idx in mapping.items():
            if idx == index:
                return label
    def decode_list(indices: List[int]) -> List[str]:
        return [LabelEncoder.decode(index) for index in indices]


class LinearSequenceEncoder:
    def __init__(self) -> None:
        self.mapping: Dict[int, int] = {}

    def encode(self, label: int) -> int:
        if label not in self.mapping:
            self.mapping[label] = len(self.mapping)
        return self.mapping[label]

    def encode_list(self, labels: Union[List[int], Tuple[int]]) -> List[int]:
        return [self.encode(label) for label in labels]
    
    def decode(self, index: int) -> int:
        for label, idx in self.mapping.items():
            if idx == index:
                return label
    def decode_list(self, indices: Union[List[int], Tuple[int]]) -> List[int]:
        return [self.decode(index) for index in indices]


if __name__ == "__main__":
    le = LabelEncoder
    print(le.encode("a"))
    print(le.encode("b"))
    print(le.encode("a"))
    print(le.encode_list(["a", "b", "a"]))
    le2 = LabelEncoder
    print(mapping.keys())
    print(le2.decode_list([0, 1, 0]))
    
