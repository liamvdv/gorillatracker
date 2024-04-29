from typing import Dict, List, Tuple, Union

mapping: Dict[str, int] = {}


class LabelEncoder:
    @staticmethod
    def encode(label: str) -> int:
        if label not in mapping:
            mapping[label] = len(mapping)
        return mapping[label]

    @staticmethod
    def encode_list(labels: List[str]) -> List[int]:
        return [LabelEncoder.encode(label) for label in labels]

    @staticmethod
    def decode(index: int) -> str:
        decode_mapping = {v: k for k, v in mapping.items()}
        assert len(decode_mapping) == len(mapping), "1:1 mapping"
        return decode_mapping[index]

    @staticmethod
    def decode_list(indices: List[int]) -> List[str]:
        decode_mapping = {v: k for k, v in mapping.items()}
        assert len(decode_mapping) == len(mapping), "1:1 mapping"
        return [decode_mapping[index] for index in indices]


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
        decode_mapping = {v: k for k, v in self.mapping.items()}
        assert len(decode_mapping) == len(self.mapping), "1:1 mapping"
        return decode_mapping[index]

    def decode_list(self, indices: Union[List[int], Tuple[int]]) -> List[int]:
        decode_mapping = {v: k for k, v in self.mapping.items()}
        assert len(decode_mapping) == len(self.mapping), "1:1 mapping"
        return [decode_mapping[index] for index in indices]


if __name__ == "__main__":
    le = LabelEncoder
    print(le.encode("a"))
    print(le.encode("b"))
    print(le.encode("a"))
    print(le.encode_list(["a", "b", "a"]))
    le2 = LabelEncoder
    print(mapping.keys())
    print(le2.decode_list([0, 1, 0]))
