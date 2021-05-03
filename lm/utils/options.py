from dataclasses import dataclass, field
from typing import Generic, TypeVar, List

T = TypeVar('T')

@dataclass
class OptionBridge(Generic[T]):
    name: str 
    value: T 
    desc: str = field(default="")
    required: bool = field(default=False)
    choices: List[T] = field(default_factory=list)
    default: T = field(default="")

    def __call__(self) -> T:
        return self.value

    def __repr__(self) -> str:
        return f"{self.value}"
