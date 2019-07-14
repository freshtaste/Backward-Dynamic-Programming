from typing import TypeVar, Mapping, Tuple

S = TypeVar('S')
A = TypeVar('A')
SASTff = Mapping[S, Mapping[A, Mapping[S, Tuple[float, float]]]]

epsilon = 1e-8

def is_approx_eq(a: float, b: float) -> bool:
    return abs(a - b) <= epsilon