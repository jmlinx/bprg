from dataclasses import dataclass
from typing import Protocol

from jax import Array

type VertexNum = int
type TypeNum = int
type IntegralNum = int
type TypeDomain = tuple[float, float]
type VertexType = Array[float]
type VertexTypeIndex = Array[int]
type VertexThreshold = Array[int]
type Simplex = Array[float]
type Threshold = int
type AdjacencyMatrix = Array[int]
type Key = Array[int]
type SimulationNum = int
type Parameters = dict[str, Array]


class Function(Protocol):
    def __call__(self, x: VertexType) -> Array: ...


class TypeMeasure(Protocol):
    def __call__(self, x: VertexType) -> Simplex: ...


class ThresholdMeasure(Protocol):
    def __call__(self, x: VertexType) -> Simplex: ...


class Kernel(Protocol):
    def __call__(self, x: VertexType, y: VertexType) -> Array: ...


@dataclass
class RandomGraph:
    κ: Kernel
    μ: TypeMeasure
    S: TypeDomain
    nv: VertexNum
    ns: TypeNum


@dataclass
class BootstrapPercolation:
    G: RandomGraph
    η: ThresholdMeasure


@dataclass
class BootstrapPercolationRandomGraph:
    κ: Kernel
    μ: TypeMeasure
    η: ThresholdMeasure
    S: TypeDomain
    nv: VertexNum
    ns: TypeNum
