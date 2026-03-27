<div align="center">

# My Tiling Pass for matrix multiplication operations in Linalg dialect
  ![C++](https://img.shields.io/badge/C++-20-blue?style=for-the-badge&logo=cplusplus)
  ![CMake](https://img.shields.io/badge/CMake-3.20+-green?style=for-the-badge&logo=cmake)
  ![MLIR](https://img.shields.io/badge/MLIR-18.1+-yellow?style=for-the-badge&logo=llvm&logoColor=white)
  ![LLVM](https://img.shields.io/badge/LLVM-18.1+-blue?style=for-the-badge&logo=llvm)

</div>

My standalone MLIR project.

## Table of Contents
- [0. Running the program](#running-the-program)
- [1. Annotation](#annotation)
- [2. Introduction](#introduction)
- [3. Methodology](#methodology)
- [4. Implementation of the tiling pass](#implementation-of-the-tiling-pass)
- [5. Results](#results)
- [6. Conclusions](#conclusions)
- [Project structure](#project-structure)
- [Project author](#project-author)


## Running the program

This project expects a working MLIR/LLVM build with `MLIRConfig.cmake` available. Repository cloning, build and compilation is performed using the following commands:

```bash
git clone git@github.com:RTCupid/MLIR_MyTilingPass.git
cd MLIR_MyTilingPass
cmake -S . -B build -DMLIR_DIR=/path/to/lib/cmake/mlir
cmake --build build
```

Program execution is performed in the following format:

```bash
cd build/bin
./my-tiling-opt <mlir_program>
```

Also this tool contain options for M, N, K dimensions of tiling. Entry `-my-tiling="tile-sizes=%s,%s,%s`.

## Annotation
A tool based on `mlir-opt` implements `tiling` (block partitioning) for `matrix multiplication operations` within the `Linalg` dialect. It utilizes a `TableGen` specification alongside a `C++` class that derives from `PassWrapper`. The implementation employs the `TilingInterface` to identify applicable patterns and leverages `scf::tileUsingSCFForOp` to perform the tiling transformation.

The tool has been successfully tested on a wide range of configurations, including matrices with dimensions `divisible` and `non‑divisible` by `tile sizes`, `dynamic shapes`, two sequential `matrix multiplications`, `rectangular matrices` with different `M`, `N`, and `K` values, `tensors` and `memrefs`, as well as both `linalg.matmul` and `linalg.generic` operations. Additionally, correctness was verified for the case where the matrix multiplication operation is already placed inside a `scf.for` — a scenario that was not supported in the initial version.


## Introduction
In modern tensor compilers for deep learning, matrix multiplication (`matmul`) is the dominant operation in terms of computational cost. It accounts for the majority of execution time in both training and inference. The efficiency of matmul implementation heavily depends on how memory accesses are organized: without considering the memory hierarchy and target architecture, code generation results in excessive `DRAM` traffic, poor `cache` utilization, and underutilization of computational units.

A naive implementation using nested loops does not reorder data accesses. Each access loads an entire `cache line`, and repeated traversals over the same data cause massive unnecessary transfers between memory levels. Data reuse is minimal, performance becomes bounded by memory bandwidth, and `vector` instructions along with `core‑level` parallelism remain unused.

The proposed `tiling` pass addresses these issues by partitioning the operation into blocks. Tile sizes are selected to match `cache capacity`, maximizing data reuse and reducing `cache misses`. Additionally, the pass automates the transformation, eliminating the need to manually write complex loop nests. Its general design correctly handles dynamic shapes, rectangular matrices, and scenarios where the matmul operation is already nested inside loops.

## Methodology
The tiling pass is implemented using the `MLIR` framework, which provides a flexible intermediate representation (`IR`) and a comprehensive set of transformation utilities. Since matrix multiplication operations are part of the `Linalg` dialect, the optimization pass is specifically designed for this dialect.

Two main approaches can be used to implement such a pass in `MLIR`:

- Pattern‑based approach – defining a set of classes derived from `OpRewritePattern` and applying them using rewrite drivers such as `applyPatternsAndFoldGreedily`. This method is suitable when multiple related transformations need to be combined, but it requires explicit control over pattern application order.

- Manual operation traversal – iterating over all operations in the module and checking whether they implement the `TilingInterface`. This approach provides full control over the transformation process and simplifies debugging, as it does not rely on a `non‑deterministic` pattern walk.

In this work, the second approach is adopted, allowing precise tracking of tiling applied to each `linalg.matmul` and `linalg.generic` operation that performs matrix multiplication. The actual tiling is performed using the existing function `scf::tileUsingSCFForOp`, which takes an operation that implements `TilingInterface` and returns new `scf.for` loops containing the tiled operations.

The project is built in a `standalone` configuration, which isolates the development of the test pass from the main `MLIR` source tree and simplifies its integration into external tools.

## Implementation of the tiling pass

## Results

## Conclusions

## Project structure

## Project author

<div align="center">

  <a href="https://github.com/RTCupid">
    <img src="https://raw.githubusercontent.com/BulgakovDmitry/3D_triangles/main/img/A.jpeg" width="160" height="160" style="border-radius: 50%;">
  </a>
  <br>
  <a href="https://github.com/RTCupid"><strong>@RTCupid </strong></a>
  <br>
</div>
