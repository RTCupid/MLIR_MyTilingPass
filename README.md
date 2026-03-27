<div align="center">

# My Tiling Pass for matrix multiplication operations in Linalg dialect
  ![C++](https://img.shields.io/badge/C++-23-blue?style=for-the-badge&logo=cplusplus)
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

## Introduction

## Methodology

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
