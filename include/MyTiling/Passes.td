//===- Passes.td - Linalg Tiling pass -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MY_TILING_PASS
#define MY_TILING_PASS

include "mlir/Pass/PassBase.td"

def MyTilingPass : Pass<"my-tiling", "mlir::func::FuncOp"> {
  let summary = "Tile linalg operations with user-specified tile sizes";
  let description = [{
    Applies tiling transformation to linalg operations using TilingInterface.
    Takes tile sizes as pass option.
  }];

  let constructor = "mlir::createMyTilingPass()";
  let dependentDialects = ["mlir::linalg::LinalgDialect", "mlir::scf::SCFDialect",
                           "mlir::memref::MemRefDialect", "mlir::tensor::TensorDialect"];

  let options = [
    ListOption<"tileSizes", "tile-sizes", "int64_t",
          "Tile sizes for each loop dimension">
  ];
}

#endif // MY_TILING_PASS
