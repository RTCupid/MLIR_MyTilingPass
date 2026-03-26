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

def MyTilingPass : Pass<"my-tiling"> {
  let summary = "Tile linalg operations with user-specified tile sizes";
  let description = [{
    Applies tiling transformation to linalg operations using TilingInterface.
    Takes tile sizes as pass option.
  }];

  let constructor = "mlir::createMyTilingPass()";
  let dependentDialects = ["mlir::Linalg::LinalgDialect", "mlir::scf::SCFDialect"];

  let options = [
    Option<"tileSizes", "tile-sizes", "ArrayRef<int64_t>",
           "/*default=*/{32, 32, 32}", "Tile sizes for each loop dimansion">
  ];
}

#endif // MY_TILING_PASS
