//===- Passes.h - Linalg Tiling passes  ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MY_TILING_PASSES_H
#define MY_TILING_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace my_tiling {
#define GEN_PASS_DECL
#include "MyTiling/Passes.h.inc"
} // namespace my_tiling

std::unique_ptr<Pass> createMyTilingPass();

namespace my_tiling {
#define GEN_PASS_REGISTRATION
#include "MyTiling/Passes.h.inc"
} // namespace my_tiling
} // namespace mlir

#endif
