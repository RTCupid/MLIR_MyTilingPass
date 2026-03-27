// ============================================================================
// Test 4: Operations with tensor semantics (tensor, not memref)
// Expectation: tensor.extract_slice/insert_slice instead of memref.subview
// ============================================================================

// RUN: my-tiling-opt %s -my-tiling="tile-sizes=32,32,32" | FileCheck %s

func.func @matmul_tensor(
    %A: tensor<64x64xf32>,
    %B: tensor<64x64xf32>) -> tensor<64x64xf32> {

  %init = tensor.empty() : tensor<64x64xf32>

  %0 = linalg.matmul
    ins(%A, %B: tensor<64x64xf32>, tensor<64x64xf32>)
    outs(%init: tensor<64x64xf32>)
    -> tensor<64x64xf32>

  return %0 : tensor<64x64xf32>
}

// ===========================================================================
// CHECK-LABEL: func.func @matmul_tensor
// ===========================================================================

// CHECK-DAG: tensor.empty
// CHECK-DAG: arith.constant 0 : index
// CHECK-DAG: arith.constant 64 : index
// CHECK-DAG: arith.constant 32 : index

// CHECK-DAG: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK-DAG: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK-DAG: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}

// CHECK-DAG: tensor.extract_slice
// CHECK-DAG: tensor.extract_slice
// CHECK-DAG: tensor.extract_slice

// CHECK-DAG: tensor.insert_slice

// CHECK-DAG: linalg.matmul

// CHECK: return
