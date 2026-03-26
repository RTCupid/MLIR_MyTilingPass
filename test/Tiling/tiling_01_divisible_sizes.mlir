// ============================================================================
// Test 1: Matrices with dimensions divisible by tile size
// ============================================================================

// RUN: my-tiling-opt %s -my-tiling="tile-sizes=32,32,32" | FileCheck %s

func.func @matmul_64x64x64(
    %A: memref<64x64xf32>,
    %B: memref<64x64xf32>,
    %C: memref<64x64xf32>) {

  linalg.matmul
    ins(%A, %B: memref<64x64xf32>, memref<64x64xf32>)
    outs(%C: memref<64x64xf32>)

  return
}

// ===========================================================================
// CHECK-LABEL: func.func @matmul_64x64x64
// ===========================================================================

// CHECK-DAG: arith.constant 0 : index
// CHECK-DAG: arith.constant 64 : index
// CHECK-DAG: arith.constant 32 : index

// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}

// CHECK: memref.subview
// CHECK: memref.subview
// CHECK: memref.subview

// CHECK: linalg.matmul

// CHECK: }
// CHECK: }
// CHECK: }

// CHECK: return
