// ============================================================================
// Test 3: Two sequential matrix multiplication operations
// Both operations should be tiled
// ============================================================================

// RUN: my-tiling-opt %s -my-tiling="tile-sizes=32,32,32" | FileCheck %s

func.func @two_matmuls_chain(
    %A: memref<64x64xf32>,
    %B: memref<64x64xf32>,
    %C: memref<64x64xf32>,
    %E: memref<64x64xf32>,
    %D: memref<64x64xf32>) {

  linalg.matmul
    ins(%A, %B: memref<64x64xf32>, memref<64x64xf32>)
    outs(%C: memref<64x64xf32>)

  linalg.matmul
    ins(%C, %E: memref<64x64xf32>, memref<64x64xf32>)
    outs(%D: memref<64x64xf32>)

  return
}

// ===========================================================================
// CHECK-LABEL: func.func @two_matmuls_chain
// ===========================================================================

// CHECK-DAG: arith.constant 0 : index
// CHECK-DAG: arith.constant 64 : index
// CHECK-DAG: arith.constant 32 : index

// CHECK-DAG: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK-DAG: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK-DAG: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK-DAG: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK-DAG: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK-DAG: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}

// CHECK-DAG: memref.subview
// CHECK-DAG: memref.subview
// CHECK-DAG: memref.subview
// CHECK-DAG: memref.subview
// CHECK-DAG: memref.subview
// CHECK-DAG: memref.subview

// CHECK-DAG: linalg.matmul
// CHECK-DAG: linalg.matmul

// CHECK: return
