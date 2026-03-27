// ============================================================================
// Test 5: Asymmetric tile sizes for different dimensions
// tile-sizes = [16, 32, 64] instead of [32, 32, 32]
// Expectation: Different loop steps for different dimensions
// ============================================================================

// RUN: my-tiling-opt %s -my-tiling="tile-sizes=16,32,64" | FileCheck %s

func.func @matmul_asymmetric_tiles(
    %A: memref<128x64xf32>,
    %B: memref<64x96xf32>,
    %C: memref<128x96xf32>) {

  linalg.matmul
    ins(%A, %B: memref<128x64xf32>, memref<64x96xf32>)
    outs(%C: memref<128x96xf32>)

  return
}

// ===========================================================================
// CHECK-LABEL: func.func @matmul_asymmetric_tiles
// ===========================================================================

// CHECK-DAG: arith.constant 0 : index
// CHECK-DAG: arith.constant 128 : index
// CHECK-DAG: arith.constant 64 : index
// CHECK-DAG: arith.constant 96 : index
// CHECK-DAG: arith.constant 16 : index
// CHECK-DAG: arith.constant 32 : index

// CHECK-DAG: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %c16
// CHECK-DAG: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %c32
// CHECK-DAG: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %c64

// CHECK-DAG: memref.subview
// CHECK-DAG: memref.subview
// CHECK-DAG: memref.subview

// CHECK-DAG: linalg.matmul

// CHECK: return
