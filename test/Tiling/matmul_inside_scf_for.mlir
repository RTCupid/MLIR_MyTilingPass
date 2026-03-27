// ============================================================================
// Test 9: Matmul inside scf.for loop
// Expectation: Outer loop remains, three inner loops over tiles of size 32
// ============================================================================

// RUN: my-tiling-opt %s -my-tiling="tile-sizes=32,32,32" | FileCheck %s

func.func @matmul_64x64x64_in_scf_for(
    %A: memref<64x64xf32>,
    %B: memref<64x64xf32>,
    %C: memref<64x64xf32>,
    %iter: index) {

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  scf.for %i = %c0 to %iter step %c1 {
    linalg.matmul
      ins(%A, %B : memref<64x64xf32>, memref<64x64xf32>)
      outs(%C : memref<64x64xf32>)
  }

  return
}

// ===========================================================================
// CHECK-LABEL: func.func @matmul_64x64x64_in_scf_for
// ===========================================================================

// Check constants and outer loop
// CHECK-DAG: %c0 = arith.constant 0 : index
// CHECK-DAG: %c1 = arith.constant 1 : index
// CHECK: scf.for %{{.*}} = %c0 to %{{.*}} step %c1 {

// Check inner constants for tile loops (0, 64, 32)
// CHECK-DAG: %c0_0 = arith.constant 0 : index
// CHECK-DAG: %c64 = arith.constant 64 : index
// CHECK-DAG: %c32 = arith.constant 32 : index

// Check three nested loops with step 32
// CHECK: scf.for %{{.*}} = %c0_0 to %c64 step %c32 {
// CHECK:   scf.for %{{.*}} = %c0_1 to %c64_2 step %c32_3 {
// CHECK:     scf.for %{{.*}} = %c0_4 to %c64_5 step %c32_6 {

// Check subviews and matmul inside the innermost loop
// CHECK-DAG:   %subview = memref.subview %{{.*}}[%{{.*}}, %{{.*}}] [32, 32] [1, 1] : memref<64x64xf32> to memref<32x32xf32, strided<[64, 1], offset: ?>>
// CHECK-DAG:   %subview_7 = memref.subview %{{.*}}[%{{.*}}, %{{.*}}] [32, 32] [1, 1] : memref<64x64xf32> to memref<32x32xf32, strided<[64, 1], offset: ?>>
// CHECK-DAG:   %subview_8 = memref.subview %{{.*}}[%{{.*}}, %{{.*}}] [32, 32] [1, 1] : memref<64x64xf32> to memref<32x32xf32, strided<[64, 1], offset: ?>>
// CHECK:       linalg.matmul ins(%subview, %subview_7 : memref<32x32xf32, strided<[64, 1], offset: ?>>, memref<32x32xf32, strided<[64, 1], offset: ?>>) outs(%subview_8 : memref<32x32xf32, strided<[64, 1], offset: ?>>)

// CHECK:     }
// CHECK:   }
// CHECK: }
// CHECK: return
