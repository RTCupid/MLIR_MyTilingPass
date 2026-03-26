#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Support/LogicalResult.h"

#include "MyTiling/Passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);

  mlir::my_tiling::registerMyTilingPass();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv,
                        "My Tiling Optimization Tool",
                        registry));
}
