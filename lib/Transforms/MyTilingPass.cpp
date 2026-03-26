#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

using namespace mlir;

struct TilingPattern : public RewritePattern {

  scf::SCFTilingOptions options;

  TilingPattern(MLIRContext *ctx, const scf::SCFTilingOptions &opts)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx), options(opts) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Operation *parent = op->getParentOp();
    while (parent) {
      if (isa<scf::ForOp, scf::ParallelOp>(parent)) {
        return failure();
      }
      parent = parent->getParentOp();
    }

    TilingInterface tilingOp = dyn_cast<TilingInterface>(op);
    if (!tilingOp) return failure();

    FailureOr<scf::SCFTilingResult> result =
      scf::tileUsingSCFForOp(rewriter, tilingOp, options);

    if (failed(result)) return failure();

    rewriter.replaceOp(op, result->replacements);

    return success();
  }
};

struct MyTilingPass
    : public PassWrapper<MyTilingPass, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MyTilingPass)

  ListOption<int64_t> tileSizes{
      *this, "tile-sizes",
      llvm::cl::desc("Tile sizes for each dimension"),
      llvm::cl::ZeroOrMore
  };

  MyTilingPass() = default;
  MyTilingPass(const MyTilingPass &other)
      : PassWrapper<MyTilingPass, OperationPass<func::FuncOp>>(other) {}

  StringRef getArgument() const override {
    return "my-tiling";
  }

  StringRef getDescription() const override {
    return "Tile linalg operations with user-specified tile sizes";
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    memref::MemRefDialect, tensor::TensorDialect>();
  }
};

void MyTilingPass::runOnOperation() {
  func::FuncOp func = getOperation();
  MLIRContext *ctx = &getContext();
  Builder b(ctx);

  SmallVector<OpFoldResult> tileSizesOFR =
      llvm::to_vector(llvm::map_range(tileSizes, [&](int64_t v) -> OpFoldResult {
        return b.getIndexAttr(v);
      }));
  if (tileSizesOFR.empty()) {
    tileSizesOFR = {b.getIndexAttr(32), b.getIndexAttr(32), b.getIndexAttr(32)};
  }

  scf::SCFTilingOptions tilingOptions;
  tilingOptions.setTileSizes(tileSizesOFR);

  RewritePatternSet patterns(ctx);
  patterns.add<TilingPattern>(ctx, tilingOptions);

  GreedyRewriteConfig config;
  config.maxIterations = 10;
  config.useTopDownTraversal = true;
  config.enableRegionSimplification = true;

  if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns), config))) {
    signalPassFailure();
    return;
  }
}

namespace mlir {
std::unique_ptr<Pass> createMyTilingPass() {
  return std::make_unique<MyTilingPass>();
}
} // namespace mlir
