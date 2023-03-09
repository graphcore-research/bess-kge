// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popart/popx/opx.hpp>
#include <popops/ElementWise.hpp>

#include "distance.hpp"
namespace CustomOperators {
const popart::OperatorIdentifier L2DistanceId = {"custom.ops", "L2Distance", 1};
} // namespace CustomOperators
namespace CustomGradOperators {
const popart::OperatorIdentifier L2DistanceGradId = {"custom.ops",
                                                     "L2DistanceGrad", 1};
} // namespace CustomGradOperators

class L2DistanceOp;
class L2DistanceOpx;
class L2DistanceGradOpx;

class L2DistanceGradOp : public popart::Op {
public:
  L2DistanceGradOp(const L2DistanceOp &fwdOp);

  std::unique_ptr<popart::Op> clone() const final {
    return std::make_unique<L2DistanceGradOp>(*this);
  }
  void setup() final {
    auto AInfo = inInfo(1);
    auto BInfo = inInfo(2);
    outInfo(0) = AInfo;
    outInfo(1) = BInfo;
  };

  const std::vector<popart::GradInOutMapper> &gradInputInfo() const;

  // The Grad Op has 1 output, which is the gradient of the only input
  const std::map<int, int> &gradOutToNonGradIn() const;

  bool requiresRandomSeed() const override { return false; }

  // an estimate of how valuable sub-graph matching will be
  float getSubgraphValue() const final { return getHighSubgraphValue(); }
};

class L2DistanceOp : public popart::Op {
public:
  L2DistanceOp(const popart::OperatorIdentifier &_opid,
               const popart::Op::Settings &settings_)
      : popart::Op(_opid, settings_) {}

  std::unique_ptr<Op> clone() const final {
    return std::make_unique<L2DistanceOp>(*this);
  }

  void setup() final {
    auto AInfo = inInfo(0);
    auto BInfo = inInfo(1);
    assert(AInfo.rank() == 2);
    assert(BInfo.rank() == 2);
    assert(AInfo.dim(1) == BInfo.dim(1));
    outInfo(0).set(AInfo.dataType(), {AInfo.dim(0), BInfo.dim(0)});
  }

  std::vector<std::unique_ptr<popart::Op>> getGradOps() {
    std::vector<std::unique_ptr<Op>> upops;
    upops.emplace_back(new L2DistanceGradOp(*this));
    return upops;
  }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  bool requiresRandomSeed() const override { return false; }
};

namespace {
using popart::DataType;
using popart::OpDefinition;

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition l2DistanceOpDef({OpDefinition::Inputs({{"a", T}, {"b", T}}),
                                     OpDefinition::Outputs({{"output", T}}),
                                     OpDefinition::Attributes()});

static popart::OpCreator<L2DistanceOp> l2DistanceOpCreator(
    popart::OpDefinitions({{CustomOperators::L2DistanceId, l2DistanceOpDef}}),
    [](const popart::OpCreatorInfo &info) {
      return std::make_unique<L2DistanceOp>(info.opid, info.settings);
    },
    true);
} // namespace

namespace pe = popops::expr;

class L2DistanceOpx : public popart::popx::Opx {
public:
  L2DistanceOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<L2DistanceOp>(op, {CustomOperators::L2DistanceId});
    graph().addCodelets("custom_ops/codelet.cpp"); // add codelets to the graph
  }

  void grow(poplar::program::Sequence &prog) const final {

    auto op = getOp<L2DistanceOp>();

    poplar::Tensor A = getInTensor(0);
    poplar::Tensor B = getInTensor(1);
    poplar::Tensor out = l2distance(graph(), A, B, prog, "l2distance");
    setOutTensor(0, out);
  }
};

class L2DistanceGradOpx : public popart::popx::Opx {
public:
  L2DistanceGradOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<L2DistanceGradOp>(op, {CustomGradOperators::L2DistanceGradId});
  }

  void grow(poplar::program::Sequence &prog) const final {

    auto op = getOp<L2DistanceGradOp>();

    poplar::Tensor grad = getInTensor(0);
    poplar::Tensor A = getInTensor(1);
    poplar::Tensor B = getInTensor(2);
    poplar::Tensor dist = getInTensor(3);
    auto gradA =
        l2distancegrad(graph(), A, B, dist, grad, prog, "l2distanceGrad");
    auto gradB = l2distancegrad(graph(), B, A, dist.transpose(),
                                grad.transpose(), prog, "l2distanceGrad");
    setOutTensor(0, gradA);
    setOutTensor(1, gradB);
  }
};

L2DistanceGradOp::L2DistanceGradOp(const L2DistanceOp &fwdOp)
    : popart::Op(CustomGradOperators::L2DistanceGradId, fwdOp.settings) {}

const std::vector<popart::GradInOutMapper> &
L2DistanceGradOp::gradInputInfo() const {
  static const std::vector<popart::GradInOutMapper> inInfo = {
      {0, 0, popart::GradOpInType::GradOut},
      {1, 0, popart::GradOpInType::In},
      {2, 1, popart::GradOpInType::In},
      {3, 0, popart::GradOpInType::Out}};
  return inInfo;
}

// The Grad Op has 1 output, which is the gradient of the only input
const std::map<int, int> &L2DistanceGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {{0, 0}, {1, 1}};
  return outInfo;
}

static popart::popx::OpxCreator<L2DistanceOpx>
    L2DistanceOpxCreator({CustomOperators::L2DistanceId});
static popart::popx::OpxCreator<L2DistanceGradOpx>
    L2DistanceGradOpxCreator({CustomGradOperators::L2DistanceGradId});

