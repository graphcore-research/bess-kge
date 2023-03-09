// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popart/popx/opx.hpp>
#include <popops/ElementWise.hpp>

#include "distance.hpp"
namespace CustomOperators {
const popart::OperatorIdentifier L1DistanceId = {"custom.ops", "L1Distance", 1};
} // namespace CustomOperators
namespace CustomGradOperators {
const popart::OperatorIdentifier L1DistanceGradId = {"custom.ops",
                                                     "L1DistanceGrad", 1};
} // namespace CustomGradOperators

class L1DistanceOp;
class L1DistanceOpx;
class L1DistanceGradOpx;

class L1DistanceGradOp : public popart::Op {
public:
  L1DistanceGradOp(const L1DistanceOp &fwdOp);

  std::unique_ptr<popart::Op> clone() const final {
    return std::make_unique<L1DistanceGradOp>(*this);
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

class L1DistanceOp : public popart::Op {
public:
  L1DistanceOp(const popart::OperatorIdentifier &_opid,
               const popart::Op::Settings &settings_)
      : popart::Op(_opid, settings_) {}

  std::unique_ptr<Op> clone() const final {
    return std::make_unique<L1DistanceOp>(*this);
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
    upops.emplace_back(new L1DistanceGradOp(*this));
    return upops;
  }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  bool requiresRandomSeed() const override { return false; }
};

namespace {
using popart::DataType;
using popart::OpDefinition;

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition l1DistanceOpDef({OpDefinition::Inputs({{"a", T}, {"b", T}}),
                                     OpDefinition::Outputs({{"output", T}}),
                                     OpDefinition::Attributes()});

static popart::OpCreator<L1DistanceOp> l1DistanceOpCreator(
    popart::OpDefinitions({{CustomOperators::L1DistanceId, l1DistanceOpDef}}),
    [](const popart::OpCreatorInfo &info) {
      return std::make_unique<L1DistanceOp>(info.opid, info.settings);
    },
    true);
} // namespace

namespace pe = popops::expr;

class L1DistanceOpx : public popart::popx::Opx {
public:
  L1DistanceOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<L1DistanceOp>(op, {CustomOperators::L1DistanceId});
    graph().addCodelets("custom_ops/codelet.cpp"); // add codelets to the graph
  }

  void grow(poplar::program::Sequence &prog) const final {

    auto op = getOp<L1DistanceOp>();

    poplar::Tensor A = getInTensor(0);
    poplar::Tensor B = getInTensor(1);
    poplar::Tensor out = l1distance(graph(), A, B, prog, "l1distance");
    setOutTensor(0, out);
  }
};

class L1DistanceGradOpx : public popart::popx::Opx {
public:
  L1DistanceGradOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<L1DistanceGradOp>(op, {CustomGradOperators::L1DistanceGradId});
  }

  void grow(poplar::program::Sequence &prog) const final {

    auto op = getOp<L1DistanceGradOp>();

    poplar::Tensor grad = getInTensor(0);
    poplar::Tensor A = getInTensor(1);
    poplar::Tensor B = getInTensor(2);
    auto gradA = l1distancegrad(graph(), A, B, grad, prog, "l1distanceGrad");
    auto gradB =
        l1distancegrad(graph(), B, A, grad.transpose(), prog, "l1distanceGrad");
    setOutTensor(0, gradA);
    setOutTensor(1, gradB);
  }
};

L1DistanceGradOp::L1DistanceGradOp(const L1DistanceOp &fwdOp)
    : popart::Op(CustomGradOperators::L1DistanceGradId, fwdOp.settings) {}

const std::vector<popart::GradInOutMapper> &
L1DistanceGradOp::gradInputInfo() const {
  static const std::vector<popart::GradInOutMapper> inInfo = {
      {0, 0, popart::GradOpInType::GradOut},
      {1, 0, popart::GradOpInType::In},
      {2, 1, popart::GradOpInType::In}};
  return inInfo;
}

// The Grad Op has 1 output, which is the gradient of the only input
const std::map<int, int> &L1DistanceGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {{0, 0}, {1, 1}};
  return outInfo;
}

static popart::popx::OpxCreator<L1DistanceOpx>
    L1DistanceOpxCreator({CustomOperators::L1DistanceId});
static popart::popx::OpxCreator<L1DistanceGradOpx>
    L1DistanceGradOpxCreator({CustomGradOperators::L1DistanceGradId});