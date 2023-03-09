// Copyright(c) 2022 Graphcore Ltd.All rights reserved.

#include "replicatedalltoall.hpp"
#include <algorithm>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/collectives/collectives.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

namespace popart
{

  ReplicatedAllToAllOp::ReplicatedAllToAllOp(const OperatorIdentifier &_opid,
                                             CommGroup group,
                                             const Op::Settings &settings_)
      : CollectivesBaseOp(_opid, group, settings_) {}

  std::unique_ptr<Op> ReplicatedAllToAllOp::clone() const
  {
    return std::make_unique<ReplicatedAllToAllOp>(*this);
  }

  void ReplicatedAllToAllOp::setup()
  {
    outInfo(getOutIndex()) = inInfo(getInIndex());
  }

  static OpDefinition::DataTypes T = {DataType::FLOAT, DataType::FLOAT16,
                                      DataType::INT32, DataType::UINT32};

  static OpDefinition ReplicatedAllToAllOpDef({OpDefinition::Inputs({{"X", T}}),
                                               OpDefinition::Outputs({{"Y", T}}),
                                               OpDefinition::Attributes({})});

  static OpCreator<ReplicatedAllToAllOp> ReplicatedAllToAllOpCreator(
      OpDefinitions({{{"custom.ops", "ReplicatedAllToAll", 1}, ReplicatedAllToAllOpDef}}),
      [](const OpCreatorInfo &info)
      {
        return std::make_unique<ReplicatedAllToAllOp>(info.opid, extractCommGroupFromAttrs(info.attributes), info.settings);
      },
      true);

  ReplicatedTensorShardingIndices
  ReplicatedAllToAllOp::getReplicatedTensorShardingIndices() const
  {
    return {{{ReplicatedAllToAllOp::getInIndex()}, {}}};
  }

  std::vector<std::unique_ptr<popart::Op>> ReplicatedAllToAllOp::getGradOps()
  {
    std::vector<std::unique_ptr<Op>> upops;
    upops.emplace_back(std::make_unique<ReplicatedAllToAllGradOp>(*this));
    return upops;
  }

  ReplicatedAllToAllGradOp::ReplicatedAllToAllGradOp(
      const ReplicatedAllToAllOp &op)
      : ReplicatedAllToAllOp(op.opid, op.getGCLCommGroup(), op.settings) {}

  const std::map<int, int> &ReplicatedAllToAllGradOp::gradOutToNonGradIn() const
  {
    static const std::map<int, int> outInfo = {
        {getOutIndex(), ReplicatedAllToAllOp::getInIndex()}};
    return outInfo;
  }

  const std::vector<GradInOutMapper> &ReplicatedAllToAllGradOp::gradInputInfo() const
  {
    static const std::vector<GradInOutMapper> inInfo = {
        {getInIndex(), ReplicatedAllToAllOp::getOutIndex(), GradOpInType::GradOut}};
    return inInfo;
  }

} // namespace popart
