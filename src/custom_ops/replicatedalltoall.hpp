// Copyright(c) 2022 Graphcore Ltd.All rights reserved.

#ifndef GUARD_NEURALNET_REPLICATEDALLTOALL_HPP
#define GUARD_NEURALNET_REPLICATEDALLTOALL_HPP

#include <popart/op/collectives/collectives.hpp>

namespace popart
{

  class ReplicatedAllToAllOp : public CollectivesBaseOp
  {
  public:
    ReplicatedAllToAllOp(const OperatorIdentifier &, CommGroup group,
                         const Op::Settings &);

    std::unique_ptr<Op> clone() const final;
    void setup() final;

    float getSubgraphValue() const final { return getHighSubgraphValue(); }

    ReplicatedTensorShardingIndices
    getReplicatedTensorShardingIndices() const override;

    std::vector<std::unique_ptr<Op>> getGradOps() final;
  };

  class ReplicatedAllToAllGradOp : public ReplicatedAllToAllOp
  {
  public:
    ReplicatedAllToAllGradOp(const ReplicatedAllToAllOp &);

    const std::map<int, int> &gradOutToNonGradIn() const override final;
    const std::vector<GradInOutMapper> &gradInputInfo() const final;
  };

} // namespace popart

#endif
