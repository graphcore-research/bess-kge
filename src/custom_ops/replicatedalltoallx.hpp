// Copyright(c) 2022 Graphcore Ltd.All rights reserved.

#ifndef GUARD_NEURALNET_REPLICATEDALLTOALLX_HPP
#define GUARD_NEURALNET_REPLICATEDALLTOALLX_HPP

#include <popart/popx/op/collectives/collectivesx.hpp>

namespace popart
{
  namespace popx
  {

    class ReplicatedAllToAllOpx : public CollectivesBaseOpx
    {
    public:
      ReplicatedAllToAllOpx(Op *, Devicex *);
      void grow(snap::program::Sequence &) const final;
      InputCreatorType getInputCreatorType(InIndex index) const final;
      snap::Tensor unwindTensorLayout(snap::Tensor, InIndex, OutIndex) const final;
      view::RegMap unwindRegion(InIndex, OutIndex) const final;
    };

    class ReplicatedAllToAllGradOpx : public ReplicatedAllToAllOpx
    {
    public:
      ReplicatedAllToAllGradOpx(Op *, Devicex *);
    };

  } // namespace popx
} // namespace popart

#endif
