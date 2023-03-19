// Copyright (c) 2022, Graphcore Ltd, All rights reserved.

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/tensorindex.hpp>
#include <popart/op/collectives/replicatedallreduce.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/tensor.hpp>


namespace {
struct RemoveAllReducePattern : popart::PreAliasPattern {
    bool matches(popart::Op* op) const override {
        return op->isConvertibleTo<popart::ReplicatedAllReduceOp>();
    }

    std::vector<const popart::Tensor*> touches(popart::Op*) const override { return {}; }

    bool apply(popart::Op* op) const override {
        auto rar_op = static_cast<popart::ReplicatedAllReduceOp*>(op);
        if (rar_op->getReplicaGrouping().getGroupSize() == 1) {
            popart::Tensor* in_rar = rar_op->inTensor(popart::ReplicatedAllReduceOp::getInIndex());
            popart::Tensor* out_rar =
                rar_op->outTensor(popart::ReplicatedAllReduceOp::getOutIndex());
            // std::cerr << "Removing ReplicatedAllReduceOp with groupSize=1: " << in_rar->id
            //           << std::endl;
            for (auto cons : out_rar->consumers.getOps()) {
                for (auto in_index : cons->input->indices(out_rar)) {
                    cons->disconnectInTensor(out_rar);
                    cons->connectInTensor(in_index, in_rar->id);
                }
            }
            op->disconnectAllInputs();
            op->disconnectAllOutputs();
            op->getGraph().eraseOp(rar_op->id);
            return true;
        }
        return false;
    }
};

static popart::PatternCreator<RemoveAllReducePattern> RemoveAllReducePatternCreator(
    "RemoveAllReducePattern",
    false);
}
