// Copyright (c) 2022, Graphcore Ltd, All rights reserved.

#ifndef DISTANCE_H_
#define DISTANCE_H_

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>

poplar::Tensor l1distance(poplar::Graph &graph, const poplar::Tensor &a,
                          const poplar::Tensor &b,
                          poplar::program::Sequence &prog,
                          const poplar::DebugContext &debugContext);

poplar::Tensor l1distancegrad(poplar::Graph &graph, const poplar::Tensor &a,
                              const poplar::Tensor &b,
                              const poplar::Tensor &gradOutput,
                              poplar::program::Sequence &prog,
                              const poplar::DebugContext &debugContext);

poplar::Tensor l2distance(poplar::Graph &graph, const poplar::Tensor &a,
                          const poplar::Tensor &b,
                          poplar::program::Sequence &prog,
                          const poplar::DebugContext &debugContext);

poplar::Tensor l2distancegrad(poplar::Graph &graph, const poplar::Tensor &a,
                              const poplar::Tensor &b,
                              const poplar::Tensor &dist,
                              const poplar::Tensor &gradOutput,
                              poplar::program::Sequence &prog,
                              const poplar::DebugContext &debugContext);

#endif // DISTANCE_H_

