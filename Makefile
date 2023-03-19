# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

all: custom_ops.so

custom_ops.so: custom_ops/*.cpp custom_ops/*.hpp
	g++ -std=c++14 -fPIC \
		-DONNX_NAMESPACE=onnx \
		custom_ops/remove_all_reduce_pattern.cpp \
		custom_ops/replicatedalltoall.cpp \
		custom_ops/replicatedalltoallx.cpp \
		custom_ops/replicatedallgather.cpp \
		custom_ops/replicatedallgatherx.cpp \
		custom_ops/distance_custom_ops.cpp \
		-shared -lpopart -lpoplar -lpoplin -lpopnn -lpopops -lpoputil -lpoprand \
		-o custom_ops.so

.PHONY : clean
clean:
	-rm custom_ops.so || true

