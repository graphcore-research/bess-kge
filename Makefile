# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

CXX ?= g++
OUT ?= build/besskge_custom_ops.so
OBJDIR ?= $(dir $(OUT))obj

CXXFLAGS = -Wall -Wextra -Werror -std=c++17 -fPIC -DONNX_NAMESPACE=onnx
LIBS = -lpopart -lpoplar -lpoplin -lpopnn -lpopops -lpoputil -lpoprand -lgcl

OBJECTS = $(OBJDIR)/remove_all_reduce_pattern.o # Add new custom ops here

.DEFAULT_GOAL := $(OUT)

$(OBJECTS): $(OBJDIR)/%.o: besskge/custom_ops/%.cpp
	@mkdir -p $(@D)
	$(CXX) -c $(CXXFLAGS) $< -o $@

$(OUT): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -shared $^ -o $@ -Wl,--no-undefined $(LIBS)
