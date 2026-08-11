BUILD_DIR ?= build-cpp
CMAKE ?= cmake

.PHONY: configure native-build native-test stage0-smoke stage0-gate stage1-test stage1-gate stage2-test stage2-gate stage3-test stage3-gate stage4-test stage4-gate stage5-test stage5-gate ci-stage2 ci-stage3 ci-stage4 ci-stage5 ci clean

configure:
	$(CMAKE) -S cpp -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release

native-build: configure
	$(CMAKE) --build $(BUILD_DIR) --parallel 2

native-test: native-build
	ctest --test-dir $(BUILD_DIR) --output-on-failure

stage0-smoke: native-build
	./$(BUILD_DIR)/cct_stage0_gate --output artifacts/stage-0/cpp-gate

stage0-gate: stage0-smoke

stage1-test: native-build
	./$(BUILD_DIR)/cct_tests

stage1-gate: native-build
	./$(BUILD_DIR)/cct_stage1_gate --output artifacts/stage-1/cpp-gate

stage2-test: native-build
	./$(BUILD_DIR)/cct_sequence_tests

stage2-gate: native-build
	./$(BUILD_DIR)/cct_stage2_gate --output artifacts/stage-2/cpp-gate

stage3-test: native-build
	./$(BUILD_DIR)/cct_causal_tests

stage3-gate: native-build
	./$(BUILD_DIR)/cct_stage3_gate --output artifacts/stage-3/cpp-gate
stage4-test: native-build
	./$(BUILD_DIR)/cct_memory_tests
stage4-gate: native-build
	./$(BUILD_DIR)/cct_stage4_gate --output artifacts/stage-4/cpp-gate
stage5-test: native-build
	./$(BUILD_DIR)/cct_scaling_tests
stage5-gate: native-build
	./$(BUILD_DIR)/cct_stage5_gate --output artifacts/stage-5/cpp-gate
ci-stage2: native-build native-test stage0-gate stage1-test stage1-gate stage2-test stage2-gate

ci-stage3: ci-stage2 stage3-test stage3-gate
ci-stage4: ci-stage3 stage4-test stage4-gate
ci-stage5: ci-stage4 stage5-test stage5-gate
ci: native-build native-test stage0-gate stage1-test stage1-gate

clean:
	rm -rf $(BUILD_DIR) artifacts/stage-0 artifacts/stage-1 artifacts/stage-2 artifacts/stage-3 artifacts/stage-4 artifacts/stage-5
