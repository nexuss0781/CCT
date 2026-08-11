BUILD_DIR ?= build-cpp
CMAKE ?= cmake

.PHONY: configure native-build native-test stage0-smoke stage0-gate stage1-test stage1-gate ci clean

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

ci: native-build native-test stage0-gate stage1-test stage1-gate

clean:
	rm -rf $(BUILD_DIR) artifacts/stage-0 artifacts/stage-1
