BUILD_DIR ?= build-cpp
CMAKE ?= cmake
TRACK1_INPUT ?= artifacts/track1
TRACK1_OUTPUT ?= artifacts/track1/training
TRACK1_TOKENIZER ?= data/stage-10/tokenizer_snapshot.bin
TRACK1_PRETRAIN_STEPS ?= 200
TRACK1_SFT_STEPS ?= 120
TRACK1_CONTEXT ?= 32
TRACK1_EMBEDDING ?= 4
TRACK1_HIDDEN ?= 4
TRACK1_SFT_CONTEXT_BYTES ?= 1024
TRACK1_SEED ?= 1701

.PHONY: configure native-build native-test stage0-smoke stage0-gate stage1-test stage1-gate stage2-test stage2-gate stage3-test stage3-gate stage4-test stage4-gate stage5-test stage5-gate stage6-test stage6-gate stage7-test stage7-gate stage8-test stage8-gate stage9-test stage9-gate stage10-test stage10-gate stage11-test stage11-gate stage12-test stage12-gate stage13-test stage13-gate stage14-test stage14-gate stage15-test stage15-gate stage16-test stage16-gate stage17-test stage17-gate track1-test track1-gate track1-train ci-stage2 ci-stage3 ci-stage4 ci-stage5 ci-stage6 ci-stage7 ci-stage8 ci-stage9 ci-stage10 ci-stage11 ci-stage12 ci-stage13 ci-stage14 ci-stage15 ci-stage16 ci-stage17 ci-track1 ci clean

configure:
	$(CMAKE) -S cpp -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release

native-build: configure
	$(CMAKE) --build $(BUILD_DIR) --parallel 2

native-test: native-build
	ctest --test-dir $(BUILD_DIR) --output-on-failure

stage0-smoke: native-build
	./$(BUILD_DIR)/cct_gate_envelope --output artifacts/stage-0/cpp-gate -- ./$(BUILD_DIR)/cct_stage0_gate --output artifacts/stage-0/cpp-gate

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
stage6-test: native-build
	./$(BUILD_DIR)/cct_deliberation_tests
stage6-gate: native-build
	./$(BUILD_DIR)/cct_stage6_gate --output artifacts/stage-6/cpp-gate
stage7-test: native-build
	./$(BUILD_DIR)/cct_multimodal_tests
stage7-gate: native-build
	./$(BUILD_DIR)/cct_stage7_gate --output artifacts/stage-7/cpp-gate
stage8-test: native-build
	./$(BUILD_DIR)/cct_production_tests
stage8-gate: native-build
	./$(BUILD_DIR)/cct_stage8_gate --output artifacts/stage-8/cpp-gate
stage9-test: native-build
	./$(BUILD_DIR)/cct_corpus_tests
stage9-gate: native-build
	./$(BUILD_DIR)/cct_stage9_gate --output artifacts/stage-9/cpp-gate
stage10-test: native-build
	./$(BUILD_DIR)/cct_tokenizer_tests
stage10-gate: native-build
	./$(BUILD_DIR)/cct_stage10_gate --output artifacts/stage-10/cpp-gate
stage11-test: native-build
	./$(BUILD_DIR)/cct_nlp_trainer_tests
stage11-gate: native-build
	./$(BUILD_DIR)/cct_stage11_gate --output artifacts/stage-11/cpp-gate
stage12-test: native-build
	./$(BUILD_DIR)/cct_scaling_systems_tests
stage12-gate: native-build
	./$(BUILD_DIR)/cct_stage12_gate --output artifacts/stage-12/cpp-gate
stage13-test: native-build
	./$(BUILD_DIR)/cct_sft_tests
stage13-gate: native-build
	./$(BUILD_DIR)/cct_stage13_gate --output artifacts/stage-13/cpp-gate
stage14-test: native-build
	./$(BUILD_DIR)/cct_preference_tests
stage14-gate: native-build
	./$(BUILD_DIR)/cct_stage14_gate --output artifacts/stage-14/cpp-gate
stage15-test: native-build
	./$(BUILD_DIR)/cct_knowledge_tests
stage15-gate: native-build
	./$(BUILD_DIR)/cct_stage15_gate --output artifacts/stage-15/cpp-gate
stage16-test: native-build
	./$(BUILD_DIR)/cct_inference_tests
stage16-gate: native-build
	./$(BUILD_DIR)/cct_stage16_gate --output artifacts/stage-16/cpp-gate
stage17-test: native-build
	./$(BUILD_DIR)/cct_release_tests
stage17-gate: native-build
	./$(BUILD_DIR)/cct_stage17_gate --output artifacts/stage-17/cpp-gate

track1-test: native-build
	./$(BUILD_DIR)/cct_track1_tests

track1-gate: native-build
	./$(BUILD_DIR)/cct_track1_gate --output artifacts/track1/cpp-gate

track1-train: native-build
	./$(BUILD_DIR)/cct_track1_train --input $(TRACK1_INPUT) --output $(TRACK1_OUTPUT) --tokenizer $(TRACK1_TOKENIZER) \
		--pretrain-steps $(TRACK1_PRETRAIN_STEPS) --sft-steps $(TRACK1_SFT_STEPS) --context $(TRACK1_CONTEXT) \
		--embedding $(TRACK1_EMBEDDING) --hidden $(TRACK1_HIDDEN) --sft-context-bytes $(TRACK1_SFT_CONTEXT_BYTES) --seed $(TRACK1_SEED)

ci-stage2: native-build native-test stage0-gate stage1-test stage1-gate stage2-test stage2-gate

ci-stage3: ci-stage2 stage3-test stage3-gate
ci-stage4: ci-stage3 stage4-test stage4-gate
ci-stage5: ci-stage4 stage5-test stage5-gate
ci-stage6: ci-stage5 stage6-test stage6-gate
ci-stage7: ci-stage6 stage7-test stage7-gate
ci-stage8: ci-stage7 stage8-test stage8-gate
ci-stage9: ci-stage8 stage9-test stage9-gate
ci-stage10: ci-stage9 stage10-test stage10-gate
ci-stage11: ci-stage10 stage11-test stage11-gate
ci-stage12: ci-stage11 stage12-test stage12-gate
ci-stage13: ci-stage12 stage13-test stage13-gate
ci-stage14: ci-stage13 stage14-test stage14-gate
ci-stage15: ci-stage14 stage15-test stage15-gate
ci-stage16: ci-stage15 stage16-test stage16-gate
ci-stage17: ci-stage16 stage17-test stage17-gate
ci-track1: ci-stage17 track1-test track1-gate
ci: native-build native-test stage0-gate stage1-test stage1-gate

clean:
	rm -rf $(BUILD_DIR) artifacts/stage-0 artifacts/stage-1 artifacts/stage-2 artifacts/stage-3 artifacts/stage-4 artifacts/stage-5 artifacts/stage-6 artifacts/stage-7 artifacts/stage-8 artifacts/stage-9 artifacts/stage-10 artifacts/stage-11 artifacts/stage-12 artifacts/stage-13 artifacts/stage-14 artifacts/stage-15 artifacts/stage-16 artifacts/stage-17
