#include "cct/track1.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

std::size_t number_argument(const std::string& value, const std::string& name) {
    try {
        std::size_t consumed = 0U;
        const auto parsed = std::stoull(value, &consumed);
        if (consumed != value.size()) throw std::invalid_argument("trailing characters");
        return static_cast<std::size_t>(parsed);
    } catch (const std::exception&) {
        throw cct::Track1Error("invalid numeric value for " + name);
    }
}

void require_argument(const int index, const int argc, const std::string& name) {
    if (index + 1 >= argc) throw cct::Track1Error("missing value for " + name);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        cct::Track1Config config;
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument == "--output") {
                require_argument(index, argc, argument);
                config.output_root = argv[++index];
            } else if (argument == "--page-length") {
                require_argument(index, argc, argument);
                config.page_length = number_argument(argv[++index], argument);
            } else if (argument == "--pretrain-token-cap") {
                require_argument(index, argc, argument);
                config.pretrain_token_cap = number_argument(argv[++index], argument);
            } else if (argument == "--sft-examples") {
                require_argument(index, argc, argument);
                config.sft_examples = number_argument(argv[++index], argument);
            } else if (argument == "--sft-eval-examples") {
                require_argument(index, argc, argument);
                config.sft_eval_examples = number_argument(argv[++index], argument);
            } else if (argument == "--source-row-limit") {
                require_argument(index, argc, argument);
                config.source_row_limit = number_argument(argv[++index], argument);
            } else if (argument == "--squad-train-offset") {
                require_argument(index, argc, argument);
                config.squad_train_row_offset = number_argument(argv[++index], argument);
            } else if (argument == "--squad-final-offset") {
                require_argument(index, argc, argument);
                config.squad_final_test_row_offset = number_argument(argv[++index], argument);
            } else if (argument == "--seed") {
                require_argument(index, argc, argument);
                config.selection_seed = static_cast<std::uint64_t>(number_argument(argv[++index], argument));
            } else if (argument == "--no-download") {
                config.acquire_remote = false;
            } else if (argument == "--fixture") {
                config.allow_small_fixture = true;
            } else if (argument == "--help") {
                std::cout << "track1_prepare --output PATH --page-length N --pretrain-token-cap N --sft-examples N "
                             "--sft-eval-examples N --source-row-limit N --squad-train-offset N --squad-final-offset N "
                             "--seed N [--no-download] [--fixture]\n";
                return 0;
            } else {
                throw cct::Track1Error("unknown argument " + argument);
            }
        }
        cct::Track1Pipeline pipeline(config);
        pipeline.prepare();
        std::cout << "{\"status\":\"PASS\",\"manifest\":\"" << pipeline.report().manifest_path << "\",\"pretrain_tokens\":"
                  << pipeline.manifest().pretrain_train_tokens << ",\"sft_train_examples\":" << pipeline.manifest().sft_train_examples
                  << ",\"sft_evaluation_examples\":" << pipeline.manifest().sft_evaluation_examples << ",\"final_test_examples\":"
                  << pipeline.manifest().final_test_examples << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "{\"status\":\"FAIL\",\"error\":\"" << error.what() << "\"}\n";
        return 1;
    }
}
