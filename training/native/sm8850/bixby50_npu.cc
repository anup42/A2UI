// Android evaluation driver for LiteRT-LM v0.17.0. One engine, one fresh
// session per source. Only successful completions enter results/index.tsv.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "c/engine.h"

namespace {
template <class T, class D>
auto Own(T* handle, D destroy) {
  return std::unique_ptr<T, D>(handle, destroy);
}

bool ReadStartToken(LiteRtLmEngine* engine, int* id, std::string* text) {
  auto token = Own(litert_lm_engine_get_start_token(engine), litert_lm_token_union_delete);
  if (!token) return false;
  const int* ids = nullptr;
  size_t count = 0;
  if (litert_lm_token_union_get_type(token.get()) == kLiteRtLmTokenUnionTypeString) {
    const char* value = litert_lm_token_union_get_string(token.get());
    if (!value || !*value) return false;
    auto encoded = Own(litert_lm_engine_tokenize(engine, value), litert_lm_tokenize_result_delete);
    if (!encoded || litert_lm_tokenize_result_get_num_tokens(encoded.get()) != 1) return false;
    *id = litert_lm_tokenize_result_get_tokens(encoded.get())[0];
    *text = value;
  } else {
    if (litert_lm_token_union_get_ids(token.get(), &ids, &count) || count != 1) return false;
    *id = ids[0];
    auto decoded = Own(litert_lm_engine_detokenize(engine, ids, count), litert_lm_detokenize_result_delete);
    if (!decoded || !litert_lm_detokenize_result_get_string(decoded.get())) return false;
    *text = litert_lm_detokenize_result_get_string(decoded.get());
  }
  return !text->empty();
}

bool Evaluate(LiteRtLmEngine* engine, LiteRtLmSessionConfig* config, const std::string& id,
              int bos, const std::string& bos_text, int capacity, int max_output, std::ofstream& index) {
  std::ifstream prompt_file("inputs/" + id + ".prompt", std::ios::binary);
  std::ifstream ids_file("inputs/" + id + ".ids");
  if (!prompt_file || !ids_file) return false;
  std::string prompt((std::istreambuf_iterator<char>(prompt_file)), {});
  std::vector<int> expected((std::istream_iterator<int>(ids_file)), {});
  if (expected.empty() || expected.front() != bos || prompt.find(bos_text) != 0 ||
      expected.size() + max_output > static_cast<size_t>(capacity)) return false;
  // Raw sessions prepend metadata BOS even when template application is off.
  prompt.erase(0, bos_text.size());
  auto tokens = Own(litert_lm_engine_tokenize(engine, prompt.c_str()), litert_lm_tokenize_result_delete);
  if (!tokens || litert_lm_tokenize_result_get_num_tokens(tokens.get()) + 1 != expected.size() ||
      !std::equal(expected.begin() + 1, expected.end(), litert_lm_tokenize_result_get_tokens(tokens.get()))) {
    std::cerr << id << ": native token IDs differ from the trained tokenizer\n";
    return false;
  }
  auto session = Own(litert_lm_engine_create_session(engine, config), litert_lm_session_delete);
  auto input = Own(litert_lm_input_data_create(kLiteRtLmInputDataTypeText, prompt.data(), prompt.size()),
                   litert_lm_input_data_delete);
  const LiteRtLmInputData* inputs[] = {input.get()};
  const auto start = std::chrono::steady_clock::now();
  if (!session || !input || litert_lm_session_run_prefill(session.get(), inputs, 1)) return false;
  auto result = Own(litert_lm_session_run_decode(session.get()), litert_lm_responses_delete);
  if (!result || litert_lm_responses_get_num_candidates(result.get()) != 1) return false;
  const char* text = litert_lm_responses_get_response_text_at(result.get(), 0);
  if (!text) return false;
  const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  std::ofstream output("results/" + id + ".out", std::ios::binary);
  output << text;
  output.close();
  if (!output) return false;
  index << id << '\t' << expected.size() << '\t' << elapsed << std::endl;
  std::cerr << id << " completed in " << elapsed << " s\n";
  return index.good();
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) return 2;  // model path, absolute library directory, cache, output budget
  const int capacity = std::atoi(argv[3]), max_output = std::atoi(argv[4]);
  if (capacity <= max_output || max_output <= 0) return 2;
  auto settings = Own(litert_lm_engine_settings_create(argv[1], "npu", nullptr, nullptr),
                      litert_lm_engine_settings_delete);
  if (!settings) return 1;
  litert_lm_engine_settings_set_litert_dispatch_lib_dir(settings.get(), argv[2]);
  litert_lm_engine_settings_set_cache_dir(settings.get(), "cache");
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), capacity);
  litert_lm_engine_settings_set_enable_speculative_decoding(settings.get(), false);
  auto engine = Own(litert_lm_engine_create(settings.get()), litert_lm_engine_delete);
  auto config = Own(litert_lm_session_config_create(), litert_lm_session_config_delete);
  auto greedy = Own(litert_lm_sampler_params_create(kLiteRtLmSamplerTypeGreedy), litert_lm_sampler_params_delete);
  if (!engine || !config || !greedy) return 1;
  litert_lm_session_config_set_sampler_params(config.get(), greedy.get());
  litert_lm_session_config_set_apply_prompt_template(config.get(), false);
  litert_lm_session_config_set_max_output_tokens(config.get(), max_output);
  int bos;
  std::string bos_text;
  if (!ReadStartToken(engine.get(), &bos, &bos_text)) {
    std::cerr << "Cannot bind the packaged Gemma start token\n";
    return 1;
  }
  std::ofstream index("results/index.tsv");
  if (!index) return 1;
  for (int i = 1; i <= 50; ++i) {
    char id[16];
    std::snprintf(id, sizeof(id), "BXP-%03d", i);
    if (!Evaluate(engine.get(), config.get(), id, bos, bos_text, capacity, max_output, index)) {
      std::cerr << "Failed at " << id << '\n';
      return 1;
    }
  }
  // Confirm both the vendor dispatch and its HTP backend loaded in this process.
  std::ifstream maps("/proc/self/maps");
  std::string mappings((std::istreambuf_iterator<char>(maps)), {});
  for (const char* library : {"libLiteRtDispatch_Qualcomm.so", "libQnnHtp.so"}) {
    if (mappings.find(library) == std::string::npos) {
      std::cerr << "Missing loaded NPU runtime " << library << '\n';
      return 1;
    }
  }
  std::ofstream proof("results/backend.txt");
  proof << "qualcomm-htp\nmtp-off\n";
  proof.close();
  return proof.good() ? 0 : 1;
}
