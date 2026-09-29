// Standalone Android LiteRT-LM C API probe. Build against the pinned C API
// headers/library; no GenUICraft app or JNI code is involved.
#include "engine.h"
#include "sampler_bridge.h"

#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  std::string model;
  std::string prompt;
  std::string score_target;
  std::string output_prefix;
  std::string backend;
  std::string sampler_backend = "default";
  bool force_f32 = false;
  int max_context = 8192;
  int max_output = 2048;
};

struct Metrics {
  int pid = static_cast<int>(getpid());
  std::string status = "error";
  std::string stage = "arguments";
  std::string error;
  size_t prompt_bytes = 0;
  size_t score_target_bytes = 0;
  size_t output_bytes = 0;
  int candidates = 0;
  std::optional<size_t> input_tokenizer_count;
  std::optional<size_t> target_tokenizer_count;
  std::optional<size_t> output_retokenized_count;
  std::optional<int> response_token_length;
  std::optional<float> response_score;
  std::optional<int> selected_token_score_count;
  std::optional<float> target_score;
  std::string target_score_status = "not_requested";
  std::optional<int> target_scored_token_length;
  std::optional<int> target_token_score_count;
  std::optional<int> target_nonfinite_token_score_count;
  std::string target_token_scores_status = "not_requested";
  std::optional<double> engine_create_ms;
  std::optional<double> prefill_wall_ms;
  std::optional<double> decode_wall_ms;
  std::optional<double> score_wall_ms;
  std::optional<double> generation_wall_ms;
  std::optional<double> time_to_first_token_s;
  std::optional<double> init_benchmark_s;
  std::vector<int> prefill_tokens;
  std::vector<int> decode_tokens;
  std::vector<double> prefill_tokens_per_s;
  std::vector<double> decode_tokens_per_s;
};

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

std::string read_binary(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open input: " + path);
  std::ostringstream contents;
  contents << in.rdbuf();
  if (!in && !in.eof()) throw std::runtime_error("cannot read input: " + path);
  return contents.str();
}

void write_binary(const std::string& path, const std::string& contents) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot open output: " + path);
  out.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  if (!out) throw std::runtime_error("cannot write output: " + path);
}

std::string json_string(const std::string& input) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : input) {
    switch (c) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\b': out << "\\b"; break;
      case '\f': out << "\\f"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (c < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(c) << std::dec << std::setfill(' ');
        } else {
          out << static_cast<char>(c);
        }
    }
  }
  out << '"';
  return out.str();
}

template <typename T>
void json_optional(std::ostream& out, const std::optional<T>& value) {
  if (value && std::isfinite(static_cast<double>(*value))) out << *value;
  else out << "null";
}

template <typename T>
void json_array(std::ostream& out, const std::vector<T>& values) {
  out << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i) out << ',';
    if (std::isfinite(static_cast<double>(values[i]))) out << values[i];
    else out << "null";
  }
  out << ']';
}

void write_metrics(const Options& options, const Metrics& metrics) {
  std::ostringstream out;
  out << std::setprecision(9);
  out << "{\n"
      << "  \"pid\": " << metrics.pid << ",\n"
      << "  \"status\": " << json_string(metrics.status) << ",\n"
      << "  \"stage\": " << json_string(metrics.stage) << ",\n"
      << "  \"error\": " << json_string(metrics.error) << ",\n"
      << "  \"model_path\": " << json_string(options.model) << ",\n"
      << "  \"prompt_path\": " << json_string(options.prompt) << ",\n"
      << "  \"mode\": " << json_string(options.score_target.empty() ? "decode" : "text_scoring") << ",\n"
      << "  \"score_target_path\": " << json_string(options.score_target) << ",\n"
      << "  \"cache_dir\": " << json_string(options.output_prefix + ".cache") << ",\n"
      << "  \"backend_requested\": " << json_string(options.backend) << ",\n"
      << "  \"sampler_backend_requested\": " << json_string(options.sampler_backend) << ",\n"
      << "  \"force_f32_activations\": " << (options.force_f32 ? "true" : "false") << ",\n"
      << "  \"max_context_tokens\": " << options.max_context << ",\n"
      << "  \"max_output_tokens\": " << options.max_output << ",\n"
      << "  \"sampler\": {\"type\": \"top_p\", \"top_k\": 1, \"top_p\": 1, \"temperature\": 0, \"seed\": 42},\n"
      << "  \"speculative_decoding\": false,\n"
      << "  \"apply_prompt_template\": false,\n"
      << "  \"prompt_bytes\": " << metrics.prompt_bytes << ",\n"
      << "  \"score_target_bytes\": " << metrics.score_target_bytes << ",\n"
      << "  \"output_bytes\": " << metrics.output_bytes << ",\n"
      << "  \"candidate_count\": " << metrics.candidates << ",\n"
      << "  \"input_tokenizer_count\": ";
  json_optional(out, metrics.input_tokenizer_count);
  out << ",\n  \"target_tokenizer_count\": ";
  json_optional(out, metrics.target_tokenizer_count);
  out << ",\n  \"output_retokenized_count\": ";
  json_optional(out, metrics.output_retokenized_count);
  out << ",\n  \"response_token_length\": ";
  json_optional(out, metrics.response_token_length);
  out << ",\n  \"response_score\": ";
  json_optional(out, metrics.response_score);
  out << ",\n  \"selected_token_score_count\": ";
  json_optional(out, metrics.selected_token_score_count);
  out << ",\n  \"target_score\": ";
  json_optional(out, metrics.target_score);
  out << ",\n  \"target_score_status\": " << json_string(metrics.target_score_status);
  out << ",\n  \"target_scored_token_length\": ";
  json_optional(out, metrics.target_scored_token_length);
  out << ",\n  \"target_token_score_count\": ";
  json_optional(out, metrics.target_token_score_count);
  out << ",\n  \"target_nonfinite_token_score_count\": ";
  json_optional(out, metrics.target_nonfinite_token_score_count);
  out << ",\n  \"target_token_scores_status\": " << json_string(metrics.target_token_scores_status);
  out << ",\n  \"engine_create_ms\": ";
  json_optional(out, metrics.engine_create_ms);
  out << ",\n  \"prefill_wall_ms\": ";
  json_optional(out, metrics.prefill_wall_ms);
  out << ",\n  \"decode_wall_ms\": ";
  json_optional(out, metrics.decode_wall_ms);
  out << ",\n  \"score_wall_ms\": ";
  json_optional(out, metrics.score_wall_ms);
  out << ",\n  \"generation_wall_ms\": ";
  json_optional(out, metrics.generation_wall_ms);
  out << ",\n  \"time_to_first_token_s\": ";
  json_optional(out, metrics.time_to_first_token_s);
  out << ",\n  \"init_benchmark_s\": ";
  json_optional(out, metrics.init_benchmark_s);
  out << ",\n  \"prefill_token_counts\": ";
  json_array(out, metrics.prefill_tokens);
  out << ",\n  \"decode_token_counts\": ";
  json_array(out, metrics.decode_tokens);
  out << ",\n  \"prefill_tokens_per_s\": ";
  json_array(out, metrics.prefill_tokens_per_s);
  out << ",\n  \"decode_tokens_per_s\": ";
  json_array(out, metrics.decode_tokens_per_s);
  out << "\n}\n";
  write_binary(options.output_prefix + ".metrics.json", out.str());
}

int positive_int(const std::string& value, const std::string& name) {
  size_t consumed = 0;
  int parsed = 0;
  try {
    parsed = std::stoi(value, &consumed);
  } catch (const std::exception&) {
    throw std::runtime_error("invalid " + name + ": " + value);
  }
  if (consumed != value.size() || parsed <= 0)
    throw std::runtime_error("invalid " + name + ": " + value);
  return parsed;
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string flag = argv[i];
    if (flag == "--force-f32") {
      options.force_f32 = true;
      continue;
    }
    if (i + 1 == argc) throw std::runtime_error("missing value for " + flag);
    const std::string value = argv[++i];
    if (flag == "--model") options.model = value;
    else if (flag == "--prompt") options.prompt = value;
    else if (flag == "--score-target") options.score_target = value;
    else if (flag == "--output-prefix") options.output_prefix = value;
    else if (flag == "--backend") options.backend = value;
    else if (flag == "--sampler-backend") options.sampler_backend = value;
    else if (flag == "--max-context") options.max_context = positive_int(value, flag);
    else if (flag == "--max-output") options.max_output = positive_int(value, flag);
    else throw std::runtime_error("unknown option: " + flag);
  }
  if (options.model.empty() || options.prompt.empty() || options.output_prefix.empty() ||
      (options.backend != "cpu" && options.backend != "gpu") ||
      (options.sampler_backend != "default" && options.sampler_backend != "cpu")) {
    throw std::runtime_error(
        "usage: probe --model MODEL --prompt PROMPT --output-prefix PREFIX "
        "--backend cpu|gpu [--force-f32] [--sampler-backend default|cpu] "
        "[--score-target TARGET.txt] "
        "[--max-context 8192] [--max-output 2048]");
  }
  if (options.max_output > options.max_context)
    throw std::runtime_error("max output cannot exceed max context");
  return options;
}

template <typename T, void (*Delete)(T*)>
using Handle = std::unique_ptr<T, decltype(Delete)>;

void write_ids(const std::string& path, const int* tokens, size_t count) {
  std::ostringstream out;
  for (size_t i = 0; i < count; ++i) out << tokens[i] << '\n';
  write_binary(path, out.str());
}

void collect_benchmark(LiteRtLmSession* session, Metrics& metrics) {
  Handle<LiteRtLmBenchmarkInfo, litert_lm_benchmark_info_delete> info(
      litert_lm_session_get_benchmark_info(session), litert_lm_benchmark_info_delete);
  if (!info) return;
  metrics.time_to_first_token_s = litert_lm_benchmark_info_get_time_to_first_token(info.get());
  metrics.init_benchmark_s = litert_lm_benchmark_info_get_total_init_time_in_second(info.get());
  const int prefill_turns = litert_lm_benchmark_info_get_num_prefill_turns(info.get());
  const int decode_turns = litert_lm_benchmark_info_get_num_decode_turns(info.get());
  for (int i = 0; i < prefill_turns; ++i) {
    metrics.prefill_tokens.push_back(litert_lm_benchmark_info_get_prefill_token_count_at(info.get(), i));
    metrics.prefill_tokens_per_s.push_back(
        litert_lm_benchmark_info_get_prefill_tokens_per_sec_at(info.get(), i));
  }
  for (int i = 0; i < decode_turns; ++i) {
    metrics.decode_tokens.push_back(litert_lm_benchmark_info_get_decode_token_count_at(info.get(), i));
    metrics.decode_tokens_per_s.push_back(
        litert_lm_benchmark_info_get_decode_tokens_per_sec_at(info.get(), i));
  }
}

void run(const Options& options, Metrics& metrics) {
  litert_lm_set_min_log_level(kLiteRtLmLogSeverityInfo);
  metrics.stage = "read_prompt";
  const std::string prompt = read_binary(options.prompt);
  if (prompt.empty()) throw std::runtime_error("prompt file is empty");
  if (prompt.find('\0') != std::string::npos)
    throw std::runtime_error("prompt contains NUL byte; the C API expects UTF-8 text");
  metrics.prompt_bytes = prompt.size();
  std::string score_target;
  if (!options.score_target.empty()) {
    metrics.stage = "read_score_target";
    score_target = read_binary(options.score_target);
    if (score_target.empty()) throw std::runtime_error("score target file is empty");
    if (score_target.find('\0') != std::string::npos)
      throw std::runtime_error("score target contains NUL byte; the C API expects UTF-8 text");
    metrics.score_target_bytes = score_target.size();
    write_binary(options.output_prefix + ".scored_target.txt", score_target);
  }

  metrics.stage = "create_engine";
  Handle<LiteRtLmEngineSettings, litert_lm_engine_settings_delete> settings(
      litert_lm_engine_settings_create(options.model.c_str(), options.backend.c_str(), nullptr, nullptr),
      litert_lm_engine_settings_delete);
  if (!settings) throw std::runtime_error("engine settings creation returned null");
  const std::string cache_dir = options.output_prefix + ".cache";
  if (mkdir(cache_dir.c_str(), 0700) != 0 && errno != EEXIST)
    throw std::runtime_error("cannot create cache directory: " + cache_dir + ": " + std::strerror(errno));
  litert_lm_engine_settings_set_cache_dir(settings.get(), cache_dir.c_str());
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), options.max_context);
  litert_lm_engine_settings_set_enable_speculative_decoding(settings.get(), false);
  litert_lm_engine_settings_enable_benchmark(settings.get());
  if (options.force_f32) {
    litert_lm_engine_settings_set_activation_data_type(
        settings.get(), kLiteRtLmActivationDataTypeFloat32);
  }
  if (options.sampler_backend == "cpu") setCpuSamplerEngine(settings.get());
  const auto engine_start = Clock::now();
  Handle<LiteRtLmEngine, litert_lm_engine_delete> engine(
      litert_lm_engine_create(settings.get()), litert_lm_engine_delete);
  metrics.engine_create_ms = elapsed_ms(engine_start, Clock::now());
  if (!engine) throw std::runtime_error("engine creation returned null; inspect LiteRT-LM stderr/logcat");

  metrics.stage = "tokenize_prompt";
  Handle<LiteRtLmTokenizeResult, litert_lm_tokenize_result_delete> input_tokens(
      litert_lm_engine_tokenize(engine.get(), prompt.c_str()), litert_lm_tokenize_result_delete);
  if (input_tokens) {
    const size_t count = litert_lm_tokenize_result_get_num_tokens(input_tokens.get());
    const int* ids = litert_lm_tokenize_result_get_tokens(input_tokens.get());
    metrics.input_tokenizer_count = count;
    if (ids || count == 0) write_ids(options.output_prefix + ".input_token_ids.txt", ids, count);
  }
  if (!options.score_target.empty()) {
    metrics.stage = "tokenize_score_target";
    Handle<LiteRtLmTokenizeResult, litert_lm_tokenize_result_delete> target_tokens(
        litert_lm_engine_tokenize(engine.get(), score_target.c_str()), litert_lm_tokenize_result_delete);
    if (target_tokens) {
      const size_t count = litert_lm_tokenize_result_get_num_tokens(target_tokens.get());
      const int* ids = litert_lm_tokenize_result_get_tokens(target_tokens.get());
      metrics.target_tokenizer_count = count;
      if (ids || count == 0)
        write_ids(options.output_prefix + ".target_token_ids.txt", ids, count);
    }
  }

  metrics.stage = "create_session";
  Handle<LiteRtLmSamplerParams, litert_lm_sampler_params_delete> sampler(
      litert_lm_sampler_params_create(kLiteRtLmSamplerTypeTopP), litert_lm_sampler_params_delete);
  if (!sampler) throw std::runtime_error("top-p sampler creation returned null");
  litert_lm_sampler_params_set_top_k(sampler.get(), 1);
  litert_lm_sampler_params_set_top_p(sampler.get(), 1.0f);
  litert_lm_sampler_params_set_temperature(sampler.get(), 0.0f);
  litert_lm_sampler_params_set_seed(sampler.get(), 42);
  Handle<LiteRtLmSessionConfig, litert_lm_session_config_delete> config(
      litert_lm_session_config_create(), litert_lm_session_config_delete);
  if (!config) throw std::runtime_error("session config creation returned null");
  litert_lm_session_config_set_max_output_tokens(config.get(), options.max_output);
  litert_lm_session_config_set_apply_prompt_template(config.get(), false);
  litert_lm_session_config_set_sampler_params(config.get(), sampler.get());
  if (options.sampler_backend == "cpu") setCpuSamplerSession(config.get());
  Handle<LiteRtLmSession, litert_lm_session_delete> session(
      litert_lm_engine_create_session(engine.get(), config.get()), litert_lm_session_delete);
  if (!session) throw std::runtime_error("session creation returned null; inspect LiteRT-LM stderr/logcat");
  Handle<LiteRtLmInputData, litert_lm_input_data_delete> input(
      litert_lm_input_data_create(kLiteRtLmInputDataTypeText, prompt.data(), prompt.size()),
      litert_lm_input_data_delete);
  if (!input) throw std::runtime_error("text input creation returned null");
  const LiteRtLmInputData* inputs[] = {input.get()};

  metrics.stage = "prefill";
  const auto generate_start = Clock::now();
  const int prefill_status = litert_lm_session_run_prefill(session.get(), inputs, 1);
  const auto prefill_end = Clock::now();
  metrics.prefill_wall_ms = elapsed_ms(generate_start, prefill_end);
  if (prefill_status != 0)
    throw std::runtime_error("prefill failed with C API status " + std::to_string(prefill_status));

  if (!options.score_target.empty()) {
    metrics.stage = "text_scoring";
    metrics.target_score_status = "unavailable";
    metrics.target_token_scores_status = "unavailable";
    const char* targets[] = {score_target.c_str()};
    Handle<LiteRtLmResponses, litert_lm_responses_delete> scored(
        litert_lm_session_run_text_scoring(session.get(), targets, 1, true),
        litert_lm_responses_delete);
    metrics.score_wall_ms = elapsed_ms(prefill_end, Clock::now());
    if (!scored)
      throw std::runtime_error("text scoring returned null; this backend/model may not support scoring (inspect LiteRT-LM stderr/logcat)");
    metrics.candidates = litert_lm_responses_get_num_candidates(scored.get());
    if (metrics.candidates != 1)
      throw std::runtime_error("text scoring returned " + std::to_string(metrics.candidates) + " candidates; expected one");
    if (!litert_lm_responses_has_score_at(scored.get(), 0))
      throw std::runtime_error("text scoring returned no aggregate score for target zero");
    metrics.target_score = litert_lm_responses_get_score_at(scored.get(), 0);
    metrics.target_score_status = std::isfinite(*metrics.target_score) ? "available" : "nonfinite";
    if (litert_lm_responses_has_token_length_at(scored.get(), 0))
      metrics.target_scored_token_length = litert_lm_responses_get_token_length_at(scored.get(), 0);
    if (litert_lm_responses_has_token_scores_at(scored.get(), 0)) {
      const int count = litert_lm_responses_get_num_token_scores_at(scored.get(), 0);
      const float* scores = litert_lm_responses_get_token_scores_at(scored.get(), 0);
      if (count < 0 || (count > 0 && !scores))
        throw std::runtime_error("text scoring returned inconsistent token-score count/pointer");
      metrics.target_token_score_count = count;
      metrics.target_token_scores_status = count > 0 ? "available" : "empty";
      int nonfinite = 0;
      std::ostringstream out;
      out << std::setprecision(9);
      for (int i = 0; i < count; ++i) {
        if (!std::isfinite(scores[i])) ++nonfinite;
        out << scores[i] << '\n';
      }
      metrics.target_nonfinite_token_score_count = nonfinite;
      write_binary(options.output_prefix + ".target_token_scores.txt", out.str());
    }
    metrics.stage = "collect_metrics";
    collect_benchmark(session.get(), metrics);
    metrics.stage = "complete";
    metrics.status = "ok";
    return;
  }

  metrics.stage = "decode";
  Handle<LiteRtLmResponses, litert_lm_responses_delete> responses(
      litert_lm_session_run_decode(session.get()), litert_lm_responses_delete);
  const auto decode_end = Clock::now();
  metrics.decode_wall_ms = elapsed_ms(prefill_end, decode_end);
  metrics.generation_wall_ms = elapsed_ms(generate_start, decode_end);
  if (!responses) throw std::runtime_error("decode returned null; inspect LiteRT-LM stderr/logcat");
  metrics.candidates = litert_lm_responses_get_num_candidates(responses.get());
  if (metrics.candidates < 1) throw std::runtime_error("decode returned zero candidates");
  const char* response = litert_lm_responses_get_response_text_at(responses.get(), 0);
  if (!response) throw std::runtime_error("candidate zero contains no response text");
  const std::string output(response);
  metrics.output_bytes = output.size();
  write_binary(options.output_prefix + ".raw.txt", output);
  if (litert_lm_responses_has_token_length_at(responses.get(), 0))
    metrics.response_token_length = litert_lm_responses_get_token_length_at(responses.get(), 0);
  if (litert_lm_responses_has_score_at(responses.get(), 0))
    metrics.response_score = litert_lm_responses_get_score_at(responses.get(), 0);
  if (litert_lm_responses_has_token_scores_at(responses.get(), 0)) {
    const int count = litert_lm_responses_get_num_token_scores_at(responses.get(), 0);
    const float* scores = litert_lm_responses_get_token_scores_at(responses.get(), 0);
    if (count > 0 && scores) {
      std::ostringstream out;
      out << std::setprecision(9);
      for (int i = 0; i < count; ++i) out << scores[i] << '\n';
      write_binary(options.output_prefix + ".selected_token_scores.txt", out.str());
      metrics.selected_token_score_count = count;
    }
  }

  metrics.stage = "collect_metrics";
  collect_benchmark(session.get(), metrics);
  Handle<LiteRtLmTokenizeResult, litert_lm_tokenize_result_delete> output_tokens(
      litert_lm_engine_tokenize(engine.get(), output.c_str()), litert_lm_tokenize_result_delete);
  if (output_tokens) {
    const size_t count = litert_lm_tokenize_result_get_num_tokens(output_tokens.get());
    const int* ids = litert_lm_tokenize_result_get_tokens(output_tokens.get());
    metrics.output_retokenized_count = count;
    if (ids || count == 0) write_ids(options.output_prefix + ".output_retokenized_ids.txt", ids, count);
  }
  metrics.stage = "complete";
  metrics.status = "ok";
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  try {
    options = parse_options(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 2;
  }
  Metrics metrics;
  try {
    run(options, metrics);
  } catch (const std::exception& e) {
    metrics.error = e.what();
    std::cerr << "probe failed at " << metrics.stage << ": " << metrics.error << '\n';
  }
  try {
    write_metrics(options, metrics);
  } catch (const std::exception& e) {
    std::cerr << "cannot save metrics: " << e.what() << '\n';
    return 3;
  }
  if (metrics.status != "ok") return 1;
  if (options.score_target.empty())
    std::cout << "saved " << options.output_prefix << ".raw.txt and .metrics.json\n";
  else
    std::cout << "saved " << options.output_prefix << ".scored_target.txt and .metrics.json\n";
  return 0;
}
