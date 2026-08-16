#include "whisper/include/whisper.h"

#define DR_WAV_IMPLEMENTATION
#include "whisper/examples/dr_wav.h"

#include <TargetConditionals.h>
#include <cmath>
#include <fstream>
#include <cstdio>
#include <string>
#include <thread>
#include <mutex>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <vector>

#include <iostream>
#include "json/json.hpp"
#include <stdio.h>

using json = nlohmann::json;

void print(std::string value)
{
    std::cout << value << std::endl;
}

char *jsonToChar(json jsonData)
{
    // Whisper can emit text that splits a multi-byte UTF-8 character at a
    // token boundary; dump() would throw type_error.316 and abort the app
    // across the FFI boundary. Replace invalid bytes with U+FFFD instead.
    std::string result =
        jsonData.dump(-1, ' ', false, json::error_handler_t::replace);
    // malloc, not new[]: callers across the FFI boundary free this
    // with the C allocator (Dart's malloc.free).
    char *ch = (char *)malloc(result.size() + 1);
    strcpy(ch, result.c_str());
    return ch;
}

std::string charToString(char *value)
{
    std::string result(value);
    return result;
}

char *stringToChar(std::string value)
{
    char *ch = (char *)malloc(value.size() + 1);
    strcpy(ch, value.c_str());
    return ch;
}

// //  500 -> 00:05.000
// // 6000 -> 01:00.000
// std::string to_timestamp(int64_t t)
// {
//     int64_t sec = t / 100;
//     int64_t msec = t - sec * 100;
//     int64_t min = sec / 60;
//     sec = sec - min * 60;

//     char buf[32];
//     snprintf(buf, sizeof(buf), "%02d:%02d.%03d", (int)min, (int)sec, (int)msec);

//     return std::string(buf);
// }

// Terminal color map. 10 colors grouped in ranges [0.0, 0.1, ..., 0.9]
// Lowest is red, middle is yellow, highest is green.
const std::vector<std::string> k_colors = {
    "\033[38;5;196m",
    "\033[38;5;202m",
    "\033[38;5;208m",
    "\033[38;5;214m",
    "\033[38;5;220m",
    "\033[38;5;226m",
    "\033[38;5;190m",
    "\033[38;5;154m",
    "\033[38;5;118m",
    "\033[38;5;82m",
};

//  500 -> 00:05.000
// 6000 -> 01:00.000
std::string to_timestamp(int64_t t, bool comma = false)
{
    int64_t msec = t * 10;
    int64_t hr = msec / (1000 * 60 * 60);
    msec = msec - hr * (1000 * 60 * 60);
    int64_t min = msec / (1000 * 60);
    msec = msec - min * (1000 * 60);
    int64_t sec = msec / 1000;
    msec = msec - sec * 1000;

    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d%s%03d", (int)hr, (int)min, (int)sec, comma ? "," : ".", (int)msec);

    return std::string(buf);
}

int timestamp_to_sample(int64_t t, int n_samples)
{
    return std::max(0, std::min((int)n_samples - 1, (int)((t * WHISPER_SAMPLE_RATE) / 100)));
}

// command-line parameters
struct whisper_params
{
    int32_t seed = -1; // RNG seed, not used currently
    int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());

    int32_t n_processors = 1;
    int32_t offset_t_ms = 0;
    int32_t offset_n = 0;
    int32_t duration_ms = 0;
    int32_t max_context = -1;
    int32_t max_len = 0;
    int32_t best_of = 5;
    int32_t beam_size = -1;

    float word_thold = 0.01f;
    float entropy_thold = 2.40f;
    float logprob_thold = -1.00f;

    bool verbose = false;
    bool print_special_tokens = false;
    bool speed_up = false;
    bool translate = false;
    bool diarize = false;
    bool no_fallback = false;
    bool output_txt = false;
    bool output_vtt = false;
    bool output_srt = false;
    bool output_wts = false;
    bool output_csv = false;
    bool print_special = false;
    bool print_colors = false;
    bool print_progress = false;
    bool no_timestamps = false;
    bool split_on_word = false;
    // whisper_full_params.no_context: disable cross-segment text conditioning.
    bool no_context = false;
    // whisper_full_params.suppress_non_speech_tokens: drop [BLANK_AUDIO]-style annotations.
    bool suppress_nst = false;

    // Address of a Dart NativeCallable<Void Function(Int32)>; 0 = none.
    uint64_t progress_cb_addr = 0;

    // Park the loaded model in g_model_cache after this request instead of
    // freeing it, so the next request with the same model file skips the
    // multi-second load (issue #26). Off = load-per-request, as always.
    bool keep_model_loaded = false;

    std::string language = "id";
    std::string prompt;
    std::string model = "models/ggml-model-whisper-small.bin";
    std::string audio = "samples/jfk.wav";
    std::vector<std::string> fname_inp = {};
    std::vector<std::string> fname_outp = {};
};

struct whisper_print_user_data
{
    const whisper_params *params;

    const std::vector<std::vector<float>> *pcmf32s;
};

// ---------------------------------------------------------------------------
// Resident model cache (issue #26). transcribe() normally frees its context
// when the request finishes, so every request pays the full model load
// (seconds for the small models and up). A request with keep_model_loaded
// parks its context here instead; the next request with the same model path
// picks it up and skips the load.
//
// Checkout semantics: a request *takes* the parked context (the slot goes
// empty) and parks it again when done, so the mutex is held only for the
// swap — never during a decode. Concurrent requests keep running in
// parallel, each on its own context, exactly as without the cache. The lock
// matters because Dart issues requests from short-lived worker isolates,
// i.e. from changing threads.
// ---------------------------------------------------------------------------
static struct
{
    struct whisper_context *ctx = nullptr; // parked context; nullptr = empty
    std::string model;                     // model path ctx was loaded from
    std::mutex mutex;
} g_model_cache;

json release_model()
{
    struct whisper_context *parked = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_model_cache.mutex);
        parked = g_model_cache.ctx;
        g_model_cache.ctx = nullptr;
        g_model_cache.model.clear();
    }
    if (parked != nullptr)
    {
        whisper_free(parked);
    }
    json jsonResult;
    jsonResult["@type"] = "releaseModel";
    jsonResult["released"] = parked != nullptr;
    return jsonResult;
}

json transcribe(json jsonBody)
{
    whisper_params params;

    params.n_threads = jsonBody["threads"];
    params.verbose = jsonBody["is_verbose"];
    params.translate = jsonBody["is_translate"];
    params.language = jsonBody["language"];
    params.print_special_tokens = jsonBody["is_special_tokens"];
    params.no_timestamps = jsonBody["is_no_timestamps"];
    params.model = jsonBody["model"];
    params.audio = jsonBody["audio"];
    params.split_on_word = jsonBody["split_on_word"];
    params.diarize = jsonBody["diarize"];

    // Optional fields: absent / null / empty leaves whisper.cpp defaults.
    if (jsonBody.contains("initial_prompt") && jsonBody["initial_prompt"].is_string())
    {
        params.prompt = jsonBody["initial_prompt"].get<std::string>();
    }
    if (jsonBody.contains("no_context") && jsonBody["no_context"].is_boolean())
    {
        params.no_context = jsonBody["no_context"].get<bool>();
    }
    if (jsonBody.contains("suppress_non_speech_tokens") && jsonBody["suppress_non_speech_tokens"].is_boolean())
    {
        params.suppress_nst = jsonBody["suppress_non_speech_tokens"].get<bool>();
    }
    if (jsonBody.contains("progress_callback") && jsonBody["progress_callback"].is_number_unsigned())
    {
        params.progress_cb_addr = jsonBody["progress_callback"].get<uint64_t>();
    }
    if (jsonBody.contains("keep_model_loaded") && jsonBody["keep_model_loaded"].is_boolean())
    {
        params.keep_model_loaded = jsonBody["keep_model_loaded"].get<bool>();
    }
    json jsonResult;
    jsonResult["@type"] = "transcribe";

    if (whisper_lang_id(params.language.c_str()) == -1)
    {
        jsonResult["@type"] = "error";
        jsonResult["message"] = "error: unknown language";
        return jsonResult;
    }

    if (params.seed < 0)
    {
        params.seed = time(NULL);
    }

    // whisper init: take the parked context when the model path matches
    // (leaving the cache empty while it is in use), otherwise load from disk.
    bool reused_ctx = false;
    struct whisper_context *ctx = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_model_cache.mutex);
        if (g_model_cache.ctx != nullptr && g_model_cache.model == params.model)
        {
            ctx = g_model_cache.ctx;
            g_model_cache.ctx = nullptr;
            g_model_cache.model.clear();
            reused_ctx = true;
        }
    }
    if (ctx == nullptr)
    {
        struct whisper_context_params cparams = whisper_context_default_params();
#if TARGET_OS_SIMULATOR
        // ggml's Metal sources carry no simulator guard of their own, so
        // the simulator's Metal-to-host translation shim (MTLSimDevice)
        // gets the real init call and dies inside it during model-weight
        // upload (confirmed: EXC_BREAKPOINT, "API Misuse", reproduced live
        // - see the window-management investigation). No such shim exists
        // on a device, so this is simulator-only; guarding it here
        // restores CPU-only simulator builds, which is what the fixture
        // harness (integration_test/whisper_fixture_test.dart) actually
        // needs to run on simulator at all.
        cparams.use_gpu = false;
#else
        cparams.use_gpu = true; // falls back to CPU if init fails
#endif
        ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);
    }
    if (ctx == nullptr)
    {
        // Without this check a missing/corrupt model file crashes in
        // whisper_full instead of surfacing an error response.
        jsonResult["@type"] = "error";
        jsonResult["message"] = "failed to load model " + params.model;
        return jsonResult;
    }
    // Park or free the context on every exit path (including WAV validation
    // errors: a bad file must not cost the next request a model reload).
    auto release_ctx = [&]() noexcept
    {
        if (!params.keep_model_loaded)
        {
            whisper_free(ctx);
            return;
        }
        struct whisper_context *displaced = nullptr;
        {
            std::lock_guard<std::mutex> lock(g_model_cache.mutex);
            displaced = g_model_cache.ctx;
            g_model_cache.ctx = ctx;
            g_model_cache.model = params.model;
        }
        // A concurrent request may have parked its own context while this
        // one was decoding; only one can stay resident.
        if (displaced != nullptr)
        {
            whisper_free(displaced);
        }
    };

    std::string text_result = "";
    // for (int f = 0; f < (int)params.fname_inp.size(); ++f)
    // {
    const auto fname_inp = params.audio;
    // WAV input
    std::vector<float> pcmf32;
    {
        drwav wav;
        if (!drwav_init_file(&wav, fname_inp.c_str(), NULL))
        {
            jsonResult["@type"] = "error";
            jsonResult["message"] = " failed to open WAV file ";
            release_ctx();
            return jsonResult;
        }

        if (wav.channels != 1 && wav.channels != 2)
        {
            jsonResult["@type"] = "error";
            jsonResult["message"] = "must be mono or stereo";
            drwav_uninit(&wav);
            release_ctx();
            return jsonResult;
        }

        if (wav.sampleRate != WHISPER_SAMPLE_RATE)
        {
            jsonResult["@type"] = "error";
            jsonResult["message"] = "WAV file  must be 16 kHz";
            drwav_uninit(&wav);
            release_ctx();
            return jsonResult;
        }

        if (wav.bitsPerSample != 16)
        {
            jsonResult["@type"] = "error";
            jsonResult["message"] = "WAV file  must be 16 bit";
            drwav_uninit(&wav);
            release_ctx();
            return jsonResult;
        }

        int n = wav.totalPCMFrameCount;

        std::vector<int16_t> pcm16;
        pcm16.resize(n * wav.channels);
        drwav_read_pcm_frames_s16(&wav, n, pcm16.data());
        drwav_uninit(&wav);

        // convert to mono, float
        pcmf32.resize(n);
        if (wav.channels == 1)
        {
            for (int i = 0; i < n; i++)
            {
                pcmf32[i] = float(pcm16[i]) / 32768.0f;
            }
        }
        else
        {
            for (int i = 0; i < n; i++)
            {
                pcmf32[i] = float(pcm16[2 * i] + pcm16[2 * i + 1]) / 65536.0f;
            }
        }
    }

    // print some info about the processing
    {
        // printf("\n");
        if (!whisper_is_multilingual(ctx))
        {
            if (params.language != "en" || params.translate)
            {
                params.language = "en";
                params.translate = false;
                // printf("%s: WARNING: model is not multilingual, ignoring language and translation options\n", __func__);
            }
        }
        // printf("%s: processing '%s' (%d samples, %.1f sec), %d threads, lang = %s, task = %s, timestamps = %d ...\n",
        //        __func__, fname_inp.c_str(), int(pcmf32.size()), float(pcmf32.size()) / WHISPER_SAMPLE_RATE, params.n_threads,
        //        params.language.c_str(),
        //        params.translate ? "translate" : "transcribe",
        //        params.no_timestamps ? 0 : 1);
        // printf("\n");
    }
    // run the inference
    {
        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

        wparams.print_realtime = false;
        wparams.print_progress = false;
        wparams.print_timestamps = !params.no_timestamps;
        // wparams.print_special_tokens = params.print_special_tokens;
        wparams.translate = params.translate;
        wparams.language = params.language.c_str();
        wparams.n_threads = params.n_threads;
        wparams.split_on_word = params.split_on_word;
        // tinydiarize speaker-turn detection; needs a *-tdrz model to
        // emit turns, harmless with regular models.
        wparams.tdrz_enable = params.diarize;

        // params.prompt outlives whisper_full(), so the pointer stays valid.
        if (!params.prompt.empty()) {
            wparams.initial_prompt = params.prompt.c_str();
        }
        // A reused context still holds the previous request's decoded text
        // as conditioning history (whisper clears prompt_past only when
        // no_context is set), so force the clear on reuse: a warm request
        // must transcribe exactly like a cold one. initial_prompt is
        // unaffected — whisper re-applies it after the clear.
        wparams.no_context = params.no_context || reused_ctx;
        wparams.suppress_nst = params.suppress_nst;

        if (params.split_on_word) {
            wparams.max_len = 1;
            wparams.token_timestamps = true;
        }

        if (params.progress_cb_addr) {
            // NativeCallable.listener is safe to invoke from whisper's
            // worker thread: it posts to the owning Dart isolate.
            wparams.progress_callback = [](struct whisper_context *, struct whisper_state *, int progress, void * user_data) {
                ((void (*)(int32_t))user_data)((int32_t)progress);
            };
            wparams.progress_callback_user_data = (void *)(uintptr_t)params.progress_cb_addr;
        }

        if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0)
        {
            jsonResult["@type"] = "error";
            jsonResult["message"] = "failed to process audio";
            release_ctx();
            return jsonResult;
        }

        

        // print result;
        if (!wparams.print_realtime)
        {

            const int n_segments = whisper_full_n_segments(ctx);

            std::vector<json> segmentsJson = {};

            for (int i = 0; i < n_segments; ++i)
            {
                const char *text = whisper_full_get_segment_text(ctx, i);

                std::string str(text);
                text_result += str;
                if (params.no_timestamps)
                {
                    // printf("%s", text);
                    // fflush(stdout);
                } else {
                    json jsonSegment;
                    const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
                    const int64_t t1 = whisper_full_get_segment_t1(ctx, i);

                    // printf("[%s --> %s]  %s\n", to_timestamp(t0).c_str(), to_timestamp(t1).c_str(), text);

                    jsonSegment["from_ts"] = t0;
                    jsonSegment["to_ts"] = t1;
                    jsonSegment["text"] = text;
                    jsonSegment["speaker_turn_next"] =
                        whisper_full_get_segment_speaker_turn_next(ctx, i);

                    segmentsJson.push_back(jsonSegment);
                }
            }

            if (!params.no_timestamps) {
                jsonResult["segments"] = segmentsJson;
            }
        }
    }
    jsonResult["text"] = text_result;
    jsonResult["backend"] = whisper_backend_name(ctx);

    release_ctx();
    return jsonResult;
}

extern "C"
{
    char *request(char *body)
    {
        json jsonBody = json::parse(body);
        json jsonResult;

        if (jsonBody["@type"] == "getTextFromWavFile")
        {
            return jsonToChar(transcribe(jsonBody));
        }

        if (jsonBody["@type"] == "releaseModel")
        {
            return jsonToChar(release_model());
        }

        if (jsonBody["@type"] == "getVersion")
        {
            jsonResult["@type"] = "version";
            jsonResult["message"] = "version lib v0.0.0";
            return jsonToChar(jsonResult);
        }

        jsonResult["@type"] = "error";
        jsonResult["message"] = "method not found";
        return jsonToChar(jsonResult);
    }

    int main()
    {
        json jsonBody;
        jsonBody["@type"] = "al";
        print(transcribe(jsonBody).dump());
        return 0;
    }
}

// ---------------------------------------------------------------------------
// Live (streaming) transcription.
//
// Unlike request()/transcribe(), which load the model on every call, a stream
// keeps one whisper_context alive for the whole session:
//
//   stream_start(json)          -> loads the model (or borrows a parked one
//                                  from g_model_cache), resets state
//   stream_feed(pcm, n)         -> appends 16 kHz mono float samples; re-runs
//                                  inference when >= ~1.5 s of new audio has
//                                  accumulated and returns the partial text
//   stream_stop()               -> final text; frees the context, or parks it
//                                  in g_model_cache when the session was
//                                  started with keep_model_loaded or borrowed
//                                  a parked context (issue #26)
//
// Partials re-decode the whole current window with no_context = true, so a
// wrong early partial does not condition later ones. When the window grows
// past ~25 s its text is committed and the buffer restarts, keeping memory
// and inference time bounded (a word straddling the commit boundary may be
// clipped — acceptable for a draft).
//
// An RMS energy gate tracks the last voiced sample: inference only runs
// when new voiced audio has arrived, and the decoded window is trimmed
// shortly after the last voiced sample. Without this, whisper hallucinates
// over trailing silence (repeating earlier text or inventing phrases).
// ---------------------------------------------------------------------------

struct whisper_stream_state
{
    struct whisper_context *ctx = nullptr;
    std::vector<float> pcmf32;   // samples of the current window
    size_t n_transcribed = 0;    // window samples covered by the last run
    size_t n_voiced = 0;         // end of the last chunk with speech energy
    float noise_floor = 0.005f;  // adaptive ambient RMS estimate
    float gate_rms_min = 0.0015f;   // absolute minimum speech RMS
    float gate_ratio = 2.5f;        // voiced thold = ratio * noise_floor
    float gate_floor_cap = 0.01f;   // noise_floor cap (loud rooms)
    std::string committed;       // text of windows already committed
    std::string last_text;       // text of the current window's last run
    std::string language = "en";
    std::string prompt;
    std::string model;           // model path ctx was loaded from
    int n_threads = 4;
    bool translate = false;
    bool suppress_nst = false;
    // Park ctx into g_model_cache when the session ends: set when the
    // session asked for keep_model_loaded or borrowed a parked context.
    bool park_on_stop = false;
    // Test-only window/step overrides (0 = use the production default).
    // Set from an optional stream_start field never sent by production Dart
    // code (startWhisperLiveSession does not expose it) - existed so a test
    // can make an 11-second fixture cross several window boundaries instead
    // of needing a multi-minute one. See whisper_ggml's own test suite.
    size_t test_window_samples = 0;
    size_t test_step_samples = 0;
    std::mutex mutex;
};

static whisper_stream_state g_stream;

// Hand the session context back: park it in g_model_cache when the session
// asked for that (keep_model_loaded) or borrowed the context from there,
// free it otherwise. Caller must hold g_stream.mutex; lock order is
// g_stream.mutex -> g_model_cache.mutex, never the reverse anywhere.
static void stream_dispose_ctx()
{
    if (g_stream.ctx == nullptr) {
        return;
    }
    if (g_stream.park_on_stop) {
        struct whisper_context *displaced = nullptr;
        {
            std::lock_guard<std::mutex> lock(g_model_cache.mutex);
            displaced = g_model_cache.ctx;
            g_model_cache.ctx = g_stream.ctx;
            g_model_cache.model = g_stream.model;
        }
        if (displaced != nullptr) {
            whisper_free(displaced);
        }
    } else {
        whisper_free(g_stream.ctx);
    }
    g_stream.ctx = nullptr;
    g_stream.model.clear();
    g_stream.park_on_stop = false;
}

// A fixed-size sliding window (docs: window-management investigation,
// 2026-08-16): the decode window is capped at STREAM_WINDOW_SAMPLES on
// every call, for the whole session, not just within one commit cycle -
// unlike the previous design, where the window grew from 0 up to a
// 25-second commit threshold before resetting, so per-call decode cost
// grew across each cycle even though the ceiling was bounded. Capping
// every call keeps per-call cost roughly constant for the whole session.
// Starting point taken from whisper.cpp's own examples/stream/stream.cpp
// sliding-window defaults (step_ms/length_ms/keep_ms); tuned from there
// against the fixture harness's plateau benchmark, not assumed correct.
static const size_t STREAM_STEP_SAMPLES   = (size_t)(3.0 * WHISPER_SAMPLE_RATE);
static const size_t STREAM_WINDOW_SAMPLES = (size_t)(10.0 * WHISPER_SAMPLE_RATE);
// Audio kept across a window boundary rather than discarded, so a word
// spoken right at the boundary is not decoded with silence butted up
// against it on one side. Text-level de-dup (stream_run_inference) is
// still what actually prevents that kept audio's words from being
// committed twice - this alone is not a correctness guarantee, matching
// upstream's own "quick-n-dirty" caveat about relying on overlap alone.
static const size_t STREAM_KEEP_SAMPLES   = (size_t)(0.2 * WHISPER_SAMPLE_RATE);
// Decode this much audio past the last voiced sample (trailing consonants).
static const size_t STREAM_VOICE_PAD      = (size_t)(0.2 * WHISPER_SAMPLE_RATE);

// Resolved per-session: the test override when stream_start set one,
// otherwise the production default above. Caller must hold g_stream.mutex.
static size_t stream_step_samples()
{
    return g_stream.test_step_samples != 0
        ? g_stream.test_step_samples : STREAM_STEP_SAMPLES;
}
static size_t stream_window_samples()
{
    return g_stream.test_window_samples != 0
        ? g_stream.test_window_samples : STREAM_WINDOW_SAMPLES;
}

static std::string normalize_word(const std::string &w)
{
    std::string out;
    for (unsigned char c : w) {
        if (std::isalnum(c)) out += (char)std::tolower(c);
    }
    return out;
}

static std::vector<std::string> tokenize_words(const std::string &s)
{
    std::vector<std::string> words;
    std::string cur;
    for (unsigned char c : s) {
        if (std::isspace(c)) {
            if (!cur.empty()) { words.push_back(cur); cur.clear(); }
        } else {
            cur += (char)c;
        }
    }
    if (!cur.empty()) words.push_back(cur);
    return words;
}

// Strips leading words of `text` that duplicate the trailing words of
// `tail`, comparing case-insensitively and ignoring punctuation.
//
// A window boundary's audio cut is not always word-exact, and whisper can
// re-emit the word straddling it in both the just-committed decode and the
// next one - the kept-audio overlap (STREAM_KEEP_SAMPLES) does not prevent
// this, it only makes sure that word's audio is present to decode at all;
// upstream's own examples/stream/stream.cpp calls its keep_ms a mitigation
// for word-boundary issues, not a guarantee, and this is exactly that gap.
// Checked against real words rather than raw audio because the fixture
// harness's own comments note whisper's punctuation and casing move
// between builds and quantizations, so an exact-position audio check would
// be more fragile than this.
static std::string strip_duplicate_prefix(const std::string &tail, const std::string &text)
{
    const std::vector<std::string> tail_words = tokenize_words(tail);
    const std::vector<std::string> text_words = tokenize_words(text);
    if (tail_words.empty() || text_words.empty()) return text;

    const size_t max_check = std::min({tail_words.size(), text_words.size(), (size_t)5});
    size_t overlap = 0;
    for (size_t n = max_check; n >= 1; --n) {
        bool match = true;
        for (size_t i = 0; i < n; ++i) {
            if (normalize_word(tail_words[tail_words.size() - n + i]) !=
                normalize_word(text_words[i])) {
                match = false;
                break;
            }
        }
        if (match) { overlap = n; break; }
    }
    if (overlap == 0) return text;

    // Skip `overlap` whitespace-delimited words from the front, preserving
    // whatever spacing/punctuation the model produced in what remains
    // rather than re-joining tokens.
    size_t pos = 0;
    for (size_t skipped = 0; skipped < overlap && pos < text.size(); ++skipped) {
        while (pos < text.size() && std::isspace((unsigned char)text[pos])) pos++;
        while (pos < text.size() && !std::isspace((unsigned char)text[pos])) pos++;
    }
    return text.substr(pos);
}

// Runs whisper_full over the current window. Caller must hold g_stream.mutex.
static json stream_run_inference()
{
    json result;
    result["@type"] = "streamPartial";

    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.print_realtime   = false;
    wparams.print_progress   = false;
    wparams.print_timestamps = false;
    // NOTE (measured, not kept - see the window-management investigation):
    // upstream's stream.cpp sets this true in its own sliding-window mode,
    // but doing the same here defeats this function's own segment-based
    // commit boundary: forced to one segment per call, the per-segment
    // soft-cutoff test can never find a safe segment, so every boundary
    // falls through to the force-commit fallback and erases the whole
    // window every time - zero audio overlap survives, leaving
    // strip_duplicate_prefix as the only correctness net instead of one of
    // two. Measured against the fixture (synchronous feed, wall clock over
    // the full 11s clip): 9862ms without this, 10336ms with it - a single
    // run each, within normal run-to-run variance, but no speedup either,
    // so not worth the correctness downside above.
    // wparams.single_segment = true;
    wparams.translate        = g_stream.translate;
    wparams.language         = g_stream.language.c_str();
    wparams.n_threads        = g_stream.n_threads;
    wparams.no_context       = true;
    wparams.suppress_nst = g_stream.suppress_nst;
    if (!g_stream.prompt.empty()) {
        wparams.initial_prompt = g_stream.prompt.c_str();
    }

    // Trim trailing silence from the decode window; decoding it makes
    // whisper hallucinate (repeats or invented phrases). Also cap at the
    // window: never decode more than stream_window_samples() of audio in
    // one call, so per-call cost stays bounded for the whole session, not
    // just within one commit cycle (see the window-size comment above).
    const size_t window_samples = stream_window_samples();
    const size_t n_decode = std::min({
        g_stream.pcmf32.size(),
        g_stream.n_voiced + STREAM_VOICE_PAD,
        window_samples,
    });
    if (n_decode < (size_t)WHISPER_SAMPLE_RATE / 2) {
        result["text"] = g_stream.committed + g_stream.last_text;
        return result;
    }

    if (whisper_full(g_stream.ctx, wparams, g_stream.pcmf32.data(),
                     (int)n_decode) != 0) {
        result["@type"] = "error";
        result["message"] = "failed to process audio";
        return result;
    }

    g_stream.n_transcribed = n_decode;

    if (n_decode >= window_samples) {
        // Boundary reached. A segment counts as safe to commit only if it
        // ends at least STREAM_KEEP_SAMPLES before the end of this decode -
        // segments that close, whatever their own duration, keep the same
        // kind of continuity margin upstream's keep_ms gives at a fixed
        // audio boundary, without assuming every held segment is short.
        // `committed` is append-only from here on, exactly like the
        // Dart-side TranscriptAssembler's settled words: this is where
        // that one-way commitment actually happens, so it must not fire
        // on a segment whose audio has not stopped changing yet.
        //
        // Critically, the audio erased below must match the text
        // committed below exactly - erasing a fixed amount independent of
        // which segments were actually committed was the bug an earlier
        // version of this had: held text whose audio spans more than the
        // fixed keep window got silently discarded once that audio left
        // the buffer, because it was never re-decodable again and nothing
        // carried it forward. Erasing exactly through the last *committed*
        // segment's end guarantees every held segment's audio is still in
        // the buffer next cycle.
        const int64_t decode_end_cs =
            (int64_t)n_decode * 100 / WHISPER_SAMPLE_RATE;
        const int64_t keep_cs =
            (int64_t)STREAM_KEEP_SAMPLES * 100 / WHISPER_SAMPLE_RATE;
        const int64_t soft_cutoff_cs = decode_end_cs - keep_cs;

        std::string safe_text;
        std::string held_text;
        int64_t committed_through_cs = 0;
        bool any_committed = false;
        const int n_segments = whisper_full_n_segments(g_stream.ctx);
        for (int i = 0; i < n_segments; ++i) {
            std::string seg_text = whisper_full_get_segment_text(g_stream.ctx, i);
            if (i == 0) {
                seg_text = strip_duplicate_prefix(g_stream.committed, seg_text);
            }
            const int64_t t1 = whisper_full_get_segment_t1(g_stream.ctx, i);
            if (t1 <= soft_cutoff_cs) {
                safe_text += seg_text;
                committed_through_cs = t1;
                any_committed = true;
            } else {
                held_text += seg_text;
            }
        }

        // No segment cleared the keep margin (whisper decoded this whole
        // window as one continuous phrase with no internal break far
        // enough from the end). Erasing nothing here does NOT make this
        // a harmless no-op cycle: n_decode is capped at window_samples,
        // so the very next trigger would decode the identical leading
        // slice of pcmf32 again and hit the same "nothing safe" outcome
        // forever - audio keeps arriving, but the window never advances
        // to see it. Confirmed live: the fixture stalled permanently
        // exactly here before this fallback existed. Falling back to
        // "commit everything but the last segment" (or force-committing
        // the single segment if there is only one) guarantees the window
        // always advances; strip_duplicate_prefix on the next segment's
        // head is the safety net against the boundary landing mid-word.
        if (!any_committed && n_segments > 0) {
            for (int i = 0; i < n_segments - 1; ++i) {
                safe_text += whisper_full_get_segment_text(g_stream.ctx, i);
                committed_through_cs = whisper_full_get_segment_t1(g_stream.ctx, i);
                any_committed = true;
            }
            if (!any_committed) {
                // Exactly one segment spanning the whole window: commit it
                // in full rather than deadlock.
                safe_text = strip_duplicate_prefix(
                    g_stream.committed,
                    whisper_full_get_segment_text(g_stream.ctx, 0));
                committed_through_cs = decode_end_cs;
                any_committed = true;
                held_text.clear();
            } else {
                held_text = whisper_full_get_segment_text(g_stream.ctx, n_segments - 1);
            }
        }

        g_stream.last_text = held_text;
        if (any_committed) {
            g_stream.committed += safe_text;
            size_t n_erase = (size_t)(committed_through_cs * WHISPER_SAMPLE_RATE / 100);
            n_erase = std::min(n_erase, n_decode);
            g_stream.pcmf32.erase(g_stream.pcmf32.begin(),
                                  g_stream.pcmf32.begin() + n_erase);
            g_stream.n_transcribed = n_decode - n_erase;
            g_stream.n_voiced -= std::min(g_stream.n_voiced, n_erase);
        }
        // n_segments == 0 (silence somehow reached this far without
        // tripping the min-decode-length guard above): nothing to do:
        // n_transcribed stays at n_decode, waiting for real new audio.
    } else {
        std::string text;
        const int n_segments = whisper_full_n_segments(g_stream.ctx);
        for (int i = 0; i < n_segments; ++i) {
            std::string seg_text = whisper_full_get_segment_text(g_stream.ctx, i);
            if (i == 0) {
                seg_text = strip_duplicate_prefix(g_stream.committed, seg_text);
            }
            text += seg_text;
        }
        g_stream.last_text = text;
    }

    result["text"] = g_stream.committed + g_stream.last_text;
    return result;
}

extern "C"
{
    // body: {"model": path, "language": "en", "threads": 4,
    //        "is_translate": false, "initial_prompt": "...",
    //        "keep_model_loaded": false,
    //        "test_window_ms": 0, "test_step_ms": 0}
    // test_window_ms/test_step_ms are test-only (0 = production default):
    // never sent by startWhisperLiveSession, present so a test can shrink
    // the window/step thresholds and make a short fixture cross several
    // boundaries.
    char *stream_start(char *body)
    {
        std::lock_guard<std::mutex> lock(g_stream.mutex);
        json jsonResult;

        json jsonBody = json::parse(body, nullptr, false);
        if (jsonBody.is_discarded() || !jsonBody.contains("model")) {
            jsonResult["@type"] = "error";
            jsonResult["message"] = "stream_start: invalid request body";
            return jsonToChar(jsonResult);
        }

        if (g_stream.ctx != nullptr) {
            // Dispose per the *previous* session's policy before the new
            // session's fields overwrite it.
            stream_dispose_ctx();
        }
        g_stream.pcmf32.clear();
        g_stream.n_transcribed = 0;
        g_stream.n_voiced = 0;
        g_stream.noise_floor = 0.005f;
        g_stream.committed.clear();
        g_stream.last_text.clear();
        // Reset every session: a leftover override from a previous test
        // session must never silently apply to the next one.
        g_stream.test_window_samples = 0;
        g_stream.test_step_samples = 0;
        if (jsonBody.contains("test_window_ms") && jsonBody["test_window_ms"].is_number()) {
            g_stream.test_window_samples =
                (size_t)(jsonBody["test_window_ms"].get<double>() * WHISPER_SAMPLE_RATE / 1000.0);
        }
        if (jsonBody.contains("test_step_ms") && jsonBody["test_step_ms"].is_number()) {
            g_stream.test_step_samples =
                (size_t)(jsonBody["test_step_ms"].get<double>() * WHISPER_SAMPLE_RATE / 1000.0);
        }

        std::string model;
        bool keep_model_loaded = false;
        try {
            g_stream.language  = jsonBody.value("language", "en");
            g_stream.n_threads = jsonBody.value("threads", 4);
            g_stream.translate = jsonBody.value("is_translate", false);
            g_stream.suppress_nst = jsonBody.value("suppress_non_speech_tokens", false);
            g_stream.gate_rms_min   = (float)jsonBody.value("gate_rms_min", 0.0015);
            g_stream.gate_ratio     = (float)jsonBody.value("gate_voice_ratio", 2.5);
            g_stream.gate_floor_cap = (float)jsonBody.value("gate_floor_cap", 0.01);
            g_stream.prompt.clear();
            if (jsonBody.contains("initial_prompt") && jsonBody["initial_prompt"].is_string()) {
                g_stream.prompt = jsonBody["initial_prompt"].get<std::string>();
            }
            keep_model_loaded = jsonBody.value("keep_model_loaded", false);
            model = jsonBody["model"].get<std::string>();
        } catch (const json::exception &e) {
            // A C++ exception escaping extern "C" into FFI would be
            // std::terminate; convert type errors to an error response.
            jsonResult["@type"] = "error";
            jsonResult["message"] =
                std::string("stream_start: bad request: ") + e.what();
            return jsonToChar(jsonResult);
        }

        // Take the parked context when the model path matches (leaving the
        // cache empty while the session uses it), otherwise load from disk.
        // Fresh-session semantics come for free: every stream decode runs
        // with no_context = true, which clears whisper's rolling text
        // history.
        bool borrowed = false;
        {
            std::lock_guard<std::mutex> cache_lock(g_model_cache.mutex);
            if (g_model_cache.ctx != nullptr && g_model_cache.model == model) {
                g_stream.ctx = g_model_cache.ctx;
                g_model_cache.ctx = nullptr;
                g_model_cache.model.clear();
                borrowed = true;
            }
        }
        if (g_stream.ctx == nullptr) {
            whisper_context_params cparams = whisper_context_default_params();
#if TARGET_OS_SIMULATOR
            // See the guard on the same field in transcribe() above.
            cparams.use_gpu = false;
#else
            cparams.use_gpu = true; // falls back to CPU if init fails
#endif
            g_stream.ctx = whisper_init_from_file_with_params(model.c_str(), cparams);
        }
        if (g_stream.ctx == nullptr) {
            jsonResult["@type"] = "error";
            jsonResult["message"] = "stream_start: failed to load model " + model;
            return jsonToChar(jsonResult);
        }
        g_stream.model = model;
        // A borrowed context must go back on stop — the caller that parked
        // it with keep_model_loaded must not silently lose it.
        g_stream.park_on_stop = borrowed || keep_model_loaded;

        jsonResult["@type"] = "streamStarted";
        jsonResult["backend"] = whisper_backend_name(g_stream.ctx);
        return jsonToChar(jsonResult);
    }

    // pcm: 16 kHz mono float32 samples in [-1, 1].
    char *stream_feed(const float *pcm, int32_t n_samples)
    {
        std::lock_guard<std::mutex> lock(g_stream.mutex);
        json jsonResult;

        if (g_stream.ctx == nullptr) {
            jsonResult["@type"] = "error";
            jsonResult["message"] = "stream_feed: stream not started";
            return jsonToChar(jsonResult);
        }
        if (pcm != nullptr && n_samples > 0) {
            double sum2 = 0.0;
            for (int32_t i = 0; i < n_samples; ++i) {
                sum2 += (double)pcm[i] * pcm[i];
            }
            g_stream.pcmf32.insert(g_stream.pcmf32.end(), pcm, pcm + n_samples);

            // Adaptive noise floor: falls quickly, rises slowly, so it
            // tracks room tone without absorbing speech. A chunk is
            // voiced only when clearly above the floor.
            const float rms = (float)std::sqrt(sum2 / n_samples);
            if (rms < g_stream.noise_floor) {
                g_stream.noise_floor += 0.5f * (rms - g_stream.noise_floor);
            } else {
                g_stream.noise_floor += 0.0005f * (rms - g_stream.noise_floor);
            }
            g_stream.noise_floor =
                std::min(g_stream.noise_floor, g_stream.gate_floor_cap);
            const float voice_thold = std::max(
                g_stream.gate_ratio * g_stream.noise_floor,
                g_stream.gate_rms_min);
            if (rms >= voice_thold) {
                g_stream.n_voiced = g_stream.pcmf32.size();
            }
        }

        // Run only when new *voiced* audio arrived — silence alone
        // never triggers a decode.
        if (g_stream.n_voiced > g_stream.n_transcribed &&
            g_stream.pcmf32.size() - g_stream.n_transcribed >= stream_step_samples()) {
            return jsonToChar(stream_run_inference());
        }

        jsonResult["@type"] = "streamPartial";
        jsonResult["text"] = g_stream.committed + g_stream.last_text;
        return jsonToChar(jsonResult);
    }

    char *stream_stop()
    {
        std::lock_guard<std::mutex> lock(g_stream.mutex);
        json jsonResult;

        if (g_stream.ctx == nullptr) {
            jsonResult["@type"] = "error";
            jsonResult["message"] = "stream_stop: stream not started";
            return jsonToChar(jsonResult);
        }

        // Cover voiced audio that arrived after the last run; a silent
        // tail is dropped rather than decoded.
        const size_t n_tail = std::min(g_stream.pcmf32.size(),
                                       g_stream.n_voiced + STREAM_VOICE_PAD);
        if (g_stream.n_voiced > g_stream.n_transcribed &&
            n_tail >= (size_t)WHISPER_SAMPLE_RATE / 2) {
            stream_run_inference();
        }

        jsonResult["@type"] = "streamFinal";
        jsonResult["text"] = g_stream.committed + g_stream.last_text;

        stream_dispose_ctx();
        g_stream.pcmf32.clear();
        g_stream.pcmf32.shrink_to_fit();
        g_stream.n_transcribed = 0;
        g_stream.n_voiced = 0;
        g_stream.committed.clear();
        g_stream.last_text.clear();

        return jsonToChar(jsonResult);
    }
}
