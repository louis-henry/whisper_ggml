#include "fake_whisper.h"

#include "whisper.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

struct Run {          // a maximal stretch of identical samples
    float value;
    int offset;       // sample offset within the decode window
    int length;
};

struct Segment {
    std::string text;
    int64_t t0_cs;
    int64_t t1_cs;
};

int g_segment_chunks = 40;              // 4s at 100ms per chunk
int g_timestamp_slack_cs = 0;
bool g_trailing_fragment = false;
std::vector<Segment> g_segments;
std::vector<FakeDecode> g_decodes;
char g_ctx_marker = 0;

constexpr int kChunkSamples = 1600;     // 100ms at 16kHz
constexpr int kSampleRate = 16000;

void trace_decode(int n_samples);

int64_t cs_at(int sample_offset) { return (int64_t)sample_offset * 100 / kSampleRate; }

// A segment end as whisper would report it: near the true end, not exactly on
// it. Always strictly AFTER the segment's own start — whisper does not emit
// zero-length segments, and a fake that does feeds the code under test an
// input reality never produces.
//
// This clamp returned start_cs once. A zero-length segment makes the boundary
// commit text while erasing nothing, so the window cannot advance: it looked
// exactly like the freeze defect, against code where that was already fixed.
// Worse, once the code grew a guard for zero progress, the degenerate segment
// stopped failing and started passing THROUGH that guard — so the two tests
// named for the no-speech-window case silently stopped reaching it, and kept
// reporting green. A fake misleads by being harsher than reality as well as
// kinder.
int64_t reported_end(int64_t true_end_cs, int64_t start_cs)
{
    const int64_t reported = true_end_cs - g_timestamp_slack_cs;
    return reported > start_cs ? reported : start_cs + 1;
}

std::vector<Run> split_runs(const float *pcm, int n)
{
    std::vector<Run> runs;
    for (int i = 0; i < n;) {
        int j = i;
        while (j < n && pcm[j] == pcm[i]) j++;
        runs.push_back({pcm[i], i, j - i});
        i = j;
    }
    return runs;
}

// Builds the segments whisper would plausibly return for this window: every
// speech chunk it can see becomes a word, phrase-length runs of them become
// segments, and stretches with no speech in them produce no segment at all.
//
// A chunk the window only partly covers still yields its word. That models the
// one thing the de-dup exists for: a word straddling a window boundary gets
// emitted by the decode on both sides of it.
void build_segments(const float *pcm, int n)
{
    g_segments.clear();

    // whisper.cpp declines anything under 100ms outright, returning success
    // with no segments at all ("input is too short - %d ms < 100 ms", and it
    // is a warning, not an error). A fake that cheerfully decodes 20ms would
    // let a caller believe a fragment that short is safely transcribed.
    if (n < WHISPER_SAMPLE_RATE / 10) return;

    const std::vector<Run> runs = split_runs(pcm, n);

    FakeDecode record{-1, -1, false, n};
    std::vector<std::pair<int, Run>> speech;   // chunk id -> run
    for (const Run &run : runs) {
        if (run.value == kFakeSilence || run.value == kFakeNoise) continue;
        const int chunk_id = (int)run.value - 1;
        if (record.first_chunk < 0) {
            record.first_chunk = chunk_id;
            record.starts_mid_chunk = run.length < kChunkSamples && run.offset == 0;
        }
        record.last_chunk = chunk_id;
        speech.push_back({chunk_id, run});
    }
    g_decodes.push_back(record);
    if (speech.empty()) return;         // a window of pure silence or noise

    const size_t last = speech.size() - 1;
    const size_t fragment_at =
        (g_trailing_fragment && speech.size() > 1) ? last : speech.size();

    std::string text;
    int64_t seg_start_cs = cs_at(speech.front().second.offset);
    int in_segment = 0;
    for (size_t i = 0; i < speech.size(); ++i) {
        const Run &run = speech[i].second;
        const int64_t end_cs = cs_at(run.offset + run.length);
        if (i == fragment_at) {
            if (!text.empty()) {
                g_segments.push_back({text, seg_start_cs,
                                      reported_end(cs_at(run.offset), seg_start_cs)});
                text.clear();
            }
            g_segments.push_back({" w" + std::to_string(speech[i].first),
                                  cs_at(run.offset),
                                  reported_end(end_cs, cs_at(run.offset))});
            in_segment = 0;
            continue;
        }
        if (text.empty()) seg_start_cs = cs_at(run.offset);
        text += " w" + std::to_string(speech[i].first);
        if (++in_segment == g_segment_chunks || i == last) {
            g_segments.push_back({text, seg_start_cs,
                                  reported_end(end_cs, seg_start_cs)});
            text.clear();
            in_segment = 0;
        }
    }
    if (!text.empty()) {
        const int64_t true_end = cs_at(speech.back().second.offset +
                                       speech.back().second.length);
        g_segments.push_back({text, seg_start_cs, reported_end(true_end, seg_start_cs)});
    }

    // Trailing silence past the last voiced sample: whisper fills it with text
    // nobody said.
    const int last_voiced_end =
        speech.back().second.offset + speech.back().second.length;
    if (n - last_voiced_end > kHallucinationSilenceSamples) {
        g_segments.push_back({std::string(" ") + kHallucinatedWord,
                              cs_at(last_voiced_end), cs_at(n)});
    }
}

// Set WHISPER_HOST_TEST_TRACE=1 to see the shape of every decode. Worth doing
// when adding a test: a scenario that never reaches the branch it names proves
// nothing, and the segment ends printed here are what decide which branch the
// boundary takes.
void trace_decode(int n_samples)
{
    static const bool on = getenv("WHISPER_HOST_TEST_TRACE") != nullptr;
    if (!on) return;
    printf("    decode %5.2fs  %d segment(s) ending at:", (double)n_samples / kSampleRate,
           (int)g_segments.size());
    for (const Segment &segment : g_segments) printf(" %.2fs", segment.t1_cs / 100.0);
    printf("   (keep margin at %.2fs)\n", (double)n_samples / kSampleRate - 0.2);
}

}  // namespace

const char *const kHallucinatedWord = "wGHOST";

void fake_set_segment_chunks(int chunks) { g_segment_chunks = chunks; }
void fake_set_timestamp_slack(int cs) { g_timestamp_slack_cs = cs; }
void fake_set_trailing_fragment(bool on) { g_trailing_fragment = on; }
const std::vector<FakeDecode> &fake_decodes() { return g_decodes; }

void fake_reset()
{
    g_segment_chunks = 40;
    g_timestamp_slack_cs = 0;
    g_trailing_fragment = false;
    g_segments.clear();
    g_decodes.clear();
}

// ---- the whisper.cpp API surface whisper_flutter_plus.cpp actually uses ----

struct whisper_context_params whisper_context_default_params(void)
{
    struct whisper_context_params p;
    memset(&p, 0, sizeof(p));
    return p;
}

struct whisper_context *whisper_init_from_file_with_params(
    const char *, struct whisper_context_params)
{
    return (struct whisper_context *)&g_ctx_marker;
}

void whisper_free(struct whisper_context *) {}

const char *whisper_backend_name(struct whisper_context *ctx)
{
    return ctx == nullptr ? "none" : "FAKE";
}

struct whisper_full_params whisper_full_default_params(enum whisper_sampling_strategy)
{
    struct whisper_full_params p;
    memset(&p, 0, sizeof(p));
    return p;
}

int whisper_full(struct whisper_context *, struct whisper_full_params,
                 const float *samples, int n_samples)
{
    build_segments(samples, n_samples);
    trace_decode(n_samples);
    return 0;
}

int whisper_full_n_segments(struct whisper_context *) { return (int)g_segments.size(); }

const char *whisper_full_get_segment_text(struct whisper_context *, int i)
{
    return g_segments[i].text.c_str();
}

int64_t whisper_full_get_segment_t0(struct whisper_context *, int i) { return g_segments[i].t0_cs; }
int64_t whisper_full_get_segment_t1(struct whisper_context *, int i) { return g_segments[i].t1_cs; }
bool whisper_full_get_segment_speaker_turn_next(struct whisper_context *, int) { return false; }
int whisper_is_multilingual(struct whisper_context *) { return 0; }
int whisper_lang_id(const char *) { return 0; }
