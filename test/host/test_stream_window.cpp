// Host-side tests for the live session's sliding decode window.
//
// These drive the real stream_start/stream_feed/stream_stop in
// whisper_flutter_plus.cpp, with whisper.cpp itself replaced by the scripted
// decoder in fake_whisper.cpp. Nothing here re-implements the window logic, so
// a change to that logic is seen by these tests rather than hidden from them.
//
// The audio is self-describing (see fake_whisper.h): the fake turns every
// speech chunk into a numbered word, so the expected transcript is known
// exactly and the assertion can be equality against it. A single dropped,
// duplicated or reordered word fails. That is the failure mode that matters
// here — both the native `committed` string and the Dart-side
// TranscriptAssembler are append-only, so a boundary error is permanent in the
// user's journal, unlike a slow one.
//
// Each test names the *shape of speech* it feeds, because the shape is what
// decides which branch of the boundary the session takes. Run with
// WHISPER_HOST_TEST_TRACE=1 to see the decodes and confirm a test still
// reaches the branch it was written for.
//
// Build and run: test/host/run_host_tests.sh

#include "fake_whisper.h"

#include "json/json.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::json;

extern "C" {
char *stream_start(char *body);
char *stream_feed(const float *pcm, int32_t n_samples);
char *stream_stop();
}

namespace {

constexpr int kChunkSamples = 1600;   // 100ms at 16kHz

// Whisper never reports a segment's end exactly. Ordinary speech gets enough
// slack to make a boundary retain audio for words already committed; the
// force-commit shapes get less, because a segment reported as ending before
// the keep margin would be committed normally and never force anything.
constexpr int kOrdinarySlackCs = 30;
constexpr int kForceCommitSlackCs = 5;

int g_failures = 0;

// With WHISPER_HOST_TEST_TRACE set, the fake decoder prints the shape of every
// decode; this keeps those lines under the test that provoked them.
void begin(const std::string &name)
{
    if (getenv("WHISPER_HOST_TEST_TRACE") != nullptr) printf("  -- %s\n", name.c_str());
}

void check(const std::string &name, bool ok, const std::string &detail = "")
{
    printf("%s %s\n", ok ? "  ok  " : "  FAIL", name.c_str());
    if (!ok) {
        g_failures++;
        if (!detail.empty()) printf("        %s\n", detail.c_str());
    }
}

std::string take(char *owned)
{
    std::string s = owned == nullptr ? "" : owned;
    free(owned);
    return s;
}

std::string text_of(const std::string &response)
{
    const json parsed = json::parse(response, nullptr, false);
    if (parsed.is_discarded() || !parsed.contains("text")) return "";
    return parsed["text"].get<std::string>();
}

std::vector<std::string> words_of(const std::string &text)
{
    std::vector<std::string> words;
    std::istringstream in(text);
    std::string word;
    while (in >> word) words.push_back(word);
    return words;
}

// Feeds a script of 100ms chunks through a real session and returns the final
// transcript. Each entry is the sample value for that chunk: a speech sample,
// kFakeNoise or kFakeSilence.
std::string run_session(const std::vector<float> &chunks)
{
    std::string body = R"({"model":"fake.bin","language":"en"})";
    take(stream_start(&body[0]));

    std::vector<float> pcm(kChunkSamples);
    for (const float value : chunks) {
        for (float &sample : pcm) sample = value;
        take(stream_feed(pcm.data(), kChunkSamples));
    }
    return text_of(take(stream_stop()));
}

// Every speech chunk in the script, in order, exactly once.
std::vector<std::string> expected_words(const std::vector<float> &chunks)
{
    std::vector<std::string> words;
    for (size_t i = 0; i < chunks.size(); ++i) {
        if (chunks[i] != kFakeNoise && chunks[i] != kFakeSilence) {
            words.push_back("w" + std::to_string((int)i));
        }
    }
    return words;
}

std::vector<float> speech(int count, int first_chunk = 0)
{
    std::vector<float> chunks;
    for (int i = 0; i < count; ++i) chunks.push_back(fake_speech_sample(first_chunk + i));
    return chunks;
}

void append(std::vector<float> &into, const std::vector<float> &more)
{
    into.insert(into.end(), more.begin(), more.end());
}

std::string sample_of(const std::vector<std::string> &words)
{
    std::string out;
    for (const std::string &word : words) {
        out += word + " ";
        if (out.size() > 110) { out += "..."; break; }
    }
    return out;
}

// Reports every way the transcript differs from what was said, so one kind of
// damage never hides another: a duplicated word and a dropped word have
// different causes and both need to be visible in a single run.
std::string differences(const std::vector<std::string> &expected,
                        const std::vector<std::string> &actual)
{
    std::vector<std::string> notes;

    std::set<std::string> seen, duplicated;
    for (const std::string &word : actual) {
        if (!seen.insert(word).second) duplicated.insert(word);
    }
    if (!duplicated.empty()) {
        notes.push_back("committed twice (" + std::to_string(duplicated.size()) +
                        "): " + sample_of({duplicated.begin(), duplicated.end()}));
    }

    std::vector<std::string> missing;
    for (const std::string &word : expected) {
        if (seen.find(word) == seen.end()) missing.push_back(word);
    }
    if (!missing.empty()) {
        notes.push_back("never transcribed (" + std::to_string(missing.size()) +
                        "): " + sample_of(missing));
    }

    if (notes.empty() && expected != actual) {
        for (size_t i = 0; i < expected.size() && i < actual.size(); ++i) {
            if (expected[i] != actual[i]) {
                notes.push_back("out of order at word " + std::to_string(i) +
                                ": expected " + expected[i] + ", got " + actual[i]);
                break;
            }
        }
    }

    std::string out;
    for (size_t i = 0; i < notes.size(); ++i) out += (i ? "; " : "") + notes[i];
    return out;
}

void expect_transcript(const std::string &name, const std::vector<float> &chunks,
                       const std::string &transcript)
{
    const std::vector<std::string> actual = words_of(transcript);
    const std::vector<std::string> expected = expected_words(chunks);
    check(name, actual == expected, differences(expected, actual));
}

// --- the tests ------------------------------------------------------------

// Phrases shorter than the window: segments close well inside it, so every
// boundary commits on timestamp safety alone.
void ordinary_phrases_transcribe_exactly()
{
    const std::string name = "ordinary phrases transcribe exactly";
    begin(name);
    fake_reset();
    fake_set_timestamp_slack(kOrdinarySlackCs);
    const std::vector<float> chunks = speech(300);
    expect_transcript(name, chunks, run_session(chunks));
}

// Talking without pausing: one phrase longer than the whole window, so no
// segment ever clears the keep margin and the window has to be forced forward.
void speech_with_no_pause_in_it_transcribes_exactly()
{
    const std::string name = "speech with no pause in it transcribes exactly";
    begin(name);
    fake_reset();
    fake_set_segment_chunks(200);
    fake_set_timestamp_slack(kForceCommitSlackCs);
    const std::vector<float> chunks = speech(300);
    expect_transcript(name, chunks, run_session(chunks));
}

// The same, plus the short trailing fragment whisper routinely emits at a
// window edge, which makes it a multi-segment window where still nothing
// clears the margin.
void unpaused_speech_with_a_trailing_fragment_transcribes_exactly()
{
    const std::string name =
        "unpaused speech with a trailing fragment transcribes exactly";
    begin(name);
    fake_reset();
    fake_set_segment_chunks(200);
    fake_set_trailing_fragment(true);
    fake_set_timestamp_slack(kForceCommitSlackCs);
    const std::vector<float> chunks = speech(300);
    expect_transcript(name, chunks, run_session(chunks));
}

// Stopping is not a quiet moment: the buffer holds more than one window's
// worth for part of every cycle, and where the user's finger lands in that
// cycle is arbitrary.
void stopping_at_any_point_in_the_cycle_transcribes_exactly()
{
    const std::string name = "stopping at any point in the cycle transcribes exactly";
    begin(name);
    for (int extra = 0; extra <= 40; ++extra) {
        fake_reset();
        fake_set_segment_chunks(200);
        fake_set_timestamp_slack(kForceCommitSlackCs);
        const std::vector<float> chunks = speech(200 + extra);
        const std::vector<std::string> actual = words_of(run_session(chunks));
        const std::vector<std::string> expected = expected_words(chunks);
        if (actual != expected) {
            check(name, false,
                  "stopped after " + std::to_string(chunks.size() * 100) + "ms: " +
                      differences(expected, actual));
            return;
        }
    }
    check(name, true);
}

// Loud non-speech — music, a fan, tapping — opens the energy gate but decodes
// to nothing. A window of it must still advance, or the next call re-decodes
// the identical leading slice of the buffer, gets the identical result, and
// the session is frozen for good while still accepting audio.
void non_speech_in_the_middle_of_a_session_does_not_freeze_it()
{
    const std::string name = "non-speech in the middle of a session does not freeze it";
    begin(name);
    fake_reset();
    fake_set_timestamp_slack(kOrdinarySlackCs);
    std::vector<float> chunks = speech(100);
    append(chunks, std::vector<float>(150, kFakeNoise));    // 15s of it
    append(chunks, speech(100, 250));
    expect_transcript(name, chunks, run_session(chunks));
}

// A long pause has the same shape: once a force-commit has erased its window,
// the front of the buffer is exactly where the speaker stopped, so thinking
// for a while leaves a window with no speech in it at all.
void a_long_pause_does_not_freeze_the_session()
{
    const std::string name = "a long pause does not freeze the session";
    begin(name);
    fake_reset();
    fake_set_segment_chunks(200);
    fake_set_timestamp_slack(kForceCommitSlackCs);
    std::vector<float> chunks = speech(150);
    append(chunks, std::vector<float>(150, kFakeSilence));   // 15s of thinking
    append(chunks, speech(150, 300));
    expect_transcript(name, chunks, run_session(chunks));
}

// STREAM_KEEP_SAMPLES exists so a word at a boundary is never decoded with
// silence butted against it. A boundary that erases its whole window throws
// that away in the one case where the cut is most likely to land mid-word.
void every_boundary_keeps_audio_for_the_next_decode()
{
    const std::string name = "every boundary keeps audio for the next decode";
    begin(name);
    fake_reset();
    fake_set_segment_chunks(200);
    fake_set_timestamp_slack(kForceCommitSlackCs);
    run_session(speech(300));

    const std::vector<FakeDecode> &decodes = fake_decodes();
    int without_overlap = 0;
    for (size_t i = 1; i < decodes.size(); ++i) {
        if (decodes[i].first_chunk < 0 || decodes[i - 1].last_chunk < 0) continue;
        if (decodes[i].first_chunk > decodes[i - 1].last_chunk) without_overlap++;
    }
    check(name, without_overlap == 0,
          std::to_string(without_overlap) + " of " + std::to_string(decodes.size()) +
              " decodes began after the previous one ended, so a word spanning the "
              "cut was decoded as two halves with silence against each");
}

}  // namespace

int main()
{
    printf("live session decode window\n");
    ordinary_phrases_transcribe_exactly();
    speech_with_no_pause_in_it_transcribes_exactly();
    unpaused_speech_with_a_trailing_fragment_transcribes_exactly();
    stopping_at_any_point_in_the_cycle_transcribes_exactly();
    non_speech_in_the_middle_of_a_session_does_not_freeze_it();
    a_long_pause_does_not_freeze_the_session();
    every_boundary_keeps_audio_for_the_next_decode();

    printf("%s\n", g_failures == 0
                       ? "all passed"
                       : (std::to_string(g_failures) + " failed").c_str());
    return g_failures == 0 ? 0 : 1;
}
