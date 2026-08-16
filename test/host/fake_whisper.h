// A scripted stand-in for whisper.cpp, so the live-session window logic in
// whisper_flutter_plus.cpp can be tested on a development machine in about a
// second, with no model, no device and no simulator.
//
// The point of faking at *this* seam rather than copying the window logic into
// a test is that the test then drives the real stream_start/stream_feed/
// stream_stop and the real stream_run_inference. A test that re-implements the
// boundary arithmetic can only ever prove the copy is correct.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Audio the test feeds is self-describing: every sample in a 100ms chunk
// carries the same float, and the fake decoder recovers from that float what
// the chunk was, including which chunks a decode window covers and whether it
// was handed a whole chunk or a fragment of one.
//
//   speech  -> (float)(chunk_id + 1), decoded as the word "w<chunk_id>"
//   noise   -> kFakeNoise, loud enough to open the RMS gate, decodes to no
//              words at all (music, a fan, tapping, a slammed door)
//   silence -> 0.0f, never opens the gate
constexpr float kFakeNoise = -1.0f;
constexpr float kFakeSilence = 0.0f;
inline float fake_speech_sample(int chunk_id) { return (float)(chunk_id + 1); }

// Words per segment. Whisper segments are phrase-length, not word-length, and
// a segment longer than the decode window is what drives the force-commit
// fallback, so tests set this to choose which path they exercise.
void fake_set_segment_chunks(int chunks);

// When set, the last speech chunk of a window becomes its own short segment.
// Whisper does this routinely at a window edge, and it is what produces a
// multi-segment window in which no segment clears the keep margin.
void fake_set_trailing_fragment(bool on);

// Whisper's segment timestamps are approximate. Reporting a segment as ending
// slightly before its text really ends is the ordinary case that makes a
// boundary retain audio for words already committed, so the next decode emits
// them a second time — which is the entire reason strip_duplicate_prefix
// exists. Slack is in centiseconds.
void fake_set_timestamp_slack(int cs);

// Every decode the code under test asked for, in order. Offsets are in
// samples, absolute across the session, recovered from the audio itself, so
// this is a true record of which audio was decoded and which never was.
struct FakeDecode {
    int64_t first_chunk;      // first chunk id the window touched, -1 if none
    int64_t last_chunk;       // last chunk id the window touched, -1 if none
    bool starts_mid_chunk;    // window opened part-way through first_chunk
    int n_samples;
};
const std::vector<FakeDecode> &fake_decodes();

void fake_reset();
