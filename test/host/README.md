# Host-side tests for the live session's decode window

```sh
test/host/run_host_tests.sh                        # ~1 second
WHISPER_HOST_TEST_TRACE=1 test/host/run_host_tests.sh   # ...with every decode shown
```

No model, no device, no simulator. These compile the real
`ios/Classes/whisper_flutter_plus.cpp` and drive the real
`stream_start`/`stream_feed`/`stream_stop`, with only whisper.cpp itself
replaced — `fake_whisper.cpp` implements the dozen whisper API functions the
plugin actually calls.

Faking at that seam is the whole point. A test that copies the boundary
arithmetic into itself can only prove the copy is right, and the window logic
is exactly the kind of code where a reading and the behaviour diverge. Nothing
here re-implements it.

## How a test says what it means

The audio is self-describing: every sample in a 100ms chunk carries the same
float, encoding whether the chunk is speech (and which one), loud non-speech,
or silence. The fake decoder recovers that from the buffer it is handed, so it
knows which audio a decode covered — and the test knows the exact transcript
that audio should produce. Assertions are equality against it, so a single
dropped, duplicated or reordered word fails.

That strictness is deliberate. Both the native `committed` string and the Dart
side's `TranscriptAssembler` are append-only, so a boundary error is permanent
in the user's journal. A wrong transcript is a much worse failure here than a
slow one.

## Writing another one

Tests are named for the *shape of speech* they feed, because the shape decides
which branch of the boundary the session takes:

- phrases shorter than the window commit on timestamp safety alone
- a phrase longer than the window forces the window forward instead
- ...plus a trailing fragment makes that a multi-segment window
- non-speech, or a pause, can leave a window with no speech in it at all

`fake_set_segment_chunks`, `fake_set_trailing_fragment` and
`fake_set_timestamp_slack` choose the shape. Slack matters more than it looks:
whisper never reports a segment's end exactly, and that imprecision is the
entire reason `strip_duplicate_prefix` exists — with exact timestamps the
de-dup is never exercised and a test can pass while the de-dup is broken.

Run with `WHISPER_HOST_TEST_TRACE=1` after writing a test and check it reaches
the branch you wrote it for. A scenario that never gets there proves nothing.
