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
`fake_set_timestamp_slack` choose the shape.

### Why the timestamps are deliberately imprecise

`fake_set_timestamp_slack` exists because a fake that is kinder than reality
will happily agree with broken code. Whisper never reports a segment's end
exactly, and that imprecision is the *entire* reason `strip_duplicate_prefix`
exists: an end reported slightly early makes the boundary retain audio for
words already committed, so the next decode emits them a second time.

The first draft of these tests reported exact segment ends — and passed, while
the de-dup was broken, because with exact timestamps a duplicate never arises
and the de-dup is never called on to do anything. Modelling the imprecision is
what made the bug visible. Any fake here should be checked the same way: ask
what it makes easier than the real thing, because that is where it will agree
with a defect.

Slack also has to stay small in the force-commit tests. Above ~100ms a segment
reports as ending before the keep margin, gets committed on timestamp safety
like any other, and the force-commit branch is never reached — so the test
passes without testing anything.

### Check the test still reaches its branch

Run with `WHISPER_HOST_TEST_TRACE=1` and read the decodes: each line gives the
decode length, where every segment ended, and where the keep margin fell, which
is exactly what decides the branch.

Do this after writing a test, and again whenever one turns green. **A test can
go green by ceasing to reach its branch, and that is indistinguishable from a
fix in the pass/fail output.** The trace is what makes the difference visible,
which is the only reason either of the cautions above is checkable rather than
folklore.
