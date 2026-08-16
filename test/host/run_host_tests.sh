#!/usr/bin/env bash
# Builds and runs the host-side tests for the live session's decode window.
# No model, no device, no simulator — about a second, on any development
# machine, against the real whisper_flutter_plus.cpp.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
out="${TMPDIR:-/tmp}/whisper_host_tests"

# whisper_flutter_plus.cpp carries a vestigial main() of its own (it predates
# this fork), so rename it out of the way for this build rather than touch the
# file under test.
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

flags=(-std=c++17 -g -fexceptions -Wall -Wno-deprecated-literal-operator
       -I "$root/ios/Classes"
       -I "$root/ios/Classes/whisper/include"
       -I "$root/ios/Classes/whisper/ggml/include")

clang++ "${flags[@]}" -Dmain=whisper_flutter_plus_unused_main \
  -c "$root/ios/Classes/whisper_flutter_plus.cpp" -o "$work/subject.o"
clang++ "${flags[@]}" -c "$here/fake_whisper.cpp" -o "$work/fake.o"
clang++ "${flags[@]}" -c "$here/test_stream_window.cpp" -o "$work/test.o"
clang++ -o "$out" "$work/subject.o" "$work/fake.o" "$work/test.o"

"$out"
