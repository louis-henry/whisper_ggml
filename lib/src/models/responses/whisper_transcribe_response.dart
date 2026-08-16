// ignore_for_file: invalid_annotation_target

import 'package:freezed_annotation/freezed_annotation.dart';
import 'package:whisper_ggml/src/models/responses/whisper_transcribe_segment.dart';

part 'whisper_transcribe_response.freezed.dart';
part 'whisper_transcribe_response.g.dart';

/// Response model of whisper getVersion
@freezed
abstract class WhisperTranscribeResponse with _$WhisperTranscribeResponse {
  ///
  const factory WhisperTranscribeResponse({
    @JsonKey(name: '@type') required String type,
    required String text,
    @JsonKey(name: 'segments')
    required List<WhisperTranscribeSegment>? segments,
    // The compute backend this transcription actually resolved to (e.g.
    // "Metal", "CPU") — the backend ggml's device scheduler initialized,
    // not merely what the build compiled in. Null on platforms/builds that
    // don't report it (e.g. Android, or a fork commit before this field
    // existed).
    String? backend,
  }) = _WhisperTranscribeResponse;

  const WhisperTranscribeResponse._();

  /// Parse [json] to WhisperTranscribeResponse
  factory WhisperTranscribeResponse.fromJson(Map<String, dynamic> json) =>
      _$WhisperTranscribeResponseFromJson(json);
}
