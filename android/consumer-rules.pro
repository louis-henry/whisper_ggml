# whisper_ggml consumer rules — merged into the R8 configuration of every
# app that depends on this plugin.
#
# Fork: the ffmpeg-kit keep rules are gone with the dependency. The generic
# JNI/attribute rules below are unrelated to FFmpeg and are kept.

# Keep native method names for JNI registration.
-keepclasseswithmembernames class * {
    native <methods>;
}

-keepattributes *Annotation*
-keepattributes Signature
-keepattributes InnerClasses
