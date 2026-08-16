Pod::Spec.new do |s|
  s.name             = 'whisper_ggml'
  s.version          = '1.0.1'
  s.summary          = 'A new Flutter FFI plugin project.'
  s.description      = <<-DESC
A new Flutter FFI plugin project.
                       DESC
  s.homepage         = 'https://github.com/sk3llo/whisper_ggml'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Your Company' => 'kapraton@gmail.com' }

  # This will ensure the source files in Classes/ are included in the native
  # builds of apps using this FFI plugin. Podspec does not support relative
  # paths, so Classes contains a forwarder C file that relatively imports
  # `../src/*` so that the C sources can be shared among all target platforms.
  s.dependency 'Flutter'
  s.source           = {
    :git => 'https://github.com/sk3llo/whisper_ggml'
  }

  # ggml's Metal backend (Classes/whisper/ggml/src/ggml-metal/, vendored from
  # whisper.cpp v1.9.1 — matching GGML_VERSION below) uses
  # GGML_METAL_EMBED_LIBRARY: the .metal shader source is merged with two
  # headers and embedded as raw bytes in a hand-written assembly (.s) file's
  # __DATA,__ggml_metallib section, via .incbin — Metal compiles it from
  # source at runtime on first use, so no .metallib resource bundle is
  # needed. Upstream generates the merge and the .s file with CMake
  # add_custom_command; CocoaPods has no equivalent, so prepare_command
  # reproduces the exact same three-step sed/echo sequence CMake's
  # ggml/src/ggml-metal/CMakeLists.txt uses, once, at `pod install` time
  # — before source_files below is evaluated, which is why the generated
  # ggml-metal-embed.s and ggml-metal-embed.metal can already be listed
  # in that glob.
  s.prepare_command = <<-CMD
    set -e
    METAL_DIR="Classes/whisper/ggml/src/ggml-metal"
    COMMON_H="Classes/whisper/ggml/src/ggml-common.h"
    sed -e "/__embed_ggml-common.h__/r ${COMMON_H}" \
        -e "/__embed_ggml-common.h__/d" \
        < "${METAL_DIR}/ggml-metal.metal" \
        > "${METAL_DIR}/ggml-metal-embed.metal.tmp"
    sed -e "/#include \\"ggml-metal-impl.h\\"/r ${METAL_DIR}/ggml-metal-impl.h" \
        -e "/#include \\"ggml-metal-impl.h\\"/d" \
        < "${METAL_DIR}/ggml-metal-embed.metal.tmp" \
        > "${METAL_DIR}/ggml-metal-embed.metal"
    rm -f "${METAL_DIR}/ggml-metal-embed.metal.tmp"
    {
      echo '.section __DATA,__ggml_metallib'
      echo '.globl _ggml_metallib_start'
      echo '_ggml_metallib_start:'
      echo '.incbin "ggml-metal-embed.metal"'
      echo '.globl _ggml_metallib_end'
      echo '_ggml_metallib_end:'
    } > "${METAL_DIR}/ggml-metal-embed.s"
  CMD

  s.source_files = 'Classes/**/*.{cpp,c,h,hpp,m,s}'
  # Only whisper.h is public; the ggml tree has duplicate header basenames
  # (common.h, quants.h) that collide when flattened into the framework.
  s.public_header_files = 'Classes/whisper/include/whisper.h'
  s.platform = :ios, '15.6'
  s.ios.deployment_target  = '15.6'

  # Flutter.framework does not contain a i386 slice.
  s.xcconfig = {
    'IPHONEOS_DEPLOYMENT_TARGET' => '15.6',
    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++20',
  }
  s.library = 'c++'
  s.frameworks = 'Accelerate', 'Foundation', 'Metal', 'MetalKit'
  # ggml's Metal .m files (ggml-metal-device.m, ggml-metal-context.m) are
  # written against manual retain/release — explicit `[obj release]` calls
  # and unbridged void*<->id casts for its C-interop pattern, matching how
  # upstream's own build compiles them. This pod had no Objective-C source
  # before Metal, so nothing here previously relied on ARC; every other
  # vendored file is .cpp/.c, where this flag is a no-op.
  s.compiler_flags = '-fno-objc-arc'
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386',
    # whisper.cpp v1.9.1 include roots. CPU backend stays compiled in as
    # ggml's own fallback path (registered after Metal in
    # ggml-backend-reg.cpp); Metal is additive, not a replacement.
    'HEADER_SEARCH_PATHS' => [
      '"$(PODS_TARGET_SRCROOT)/Classes/whisper/include"',
      '"$(PODS_TARGET_SRCROOT)/Classes/whisper/ggml/include"',
      '"$(PODS_TARGET_SRCROOT)/Classes/whisper/ggml/src"',
      '"$(PODS_TARGET_SRCROOT)/Classes/whisper/ggml/src/ggml-cpu"',
      '"$(PODS_TARGET_SRCROOT)/Classes/whisper/ggml/src/ggml-metal"',
      '"$(PODS_TARGET_SRCROOT)/Classes/whisper/src"',
    ].join(' '),
    'GCC_PREPROCESSOR_DEFINITIONS' => '$(inherited) GGML_USE_CPU=1 GGML_USE_ACCELERATE=1 ACCELERATE_NEW_LAPACK=1 ACCELERATE_LAPACK_ILP64=1 GGML_USE_METAL=1 GGML_METAL_EMBED_LIBRARY=1 GGML_VERSION=\"1.9.1\" GGML_COMMIT=\"whisper.cpp-v1.9.1\" WHISPER_VERSION=\"1.9.1\"',
    # keep inference usable in debug builds
    'GCC_OPTIMIZATION_LEVEL' => '3',
  }
  s.swift_version = '5.0'
end
