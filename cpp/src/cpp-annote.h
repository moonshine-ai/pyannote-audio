// SPDX-License-Identifier: MIT
// Public diarization API using the pointer-to-implementation (pimpl) pattern.
// All internal state (ONNX Runtime sessions, clustering parameters, etc.) is
// hidden behind CppAnnote::Impl, defined in the .cpp translation unit.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cppannote {

struct DiarizationTurn {
  double start = 0.;
  double end = 0.;
  std::string speaker;
  bool operator<(const DiarizationTurn& o) const {
    if (start != o.start) {
      return start < o.start;
    }
    if (end != o.end) {
      return end < o.end;
    }
    return speaker < o.speaker;
  }
};

struct DiarizationResults {
  std::vector<DiarizationTurn> turns;

  void write_json(const std::string& path) const;
  void write_json(std::ostream& os) const;
};

/// Loads segmentation and embedding ONNX models once and manages streaming
/// diarization sessions.  All heavy implementation details (ORT sessions,
/// PLDA model, VBx clustering) are hidden behind the pimpl firewall.
class CppAnnote {
 public:
  /// Construct the diarization engine.  ``embedding_onnx_path`` is required;
  /// leave other paths empty to use compiled-in community-1 defaults.
  explicit CppAnnote(
      std::string segmentation_onnx_path,
      std::string receptive_field_json_path,
      std::string golden_speaker_bounds_json_path,
      std::string pipeline_snapshot_json_path,
      std::string embedding_onnx_path,
      std::string xvec_transform_npz_path,
      std::string plda_npz_path);

  ~CppAnnote();

  CppAnnote(const CppAnnote&) = delete;
  CppAnnote& operator=(const CppAnnote&) = delete;
  CppAnnote(CppAnnote&&) noexcept;
  CppAnnote& operator=(CppAnnote&&) noexcept;

  /// Allocate a new streaming diarization session and return its handle.
  /// ``refresh_every_sec`` controls how often VBx re-clustering runs.
  int32_t create_stream(double refresh_every_sec = 2.0);

  /// Release a stream and all associated resources.
  void free_stream(int32_t stream_id);

  /// (Re-)initialize a stream, clearing any buffered audio and cached results.
  void start_stream(int32_t stream_id);

  /// Finalize the stream (forces a last VBx pass) and return diarization.
  DiarizationResults stop_stream(int32_t stream_id);

  /// Append PCM audio to a stream.  Resampling to the model rate is handled
  /// internally; ``sample_rate`` is the rate of the supplied buffer.
  void add_audio_to_stream(int32_t stream_id, const float* audio_data,
                           uint64_t audio_length, int32_t sample_rate);

  /// Force a VBx refresh and return the current diarization snapshot without
  /// stopping the stream.
  DiarizationResults diarize_stream(int32_t stream_id);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace cppannote
