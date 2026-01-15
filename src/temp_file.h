// temp_file.h - RAII wrapper for temporary files.

#ifndef IREE_ONNX_EP_SRC_TEMP_FILE_H_
#define IREE_ONNX_EP_SRC_TEMP_FILE_H_

#include <string>
#include <string_view>

namespace iree_onnx_ep {

// RAII wrapper for temporary files.
//
// Creates a uniquely-named temporary file with the specified suffix.
// The file is automatically deleted when the TempFile object is destroyed.
// TempFile is move-only.
class TempFile {
 public:
  explicit TempFile(std::string_view suffix);
  ~TempFile();

  // Move-only.
  TempFile(TempFile&& other) noexcept;
  TempFile& operator=(TempFile&& other) noexcept;

  TempFile(const TempFile&) = delete;
  TempFile& operator=(const TempFile&) = delete;

  const std::string& Path() const { return path_; }

  // Marks the file to be kept (not deleted) when TempFile is destroyed.
  void Keep() { keep_ = true; }

 private:
  std::string path_;
  bool keep_ = false;
};

}  // namespace iree_onnx_ep

#endif  // IREE_ONNX_EP_SRC_TEMP_FILE_H_
