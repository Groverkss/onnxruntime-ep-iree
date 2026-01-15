// mlir_gen.cc - MLIR text generation from OrtGraph.

#include "mlir_gen.h"

#include <cassert>
#include <cstdint>
#include <format>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace iree_onnx_ep {
namespace {

// Sanitizes an ONNX name to be a valid MLIR SSA identifier.
// MLIR identifiers must match [a-zA-Z_][a-zA-Z0-9_$]*.
std::string SanitizeName(const std::string& name) {
  assert(!name.empty() && "Unexpected empty name");
  std::string result;
  result.reserve(name.size());
  for (char c : name) {
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '$') {
      result += c;
    } else {
      result += '_';
    }
  }
  // Ensure starts with letter or underscore.
  if (!result.empty() && std::isdigit(static_cast<unsigned char>(result[0]))) {
    result = "_" + result;
  }
  return result.empty() ? "_unnamed" : result;
}

// Joins a vector of strings with a separator.
std::string Join(const std::vector<std::string>& parts, const std::string& sep) {
  std::ostringstream ss;
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i > 0) {
      ss << sep;
    }
    ss << parts[i];
  }
  return ss.str();
}

// Returns the byte size for an ONNX tensor element type.
size_t GetElementByteSize(ONNXTensorElementDataType dtype) {
  switch (dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return 8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return 1;
    default:
      return 1;
  }
}

// Returns MLIR element type string for an ONNX tensor element type.
// If signless is true, returns signless types (i64) for all integers.
// Otherwise returns signed (si64) or unsigned (ui64) types for torch dialect.
std::string GetElementType(ONNXTensorElementDataType dtype,
                           bool signless = false) {
  switch (dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return "f32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return "f64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return "f16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return "bf16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return signless ? "i8" : "si8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return signless ? "i16" : "si16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return signless ? "i32" : "si32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return signless ? "i64" : "si64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return signless ? "i8" : "ui8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return signless ? "i16" : "ui16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return signless ? "i32" : "ui32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return signless ? "i64" : "ui64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return "i1";
    default:
      return "NYI";
  }
}

// Formats a tensor type as !torch.vtensor<[dims],dtype>.
std::string FormatTensorType(const Ort::ConstTypeInfo& type_info) {
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    return "NYI";  // NYI: non-tensor types.
  }

  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  auto shape = tensor_info.GetShape();
  auto dtype = tensor_info.GetElementType();

  std::ostringstream ss;
  ss << "!torch.vtensor<[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      ss << ",";
    }
    if (shape[i] < 0) {
      // TODO: Ensure that dynamic dimensions are actually represented as -1. I
      // checked and they seem to but an example to check would be good.
      ss << "?";
    } else {
      ss << shape[i];
    }
  }
  ss << "]," << GetElementType(dtype) << ">";
  return ss.str();
}

// Formats a tensor type as tensor<dimsxdtype> (standard MLIR format).
// Uses signless integer types as required by MLIR tensor dialect.
std::string FormatMlirTensorType(const Ort::ConstTypeInfo& type_info) {
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    return "NYI";
  }

  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  auto shape = tensor_info.GetShape();
  auto dtype = tensor_info.GetElementType();

  std::ostringstream ss;
  ss << "tensor<";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] < 0) {
      ss << "?";
    } else {
      ss << shape[i];
    }
    ss << "x";
  }
  ss << GetElementType(dtype, /*signless=*/true) << ">";
  return ss.str();
}

// MLIR generator class.
class MlirGenerator {
 public:
  MlirGenerator(const Ort::ConstGraph& graph, std::ostream& out)
      : graph_(graph), out_(out) {}

  void Generate() {
    CollectMetadata();
    EmitModuleHeader();
    EmitFunctionBody();
    EmitModuleFooter();
  }

 private:
  void CollectMetadata() {
    // Get graph name.
    graph_name_ = SanitizeName(graph_.GetName());
    if (graph_name_.empty()) {
      graph_name_ = "main";
    }

    // Get IR version.
    ir_version_ = graph_.GetOnnxIRVersion();

    // Get opset version (find default domain).
    auto opsets = graph_.GetOperatorSets();
    for (const auto& opset : opsets) {
      if (opset.domain.empty() || opset.domain == "ai.onnx") {
        opset_version_ = opset.version;
        break;
      }
    }

    // Collect inputs (excluding initializers).
    auto inputs = graph_.GetInputs();
    auto initializers = graph_.GetInitializers();

    // Build set of initializer names.
    std::unordered_set<std::string> init_names;
    for (const auto& init : initializers) {
      init_names.insert(init.GetName());
    }

    // Graph inputs are those not in initializers.
    for (const auto& input : inputs) {
      std::string name = input.GetName();
      if (init_names.find(name) == init_names.end()) {
        graph_inputs_.push_back(input);
      }
    }

    // Graph outputs.
    graph_outputs_ = graph_.GetOutputs();

    // Initializers.
    initializers_ = initializers;
  }

  void EmitModuleHeader() {
    // Build function arguments.
    std::ostringstream args;
    for (size_t i = 0; i < graph_inputs_.size(); ++i) {
      if (i > 0) {
        args << ", ";
      }
      std::string name = SanitizeName(graph_inputs_[i].GetName());
      std::string type = FormatTensorType(graph_inputs_[i].TypeInfo());
      args << "%" << name << ": " << type;
    }

    // Build return types.
    std::ostringstream ret_types;
    for (size_t i = 0; i < graph_outputs_.size(); ++i) {
      if (i > 0) {
        ret_types << ", ";
      }
      ret_types << FormatTensorType(graph_outputs_[i].TypeInfo());
    }

    constexpr std::string_view schema = R"(module {{
  func.func @{0}({1}) -> ({2})
      attributes {{
        torch.onnx_meta.ir_version = {3} : si64,
        torch.onnx_meta.opset_version = {4} : si64,
        torch.onnx_meta.producer_name = "iree-onnx-ep",
        torch.onnx_meta.producer_version = ""
      }} {{
)";

    out_ << std::format(schema,
                        graph_name_,      // {0}
                        args.str(),       // {1}
                        ret_types.str(),  // {2}
                        ir_version_,      // {3}
                        opset_version_);  // {4}
  }

  void EmitFunctionBody() {
    // Emit initializers as onnx.Constant ops.
    for (const auto& init : initializers_) {
      EmitInitializer(init);
    }

    // Emit nodes.
    auto nodes = graph_.GetNodes();
    for (const auto& node : nodes) {
      EmitNode(node);
    }

    // Emit return.
    EmitReturn();
  }

  // Emits an initializer as an onnx.Constant operation with a dense_resource
  // attribute. The actual tensor data is not emitted here; instead, a reference
  // to a named resource blob is emitted. The blob data is emitted later in the
  // dialect_resources section by EmitDialectResources().
  //
  // Output format:
  //   %name = torch.operator "onnx.Constant"()
  //       {torch.onnx.value = dense_resource<name> : tensor<...>}
  //       : () -> !torch.vtensor<[...],dtype>
  void EmitInitializer(const Ort::ConstValueInfo& init) {
    std::string name = SanitizeName(init.GetName());
    std::string vtensor_type = FormatTensorType(init.TypeInfo());
    std::string tensor_type = FormatMlirTensorType(init.TypeInfo());

    // Emit dense_resource reference. Data will be emitted in dialect_resources.
    constexpr std::string_view schema =
        R"(    %{0} = torch.operator "onnx.Constant"() {{torch.onnx.value = dense_resource<{0}> : {1}}} : () -> {2}
)";
    out_ << std::format(schema, name, tensor_type, vtensor_type);
  }

  void EmitNode(const Ort::ConstNode& node) {
    std::string op_type = node.GetOperatorType();
    auto inputs = node.GetInputs();
    auto outputs = node.GetOutputs();
    auto attrs = node.GetAttributes();

    // Build output SSA names and types.
    std::ostringstream out_names;
    std::ostringstream out_types;
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (i > 0) {
        out_names << ", ";
        out_types << ", ";
      }
      out_names << "%" << SanitizeName(outputs[i].GetName());
      out_types << FormatTensorType(outputs[i].TypeInfo());
    }

    // Build input SSA references.
    std::ostringstream in_names;
    std::ostringstream in_types;
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (i > 0) {
        in_names << ", ";
        in_types << ", ";
      }
      in_names << "%" << SanitizeName(inputs[i].GetName());
      in_types << FormatTensorType(inputs[i].TypeInfo());
    }

    // Build attributes.
    std::string attr_str = FormatAttributes(attrs);

    // Emit the operator.
    constexpr std::string_view schema =
        R"(    {0} = torch.operator "onnx.{1}"({2}) {{{3}}} : ({4}) -> {5}
)";
    out_ << std::format(schema,
                        out_names.str(),   // {0}
                        op_type,           // {1}
                        in_names.str(),    // {2}
                        attr_str,          // {3}
                        in_types.str(),    // {4}
                        out_types.str());  // {5}
  }

  std::string FormatAttributes(const std::vector<Ort::ConstOpAttr>& attrs) {
    if (attrs.empty()) {
      return "";
    }

    std::ostringstream ss;
    for (size_t i = 0; i < attrs.size(); ++i) {
      if (i > 0) {
        ss << ", ";
      }
      ss << FormatAttribute(attrs[i]);
    }
    return ss.str();
  }

  std::string FormatAttribute(const Ort::ConstOpAttr& attr) {
    std::string name = attr.GetName();
    OrtOpAttrType type = attr.GetType();

    switch (type) {
      case ORT_OP_ATTR_INT: {
        int64_t value = 0;
        attr.GetValue(value);
        return std::format("torch.onnx.{0} = {1} : si64", name, value);
      }
      case ORT_OP_ATTR_FLOAT: {
        float value = 0.0f;
        attr.GetValue(value);
        return std::format("torch.onnx.{0} = {1:e} : f32", name, value);
      }
      case ORT_OP_ATTR_STRING: {
        std::string value;
        attr.GetValue(value);
        return std::format("torch.onnx.{0} = \"{1}\"", name, value);
      }
      case ORT_OP_ATTR_INTS: {
        std::vector<int64_t> values;
        attr.GetValueArray<int64_t>(values);
        std::vector<std::string> str_values(values.size());
        std::transform(values.begin(), values.end(), str_values.begin(),
                       [](int64_t v) { return std::format("{0} : si64", v); });
        return std::format("torch.onnx.{0} = [{1}]", name,
                           Join(str_values, ", "));
      }
      default:
        return std::format("torch.onnx.{0} = \"NYI\"", name);
    }
  }

  void EmitReturn() {
    std::ostringstream ret_values;
    std::ostringstream ret_types;
    for (size_t i = 0; i < graph_outputs_.size(); ++i) {
      if (i > 0) {
        ret_values << ", ";
        ret_types << ", ";
      }
      ret_values << "%" << SanitizeName(graph_outputs_[i].GetName());
      ret_types << FormatTensorType(graph_outputs_[i].TypeInfo());
    }

    out_ << std::format("    return {0} : {1}\n", ret_values.str(),
                        ret_types.str());
  }

  // Emits the dialect_resources section containing the raw tensor data for all
  // initializers. This section is appended after the module closing brace.
  //
  // Each initializer's data is encoded as a hex string with the following
  // format (per MLIR's AsmResourceBlob specification):
  //   - First 4 bytes: alignment as little-endian uint32 (e.g., 0x04000000 for
  //     f32 which requires 4-byte alignment)
  //   - Remaining bytes: raw tensor data in little-endian byte order
  //
  // Output format:
  //   {-#
  //     dialect_resources: {
  //       builtin: {
  //         name1: "0x04000000...",
  //         name2: "0x08000000..."
  //       }
  //     }
  //   #-}
  void EmitDialectResources() {
    if (initializers_.empty()) {
      return;
    }

    constexpr std::string_view header = R"(
{-#
  dialect_resources: {
    builtin: {
)";
    out_ << header;

    bool first = true;
    for (const auto& init : initializers_) {
      Ort::ConstValue tensor_value{nullptr};
      auto status = init.GetInitializer(tensor_value);
      if (!status.IsOK()) {
        continue;
      }

      std::string name = SanitizeName(init.GetName());
      const auto* data =
          static_cast<const uint8_t*>(tensor_value.GetTensorRawData());
      size_t size = tensor_value.GetTensorSizeInBytes();

      // Get alignment from element type (e.g., 4 for f32, 8 for f64).
      auto tensor_info = init.TypeInfo().GetTensorTypeAndShapeInfo();
      uint32_t alignment = static_cast<uint32_t>(
          GetElementByteSize(tensor_info.GetElementType()));

      // Build hex string: "0x" + alignment (4 bytes LE) + data bytes.
      std::string hex_str;
      hex_str.reserve(2 + 8 + size * 2);
      hex_str = "0x";
      constexpr char hex_chars[] = "0123456789abcdef";

      // Append alignment as little-endian 32-bit integer.
      for (int i = 0; i < 4; ++i) {
        uint8_t byte = (alignment >> (i * 8)) & 0xFF;
        hex_str += hex_chars[(byte >> 4) & 0xF];
        hex_str += hex_chars[byte & 0xF];
      }

      // Append raw tensor data bytes.
      for (size_t i = 0; i < size; ++i) {
        hex_str += hex_chars[(data[i] >> 4) & 0xF];
        hex_str += hex_chars[data[i] & 0xF];
      }

      if (!first) {
        out_ << ",\n";
      }
      first = false;

      out_ << std::format("      {}: \"{}\"", name, hex_str);
    }

    constexpr std::string_view footer = R"(
    }
  }
#-}
)";
    out_ << footer;
  }

  void EmitModuleFooter() {
    out_ << "  }\n";
    out_ << "}\n";
    EmitDialectResources();
  }

  // Member variables.
  const Ort::ConstGraph& graph_;
  std::ostream& out_;

  std::string graph_name_;
  int64_t ir_version_ = 8;
  int64_t opset_version_ = 17;

  std::vector<Ort::ConstValueInfo> graph_inputs_;
  std::vector<Ort::ConstValueInfo> graph_outputs_;
  std::vector<Ort::ConstValueInfo> initializers_;
};

}  // namespace

OrtStatus* GenerateMlir(const Ort::ConstGraph& graph, const OrtApi& /*ort_api*/,
                        const std::string& output_path) {
  std::ofstream file(output_path);
  if (!file.is_open()) {
    return Ort::Status(
               std::format("Failed to open output file: {}", output_path)
                   .c_str(),
               ORT_FAIL)
        .release();
  }

  MlirGenerator gen(graph, file);
  gen.Generate();

  file.close();
  if (file.fail()) {
    return Ort::Status(
               std::format("Failed to write to file: {}", output_path).c_str(),
               ORT_FAIL)
        .release();
  }

  return nullptr;
}

}  // namespace iree_onnx_ep
