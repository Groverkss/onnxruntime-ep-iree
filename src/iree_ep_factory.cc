// iree_ep_factory.cc - IREE Execution Provider Factory Implementation

#include "iree_ep_factory.h"

#include <memory>

#include "iree_ep.h"

namespace iree_onnx_ep {

IreeEpFactory::IreeEpFactory(const char* ep_name, ApiPtrs apis,
                             const OrtLogger* default_logger)
    : OrtEpFactory{},
      ApiPtrs(apis),
      logger_(default_logger),
      ep_name_(ep_name) {
  // Set ORT version we support.
  ort_version_supported = ORT_API_VERSION;

  // Set function pointers.
  GetName = GetNameImpl;
  GetVendor = GetVendorImpl;
  GetVendorId = GetVendorIdImpl;
  GetVersion = GetVersionImpl;
  GetSupportedDevices = GetSupportedDevicesImpl;
  CreateEp = CreateEpImpl;
  ReleaseEp = ReleaseEpImpl;
  CreateAllocator = CreateAllocatorImpl;
  ReleaseAllocator = ReleaseAllocatorImpl;
  CreateDataTransfer = CreateDataTransferImpl;
  IsStreamAware = IsStreamAwareImpl;
  CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;

  // Initialize IREE runtime instance (shared across all EPs).
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);

  iree_status_t status = iree_runtime_instance_create(
      &instance_options, iree_allocator_system(), instance_.ForOutput());
  if (!iree_status_is_ok(status)) {
    // Log error but continue - EP creation will fail later if needed.
    ORT_CXX_LOG_NOEXCEPT(logger_, ORT_LOGGING_LEVEL_ERROR,
                         "IREE EP: Failed to create IREE runtime instance");
    iree_status_ignore(status);
  }
}

// Factory interface implementations

/*static*/
const char* ORT_API_CALL
IreeEpFactory::GetNameImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const IreeEpFactory*>(this_ptr);
  return factory->ep_name_.c_str();
}

/*static*/
const char* ORT_API_CALL
IreeEpFactory::GetVendorImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
  return kEpVendor;
}

/*static*/
uint32_t ORT_API_CALL
IreeEpFactory::GetVendorIdImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
  return kEpVendorId;
}

/*static*/
const char* ORT_API_CALL
IreeEpFactory::GetVersionImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
  return kEpVersion;
}

/*static*/
OrtStatus* ORT_API_CALL IreeEpFactory::GetSupportedDevicesImpl(
    OrtEpFactory* this_ptr, const OrtHardwareDevice* const* devices,
    size_t num_devices, OrtEpDevice** ep_devices, size_t max_ep_devices,
    size_t* p_num_ep_devices) noexcept {
  auto* factory = static_cast<IreeEpFactory*>(this_ptr);
  size_t& num_ep_devices = *p_num_ep_devices;
  num_ep_devices = 0;

  // Report supported devices.
  for (size_t i = 0; i < num_devices && num_ep_devices < max_ep_devices; ++i) {
    const OrtHardwareDevice& device = *devices[i];
    OrtHardwareDeviceType device_type =
        factory->ort_api.HardwareDevice_Type(&device);
    // Report support for CPU and GPU devices. We could really also report
    // support for NPU devices if we ship some NPU drivers with IREE, but we
    // don't today afaik.
    bool supported = device_type == OrtHardwareDeviceType_CPU ||
                     device_type == OrtHardwareDeviceType_GPU;
    if (supported) {
      OrtEpDevice* ep_device = nullptr;
      ORT_RETURN_IF_ERROR(factory->ep_api.CreateEpDevice(
          factory, &device, nullptr, nullptr, &ep_device));
      ep_devices[num_ep_devices++] = ep_device;
    }
  }

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL IreeEpFactory::CreateEpImpl(
    OrtEpFactory* this_ptr, const OrtHardwareDevice* const* devices,
    const OrtKeyValuePairs* const* /*ep_options*/, size_t num_devices,
    const OrtSessionOptions* session_options, const OrtLogger* logger,
    OrtEp** ep) noexcept {
  auto* factory = static_cast<IreeEpFactory*>(this_ptr);
  *ep = nullptr;

  // Validate device selection.
  if (num_devices != 1) {
    // TODO: Can we even do multi-device work with onnxruntime? Afaik, there is
    // multi-device graph partitioning support:
    // https://github.com/microsoft/onnxruntime/issues/16382
    //
    // The other option would be to partition the graph ourselves in IREE. That
    // seems like a whole other line of work which is very out-of-scope for now.
    return Ort::Status("NYI: Multi-device IREE EP", ORT_NOT_IMPLEMENTED)
        .release();
  }

  // Parse configuration from session options config entries.
  // The Python API stores options with "ep.iree." prefix.
  IreeEp::Config config = {};
  if (session_options != nullptr) {
    Ort::ConstSessionOptions sess_opts(session_options);
    config.device = sess_opts.GetConfigEntryOrDefault("ep.iree.device", "");
    config.target_arch =
        sess_opts.GetConfigEntryOrDefault("ep.iree.target_arch", "");
    config.opt_level =
        sess_opts.GetConfigEntryOrDefault("ep.iree.opt_level", "O0");
    config.save_intermediates =
        sess_opts.GetConfigEntryOrDefault("ep.iree.save_intermediates", "0") ==
        "1";
  }

  ORT_CXX_LOGF_NOEXCEPT(factory->logger_, ORT_LOGGING_LEVEL_INFO,
                        "IREE EP: Creating EP with device='%s', "
                        "target_arch='%s', opt_level='%s'",
                        config.device.c_str(), config.target_arch.c_str(),
                        config.opt_level.c_str());

  // Perform configuration validation.
  //
  // 1. Require a device to be specified.
  // 2. Require target_arch to be specified for non-CPU devices.
  if (config.device.empty()) {
    return Ort::Status("IREE EP: 'device' option must be specified",
                       ORT_INVALID_ARGUMENT)
        .release();
  }
  const OrtHardwareDevice& hardware_device = *devices[0];
  OrtHardwareDeviceType device_type =
      factory->ort_api.HardwareDevice_Type(&hardware_device);
  if (device_type != OrtHardwareDeviceType_CPU && config.target_arch.empty()) {
    return Ort::Status(
               "IREE EP: 'target_arch' option must be specified for non-CPU "
               "devices",
               ORT_INVALID_ARGUMENT)
        .release();
  }

  // Get device from the device string: device(://id)? to device.
  std::string driver_string = config.device;
  size_t pos = driver_string.find("://");
  if (pos != std::string::npos) {
    driver_string = driver_string.substr(0, pos);
  }

  // Select the backend based on the driver. We could in future expose this
  // as an user option.
  if (config.backend.empty()) {
    if (driver_string == "vulkan") {
      config.backend = "vulkan";
    } else if (driver_string == "cuda") {
      config.backend = "cuda";
    } else if (driver_string == "hip") {
      config.backend = "hip";
    } else if (driver_string == "local-task" || driver_string == "local-sync") {
      config.backend = "llvm-cpu";
    } else {
      return Ort::Status("IREE EP: Unknown driver", ORT_INVALID_ARGUMENT)
          .release();
    }
  }

  // Create IREE HAL device.
  // TODO: This is bad, we should be using iree_hal_driver_create_device_by_uri
  // directly, otherwise we are not respecting the uri format the user gave us.
  HalDevicePtr hal_device;
  iree_status_t status = iree_runtime_instance_try_create_default_device(
      factory->IreeInstance(), iree_make_cstring_view(driver_string.c_str()),
      hal_device.ForOutput());
  if (!iree_status_is_ok(status)) {
    return IreeStatusToOrtStatus(status);
  }

  // Create the EP instance with the device.
  auto ep_instance = std::make_unique<IreeEp>(
      *factory, factory->ep_name_, config, *logger, std::move(hal_device));

  *ep = ep_instance.release();
  return nullptr;
}

/*static*/
void ORT_API_CALL IreeEpFactory::ReleaseEpImpl(OrtEpFactory* /*this_ptr*/,
                                               OrtEp* ep) noexcept {
  delete static_cast<IreeEp*>(ep);
}

/*static*/
OrtStatus* ORT_API_CALL IreeEpFactory::CreateAllocatorImpl(
    OrtEpFactory* /*this_ptr*/, const OrtMemoryInfo* /*memory_info*/,
    const OrtKeyValuePairs* /*allocator_options*/,
    OrtAllocator** allocator) noexcept {
  // No custom allocator yet
  *allocator = nullptr;
  return nullptr;
}

/*static*/
void ORT_API_CALL IreeEpFactory::ReleaseAllocatorImpl(
    OrtEpFactory* /*this_ptr*/, OrtAllocator* /*allocator*/) noexcept {
  // Nothing to do
}

/*static*/
OrtStatus* ORT_API_CALL IreeEpFactory::CreateDataTransferImpl(
    OrtEpFactory* /*this_ptr*/, OrtDataTransferImpl** data_transfer) noexcept {
  // No custom data transfer yet
  *data_transfer = nullptr;
  return nullptr;
}

/*static*/
bool ORT_API_CALL
IreeEpFactory::IsStreamAwareImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
  // Not stream aware yet (synchronous execution only)
  return false;
}

/*static*/
OrtStatus* ORT_API_CALL IreeEpFactory::CreateSyncStreamForDeviceImpl(
    OrtEpFactory* /*this_ptr*/, const OrtMemoryDevice* /*memory_device*/,
    const OrtKeyValuePairs* /*stream_options*/,
    OrtSyncStreamImpl** stream) noexcept {
  // Not stream aware
  *stream = nullptr;
  return nullptr;
}

}  // namespace iree_onnx_ep
