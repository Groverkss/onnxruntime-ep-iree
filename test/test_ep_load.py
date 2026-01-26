#!/usr/bin/env python3
"""Test that the IREE ONNX Runtime EP loads and runs correctly."""

import sys
import pathlib
import tempfile
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort


def test_ep_load():
    """Test that the IREE EP plugin loads and runs inference correctly."""

    # Get the path to the built EP library
    project_root = pathlib.Path(__file__).parent.parent
    ep_lib_path = project_root / "build" / "libiree_onnx_ep.so"

    if not ep_lib_path.exists():
        print(f"ERROR: EP library not found at {ep_lib_path}")
        print("Please build the project first:")
        print("  mkdir build && cd build")
        print("  cmake .. -GNinja")
        print("  ninja")
        return False

    print(f"EP library path: {ep_lib_path}")

    # Register the EP plugin
    ort.register_execution_provider_library("IREE", str(ep_lib_path))
    print("EP plugin registered successfully")

    # Get IREE device
    iree_driver = "vulkan"
    ep_devices = ort.get_ep_devices()
    iree_device = None
    for dev in ep_devices:
        print(dev.device.device_id)
        if dev.device.metadata.get("iree.driver") == iree_driver:
            iree_device = dev
            break

    if not iree_device:
        print("ERROR: IREE EP not found in EP devices")
        return False

    print(f"IREE EP available (vendor={iree_device.ep_vendor})")
    print("Device metadata:", iree_device.device.metadata)
    print("Device type:", iree_device.device.type)

    a = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]], dtype=np.float32)
    b = np.array([[10, 20, 30, 40],
                  [50, 60, 70, 80],
                  [90, 100, 110, 120]], dtype=np.float32)

    # Create a simple Add model
    input_a = helper.make_tensor_value_info('A', TensorProto.FLOAT, [3, 4])
    # input_b = helper.make_tensor_value_info('B', TensorProto.FLOAT, [3, 4])
    output = helper.make_tensor_value_info('C', TensorProto.FLOAT, [3, 4])
    constant_node = helper.make_node('Constant', inputs=[], outputs=['D'],
                                     value=helper.make_tensor(name='const_tensor',
                                                              data_type=TensorProto.FLOAT,
                                                              dims=[3, 4],
                                                              vals=b.flatten().tolist()))
    add_node = helper.make_node('Add', inputs=['A', 'D'], outputs=['C'])

    graph = helper.make_graph([add_node, constant_node], 'test_graph', [input_a], [output])
    model = helper.make_model(graph, producer_name='iree_test',
                              opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = 8

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        model_path = f.name
        onnx.save(model, model_path)

    try:
        # Create session with IREE EP
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 0

        # Provider options for IREE EP.
        provider_options = {
            "target_arch": "vulkan-spirv",
            "save_intermediates": "1",
        }
        sess_options.add_provider_for_devices([iree_device], provider_options)

        session = ort.InferenceSession(model_path, sess_options=sess_options)

        # Prepare inputs
        expected = a + b

        # Run inference
        outputs = session.run(None, {'A': a})
        result = outputs[0]

        # Verify output
        if result.shape != expected.shape:
            print(f"FAIL: Shape mismatch: {result.shape} vs {expected.shape}")
            return False

        if not np.allclose(result, expected, rtol=1e-5, atol=1e-5):
            print(f"FAIL: Values mismatch")
            print(f"  Expected: {expected}")
            print(f"  Got: {result}")
            return False

        print("Inference output verified!")
        print(f"  Input A: {a.flatten()[:4]}...")
        print(f"  Input B: {b.flatten()[:4]}...")
        print(f"  Output:  {result.flatten()[:4]}...")

    finally:
        pathlib.Path(model_path).unlink()

    print("\n=== Test PASSED ===")
    return True


if __name__ == "__main__":
    success = test_ep_load()
    sys.exit(0 if success else 1)
