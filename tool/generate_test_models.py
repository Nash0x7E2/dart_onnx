"""
Generates tiny, lightweight ONNX models specifically for unit testing the Dart ONNX package.
These models should be trivial mathematically to make predicting their output from Dart easy.
"""

import onnx
from onnx import helper, TensorProto
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "test", "assets", "models")


def make_identity_model() -> None:
    # A simple model: Just an identity mapping (Input -> Output)
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2, 2])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2, 2])

    node = helper.make_node(
        "Identity",
        inputs=["X"],
        outputs=["Y"],
        name="IdentityNode",
    )

    graph = helper.make_graph(
        nodes=[node],
        name="identity_graph",
        inputs=[X],
        outputs=[Y],
    )

    model = helper.make_model(graph, producer_name="dart_onnx_test_generator")
    onnx.checker.check_model(model)

    out_path = os.path.join(OUT_DIR, "identity.onnx")
    onnx.save(model, out_path)
    print(f"Generated identity.onnx to {out_path}")


def make_add_model() -> None:
    # A simple addition model: A + B = C
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [3])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [3])
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [3])

    node = helper.make_node(
        "Add",
        inputs=["A", "B"],
        outputs=["C"],
        name="AddNode",
    )

    graph = helper.make_graph(
        nodes=[node],
        name="add_graph",
        inputs=[A, B],
        outputs=[C],
    )

    model = helper.make_model(graph, producer_name="dart_onnx_test_generator")
    onnx.checker.check_model(model)

    out_path = os.path.join(OUT_DIR, "add.onnx")
    onnx.save(model, out_path)
    print(f"Generated add.onnx to {out_path}")


def make_multi_output_model() -> None:
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
    Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [None, None])

    identity_y = helper.make_node("Identity", ["X"], ["Y"])
    identity_z = helper.make_node("Identity", ["X"], ["Z"])

    graph = helper.make_graph([identity_y, identity_z], "multi_output", [X], [Y, Z])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    out_path = os.path.join(OUT_DIR, "multi_output.onnx")
    onnx.save(model, out_path)
    print(f"Generated multi_output.onnx to {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    make_identity_model()
    make_add_model()
    make_multi_output_model()


if __name__ == "__main__":
    main()
