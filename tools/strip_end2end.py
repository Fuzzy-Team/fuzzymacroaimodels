"""Strip the TopK end2end tail from YOLO26 ONNX exports so OpenCV < 4.11 can load them.

OpenCV added the TopK layer in 4.11, and Intel Macs below macOS 13 are pinned to
OpenCV 4.10. The one2one head's decoded boxes (xyxy, pixels) and sigmoid class
scores are re-emitted as classic YOLO output [1, 4+nc, N] with cx/cy/w/h boxes,
which the macro decodes with NMS. Detections match the end2end output.

Usage: python tools/strip_end2end.py exported.onnx token_detection_standard.onnx
Needs: onnx, onnxruntime, numpy
"""
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper


def strip(src, dst):
    model = onnx.load(src)
    graph = model.graph
    boxes = "/model.23/Mul_2_output_0"     # [1, 4, N] xyxy * stride
    scores = "/model.23/Sigmoid_output_0"  # [1, nc, N]
    produced = {o for n in graph.node for o in n.output}
    assert boxes in produced and scores in produced, "unexpected head layout"

    def const(name, values):
        graph.initializer.append(numpy_helper.from_array(np.array(values, dtype=np.int64), name))

    const("fz_s0", [0]); const("fz_s2", [2]); const("fz_s4", [4]); const("fz_ax", [1])
    graph.initializer.append(numpy_helper.from_array(np.array(0.5, dtype=np.float32), "fz_half"))
    new_nodes = [
        helper.make_node("Slice", [boxes, "fz_s0", "fz_s2", "fz_ax"], ["fz_xy1"]),
        helper.make_node("Slice", [boxes, "fz_s2", "fz_s4", "fz_ax"], ["fz_xy2"]),
        helper.make_node("Add", ["fz_xy1", "fz_xy2"], ["fz_sum"]),
        helper.make_node("Mul", ["fz_sum", "fz_half"], ["fz_cxcy"]),
        helper.make_node("Sub", ["fz_xy2", "fz_xy1"], ["fz_wh"]),
        helper.make_node("Concat", ["fz_cxcy", "fz_wh", scores], ["output0"], axis=1),
    ]

    # Rename the old end2end output so the new node can own "output0".
    for node in graph.node:
        node.output[:] = ["fz_old_output0" if o == "output0" else o for o in node.output]
    graph.node.extend(new_nodes)
    del graph.output[:]
    graph.output.append(helper.make_tensor_value_info("output0", onnx.TensorProto.FLOAT, None))

    # Drop every node that no longer feeds output0 (the TopK tail).
    needed = {"output0"}
    keep = []
    for node in reversed(list(graph.node)):
        if any(o in needed for o in node.output):
            keep.append(node)
            needed.update(i for i in node.input if i)
    keep.reverse()
    del graph.node[:]
    graph.node.extend(keep)
    live_inits = [i for i in graph.initializer if i.name in needed]
    del graph.initializer[:]
    graph.initializer.extend(live_inits)
    del graph.value_info[:]

    _fold_shape_math(model)

    for prop in model.metadata_props:
        if prop.key == "end2end":
            prop.value = "False"
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, dst)


def _fold_shape_math(model):
    """Replace Shape-derived subgraphs with constants (OpenCV 4.6 can't parse them).

    The input size is fixed, so every value computed from Shape is constant.
    """
    import onnxruntime as ort

    graph = model.graph
    static = {i.name for i in graph.initializer}
    folded = []
    for node in graph.node:
        inputs = [i for i in node.input if i]
        if node.op_type == "Shape" or node.op_type == "Constant" or (inputs and all(i in static for i in inputs)):
            static.update(node.output)
            if node.op_type != "Constant":
                folded.append(node)
    consumers = {i for n in graph.node if n not in folded and n.op_type != "Constant" for i in n.input}
    targets = [o for n in folded for o in n.output if o in consumers]
    if not targets:
        return

    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    probe.graph.output.extend(onnx.ValueInfoProto(name=t) for t in targets)
    session = ort.InferenceSession(probe.SerializeToString())
    feed = {}
    for inp in session.get_inputs():
        feed[inp.name] = np.zeros(inp.shape, dtype=np.float32)
    values = session.run(targets, feed)

    for name, value in zip(targets, values):
        graph.initializer.append(numpy_helper.from_array(np.asarray(value), name))
    folded_outputs = {o for n in folded for o in n.output}
    keep = [n for n in graph.node if not (set(n.output) & folded_outputs)]
    del graph.node[:]
    graph.node.extend(keep)
    used = {i for n in graph.node for i in n.input}
    live = [i for i in graph.initializer if i.name in used]
    del graph.initializer[:]
    graph.initializer.extend(live)
    # Constant nodes that only fed folded math are now dead.
    live_nodes = [n for n in graph.node if n.op_type != "Constant" or set(n.output) & used]
    del graph.node[:]
    graph.node.extend(live_nodes)


if __name__ == "__main__":
    strip(sys.argv[1], sys.argv[2])
