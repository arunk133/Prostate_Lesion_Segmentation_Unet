import xir

xmodel_path = "compile_result/lesion_unet.xmodel"

graph = xir.Graph.deserialize(xmodel_path)
root = graph.get_root_subgraph()

# --------------------------------------------------
# Find DPU subgraph
# --------------------------------------------------
dpu_subgraph = None
for sg in root.get_children():
    if sg.has_attr("device") and sg.get_attr("device") == "DPU":
        dpu_subgraph = sg
        break

assert dpu_subgraph is not None, "❌ No DPU subgraph found"

print("✅ DPU Subgraph:", dpu_subgraph.get_name())

# --------------------------------------------------
# INPUT tensors
# --------------------------------------------------
print("\n=== INPUT TENSORS ===")
for t in dpu_subgraph.get_input_tensors():
    print("Name :", t.name)
    print("Shape:", t.dims)
    print("Type :", t.dtype)              # ✅ correct
    print("Fix  :", t.get_attr("fix_point"))
    print()

# --------------------------------------------------
# OUTPUT tensors
# --------------------------------------------------
print("=== OUTPUT TENSORS ===")
for t in dpu_subgraph.get_output_tensors():
    print("Name :", t.name)
    print("Shape:", t.dims)
    print("Type :", t.dtype)              # ✅ correct
    print("Fix  :", t.get_attr("fix_point"))
    print()

