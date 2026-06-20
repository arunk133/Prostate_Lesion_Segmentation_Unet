import tensorflow as tf
from tensorflow_model_optimization.quantization.keras import vitis_inspect

# --------------------------------------------------
# Load TF model (.h5)
# --------------------------------------------------
model = tf.keras.models.load_model(
    "lesion_unet_tf.h5",
    compile=False
)

print("✔ Model loaded")

# --------------------------------------------------
# Create Inspector
# (Use your DPU target – change only if needed)
# --------------------------------------------------
inspector = vitis_inspect.VitisInspector(
    target="DPUCADF8H_ISA0"
)

# --------------------------------------------------
# Run inspection
# --------------------------------------------------
inspector.inspect_model(
    model,
    plot=True,
    plot_file="lesion_unet_inspect.png",   # <-- PNG OUTPUT
    dump_results=True,
    dump_results_file="inspect_results.txt",
    verbose=1
)

print("✔ Inspection completed")
print("📷 PNG saved as lesion_unet_inspect.png")

