import os
import tensorflow as tf
import numpy as np

from tensorflow_model_optimization.quantization.keras import vitis_quantize

# --------------------------------------------------
# IMPORT YOUR DATASET (UNCHANGED)
# --------------------------------------------------
from dataset import MriLesionDatasetTF   # <-- your class

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
ROOT_DIR     = "/workspace/Project_MRI/prostate_cancer-20260114T070259Z-1-001/prostate_cancer"
FLOAT_MODEL  = "lesion_unet_tf.h5"
OUT_DIR      = "quantize_result"

BATCH_SIZE   = 4
CALIB_STEPS  = 100     # 100–200 is ideal

os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------
# LOAD FLOAT MODEL
# --------------------------------------------------
model = tf.keras.models.load_model(
    FLOAT_MODEL,
    compile=False
)

print("✅ Float TF model loaded")

# --------------------------------------------------
# CALIBRATION DATASET (EXACT SAME AS TRAINING)
# --------------------------------------------------
calib_dataset = MriLesionDatasetTF(
    ROOT_DIR,
    batch_size=BATCH_SIZE,
    patch_size=144
)

print("✅ Calibration dataset ready")
print("Calibration samples:", len(calib_dataset.samples))

# --------------------------------------------------
# CREATE VITIS QUANTIZER
# --------------------------------------------------
quantizer = vitis_quantize.VitisQuantizer(model)

# --------------------------------------------------
# POST-TRAINING QUANTIZATION
# --------------------------------------------------
quantized_model = quantizer.quantize_model(
    calib_dataset=calib_dataset,
    calib_steps=CALIB_STEPS
)

print("✅ Quantization completed")

# --------------------------------------------------
# SAVE INT8 KERAS MODEL
# --------------------------------------------------
quantized_model.save(
    os.path.join(OUT_DIR, "lesion_unet_int8.h5"),
    include_optimizer=False
)

print("💾 INT8 model saved (.h5)")

# --------------------------------------------------
# EXPORT XMODEL FOR DPU
# --------------------------------------------------
quantizer.export_xmodel(
    output_dir=OUT_DIR
)

print("🚀 XMODEL exported successfully")

