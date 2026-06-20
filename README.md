Here is the complete, production-grade `README.md` for your **`Prostate_Lesion_Segmentation_Unet`** repository. It is self-contained, fully formatted, and ready to be pasted directly into GitHub.

---

```markdown
# Prostate and Lesion Segmentation Using Dual-Stage U-Net Pipeline with Vitis AI

This repository contains the end-to-end deep learning training pipelines, model optimization blocks, and hardware deployment files for a dual-stage medical imaging workflow. The system segments the prostate gland and detects/localizes cancerous lesions from multi-sequence MRI scans (T2W, HBV, and ADC). 

The project demonstrates a complete machine learning lifecycle: transitioning from **Software-based Exploration (PyTorch/TensorFlow)** to **Post-Training Quantization (Vitis AI)**, and finally to **Edge Hardware Acceleration (Deep Learning Processing Unit - DPU)**.

---

## 📂 Repository Structure & File Directory

The code workspace is structured into a clean deployment hierarchy:

```text
├── 1_software_training/
│   ├── UNet_Prostate_Training.html   # Full Stage 1 training (PyTorch)
│   ├── UNet_Lesion_Training.html     # Full Stage 2 training (TensorFlow)
│   └── dataset.py                    # Custom multi-sequence data loader (TF)
│
├── 2_hardware_compilation/
│   ├── inspect_tf_model.py           # DPU ISA compatibility checker
│   ├── quantize_tf2.py               # Post-Training Quantizer (FP32 -> INT8)
│   └── vart_tensor_info.py           # Compiled .xmodel tensor debugger
│
└── 3_hardware_deployment/
    └── DPU_TwoStage_Inference.html   # Live dual-stage edge driver script

```

---

## 📌 Multi-Stage Pipeline Architecture

The system processes raw, multi-modal prostate MRI inputs through a sequential cascading framework to maximize segmentation accuracy on tiny target anomalies:

1. **Stage 1: Prostate Gland Segmentation (`UNet_Prostate_Training.html`)**
* **Framework:** PyTorch
* **Role:** Detects and segments the outer boundaries of the entire prostate gland to isolate the target organ from the surrounding pelvic background anatomy.
* **Validation Metrics:** Dice: `0.8795` | IoU: `0.7867` | Precision: `0.8373` | Recall: `0.9301`


2. **Stage 2: Lesion Segmentation (`UNet_Lesion_Training.html`)**
* **Framework:** TensorFlow / Keras
* **Role:** Processes a zoomed, localized patch centered around the segmented prostate region to isolate complex clinical anomalies and lesions.
* **Validation Metrics:** Dice: `0.9228` | IoU: `0.8746` | Precision: `0.9380` | Recall: `0.9300`



---

## 🛠️ Data Engineering (`dataset.py`)

Medical imaging sequences (T2W, High b-value, and ADC maps) possess varying signal intensities. The custom data runner (`MriLesionDatasetTF`) ensures mathematical parity across both stages by executing:

* **Z-Score Normalization:** Standardizes inputs dynamically using structural slice channel statistics.
* **Lesion-Centered Cropping & Padding:** Crops stable patch windows (`144x144`) centered on known coordinates to maintain consistent spatial resolution without distorting underlying pathology features.

---

## 🚀 Vitis AI Hardware Optimization & Compilation

To shift floating-point models (`float32`) onto embedded silicon without destroying accuracy, the model undergoes a strict hardware optimization pipeline:

### 1. Compatibility Inspection (`inspect_tf_model.py`)

Utilizes the `VitisInspector` to cross-examine the `lesion_unet_tf.h5` graph topology against the target instruction set architecture (**`DPUCADF8H_ISA0`**). This catches unsupported custom layers or software activation blocks before deploying.

### 2. Post-Training Quantization (`quantize_tf2.py`)

Converts model weights and biases from heavy 32-bit floating-point structures down to highly efficient 8-bit integers (`int8`) using the calibration dataset. This step exports the final compiled hardware execution binary: **`lesion_unet.xmodel`**.

### 3. Tensor Geometry Verification (`vart_tensor_info.py`)

Uses the Vitis AI Runtime (`vart`) and Xilinx Intermediate Representation (`xir`) to deserialize the compiled output binary. It explicitly prints fixed-point scaling properties (`fix_point`) and expected DPU dimension structures, preventing deployment buffer-overflow errors.

---

## ⏱️ Edge Hardware Performance Benchmark

The compiled networks run as a single execution flow monitored by the edge runtime engine (**`DPU_TwoStage_Inference.html`**).

The production hardware logs show excellent throughput:

* **Stage 1 (Prostate Core) DPU Latency:** `25.37 ms`
* **Stage 2 (Lesion Tracker) DPU Latency:** `11.04 ms`
* **Total DPU Processing Window:** `36.41 ms`
* **Real-Time Throughput Speed:** **`27.47 FPS`** (Samples/sec)

---

## 💻 Environment Setup & Quick Start

### Software Dependencies

* Python 3.8+
* PyTorch / TensorFlow 2.x
* AMD / Xilinx Vitis AI Development Kit (v3.x suggested)
* OpenCV, NumPy, Matplotlib

### Local Inspection

To inspect individual steps or convert the exported HTML notebooks back into active development notebooks, run:

```bash
jupyter nbconvert --to notebook 1_software_training/UNet_Prostate_Training.html

```

```

```
