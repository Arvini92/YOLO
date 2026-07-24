# AeroWheat - YOLO Wheat Leaf Disease Detection & ONNX Export Pipeline

An end-to-end computer vision pipeline for downloading annotated wheat leaf disease datasets from **Roboflow**, fine-tuning a **YOLO** object detection model on local GPU hardware, and exporting the trained weights to **ONNX** format for browser-based inference.

---

## Features

- **Automated Dataset Sync:** Fetches the targeted `yolov8` formatted dataset version directly via Roboflow API.
- **Model Training:** Trains object detection models on custom dataset splits with dynamic path tracking for save directories.
- **Dynamic Checkpoint Resolution:** Reads `model.trainer.save_dir` at runtime to isolate the target `best.pt` file regardless of run auto-incrementing.
- **Web-Ready Export:** Converts trained PyTorch weights into ONNX format (`imgsz=640`) for client-side WebGPU/WASM execution.

---

## Prerequisites & Setup

### 1. Environment Variables

Create a `.env` file in the root directory and add your Roboflow API key:

```env
ROBOFLOW_API_KEY=your_roboflow_api_key_here
```

2. Python Dependencies
Install the required packages using pip:
```bash
pip install -r requirements.txt
```

Note: Ensure your local PyTorch installation has CUDA enabled if you plan to train on a local GPU (device=0).

Usage
Save the script as yolo.py and run it from your terminal:
```bash
python yolo.py
```

Execution Steps
Dataset Download: Connects to Roboflow (wheatleaf/wheat-leaf-disease-detection) and extracts the dataset configuration (data.yaml).

Model Training: Loads base YOLO weights, trains for 50 epochs at a resolution of 640x640, and logs runs under ./AeroWheat/yolo26_wheat_leaf.

Dynamic Resolution: Resolves the active training output folder to grab the newly optimized best.pt weights.

ONNX Export: Converts the PyTorch checkpoint to best.onnx optimized for client-side applications.

Configuration Reference
Key training parameters configured in model.train():
| Parameter | Value | Description |
| :--- | :--- | :--- |
| `data` | `data_yaml_path` | Path to dataset YAML configuration |
| `epochs` | `50` | Total training passes |
| `imgsz` | `640` | Input image dimension |
| `device` | `0` | GPU device index |
| `batch` | `8` | Training batch size |
| `project` | `"AeroWheat"` | Output project directory name |
| `name` | `"yolo26_wheat_leaf"` | Individual run folder prefix |

Running the In-Browser Diagnostic Sandbox
The repository includes an interactive diagnostic web page (index.html) using onnxruntime-web to execute the exported best.onnx model directly inside your web browser without sending image data to a server.

1. File Placement
Copy your exported best.onnx model file into the same root folder as index.html:
```text
├── index.html
├── best.onnx
```

2. Launch Local HTTP Server
Because browsers restrict local WebAssembly asset loading (fetch requests on file:// URIs), you must serve the folder via a local HTTP server.

Using Python's built-in HTTP server:
```bash
python -m http.server 8000
```
Using Node.js npx http-server:
```bash
npx http-server -p 8000
```

3. Open in Browser
    Navigate to http://localhost:8000 in Google Chrome or Microsoft Edge and perform the following:

    Check Status: Wait for the status HUD to display:

    "Engine Online. Ready for leaf diagnostics."

    Upload Leaf Image: Click Choose File and select any sample wheat leaf image (.jpg or .png).

    View Diagnostic Output: The engine automatically letterboxes the image to 640×640, passes the input tensor through ONNX WebAssembly, applies Non-Maximum Suppression (NMS), and overlays color-coded bounding boxes on the canvas:

    🟢 Healthy: Green
    🔴 Rust: Red
    🟠 Septoria: Amber

Project Structure
```text
├── yolo.py        # Main training and export pipeline
├── .env                       # API keys and environment variables
├── README.md                  # Documentation
└── AeroWheat/
    └── yolo26_wheat_leaf/     # Training outputs and exported ONNX model
        └── weights/
            ├── best.pt        # PyTorch checkpoint
            └── best.onnx      # Client-side deployment model
```