import os
from roboflow import Roboflow
from ultralytics import YOLO
from dotenv import load_dotenv

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("ROBOFLOW_API_KEY not found in .env file.")

    # 1. Download custom dataset from Roboflow
    rf = Roboflow(api_key=api_key) 
    project = rf.workspace("wheatleaf").project("wheat-leaf-disease-detection")
    dataset = project.version(3).download("yolov8")

    data_yaml_path = os.path.join(dataset.location, "data.yaml")

    # 2. Train YOLO26 on your local Blackwell GPU
    model = YOLO("yolo26s.pt")

    model.train(
        data=data_yaml_path,
        epochs=50,                  
        imgsz=640,
        device=0,                   
        batch=8,                    
        workers=2,                  
        project="AeroWheat",
        name="yolo26_wheat_leaf"
    )

    # 3. Dynamic Path Resolution (Fixes FileNotFoundError)
    # This reads the exact auto-incremented directory path (e.g., .../yolo26_wheat_leaf-6)
    completed_run_dir = model.trainer.save_dir
    best_weights_path = os.path.join(completed_run_dir, "weights", "best.pt")
    
    print(f"\nResolving training checkpoints dynamically from: {best_weights_path}")

    # 4. Load the newly trained custom weights
    custom_model = YOLO(best_weights_path)

    # 5. Export to ONNX format for client-side Angular execution
    # 'half=False' optimizes perfectly for standard CPU compatibility.
    # Turn on 'half=True' if your client application leverages WebGPU.
    print("Initiating ONNX export layer optimizations...")
    onnx_path = custom_model.export(format="onnx", imgsz=640, quantize=False)

    print("\n=========================================================================")
    print(f"Success! Web-ready production model is optimized and saved at:\n{onnx_path}")
    print("=========================================================================")

if __name__ == '__main__':
    main()