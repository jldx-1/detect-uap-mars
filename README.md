# detect-uap-mars
A fully synthetic data-generation pipeline for training UAP (Unidentified Anomalous Phenomena) classification model on Martian imagery.

**Contributers**
Jean-Luc DeRieux and Allen Chandler

**Key Features:**
- **NASA Mastcam Integration**  
  Automatically downloads raw Perseverance Mastcam images from the official NASA archive.
- **Random ROI Sampling**  
  Selects random Regions of Interest (ROIs) within each image for anomaly insertion.
- **Semantic Inpainting**  
  Uses a Stable Diffusion–based inpainting model to place UAP targets based on target descriptions.
- **Adaptive Color Blending**  
  Computes and applies an average color tint to each inserted target for seamless integration.
- **Support for YOLO Format**  
  Generates corresponding YOLO-style bounding box annotations and label files for each synthetic anomaly. 

**Sources**
NASA. “Sol 2692: Mast Camera (Mastcam) – Raw Image from Mars Perseverance Rover.” Feb. 26, 2024. Available: https://mars.nasa.gov/raw_images/787528/  

All-domain Anomaly Resolution Office (AARO). “Official UAP Imagery.” Available: https://www.aaro.mil/UAP-Cases/Official-UAP-Imagery/  

Skywatcher Team. “Part II: UAP Classification Overview.” YouTube, Apr. 22, 2025. Available: https://www.youtube.com/watch?v=2VN3omlVqxk  

Ultralytics. “YOLOv5.” GitHub repository. Available: https://github.com/ultralytics/YOLOv5  

Stability AI. “stabilityai/stable-diffusion-2-inpainting.” GitHub repository. Available: https://github.com/Stability-AI/stable-diffusion-2-inpainting  

Allen Chandler. “uap-detection.” GitHub repository. Available: https://github.com/AllenChandler/uap-detection  

Jean-Luc DeRieux. “detect-uap-mars.” GitHub repository. Available: https://github.com/Jean-LucDeRieux/detect-uap-mars  

Middle East Object. “This clip was taken by an MQ-9 in the Middle East.” 2024. Available: https://www.aaro.mil/UAP-Cases/Official-UAP-Imagery/  

NASA. “NASA Open APIs.” Available: https://api.nasa.gov/  

### Stack
- Python

### Python Virtual Env. commands
- Downloading uv: `pip install uv`
- Creates (or updates) python uv virtual env: `uv sync`
- Open up a Vir. Env. uv shell: `uv shell`
- Downloading a library: `uv pip install PackageName`
- Run a server for jupyter notebook: `uv run jupyter notebook`