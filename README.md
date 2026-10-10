# Helmet Detection with YOLOv8

A Python webcam demo that uses a trained YOLOv8 model to find helmets, people without helmets, and number plates in video. It draws boxes and confidence scores on the camera feed. This is a local demonstration, not an enforcement or alerting system.

## Run the webcam demo

The trained checkpoint is included at `runs/detect/helmet_v3/weights/best.pt`. From the project folder, create an environment, install the packages, and start the app:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python hi.py
```

On macOS, allow Terminal to access the camera in **System Settings → Privacy & Security → Camera**, then restart Terminal if the camera was previously denied. Press **Q** or **Esc** in the video window to stop.

## Model and data

The merged training setup uses five classes: helmet, no-helmet, number-plate, rider, and bike. The webcam app currently displays the first three. It uses a confidence threshold of 0.80. A confidence threshold trades off weaker detections against missed objects; it does not guarantee correct detections.

Training datasets and generated run files are excluded from Git because they are large. The saved `best.pt` checkpoint is included so the webcam demo can run after cloning. To train from scratch, `train.py` expects `merged_dataset/data.yaml` and the dataset to be present. `download_merge_retrain.py` downloads the three Roboflow datasets, merges their labels, and starts a 50-epoch training run. Training can take hours and may need a suitable device and substantial disk space.

Roboflow download scripts read their credential from the `ROBOFLOW_API_KEY` environment variable. Set your own key locally before running one; never commit the key.

## Files

- `hi.py` — webcam detector using the saved v3 checkpoint.
- `test_camera.py` — checks camera access and the OpenCV video window without running the model.
- `train.py` — trains from an already prepared merged dataset.
- `download_dataset.py` — downloads one Roboflow dataset.
- `download_merge_retrain.py` — downloads and merges datasets, then trains a model.
- `HELMET_Project_Guide.md` — project explanation and interview preparation.
- `requirements.txt` — Python package dependencies.

## Limitations

The demo does not associate a helmet with a particular rider, read plate characters, identify a vehicle owner, store violations, or send alerts. Detection quality and speed depend on the camera, scene, and hardware. Validate it on representative footage before relying on its output.
