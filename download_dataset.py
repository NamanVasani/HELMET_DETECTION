import os
from roboflow import Roboflow

api_key = os.environ.get("ROBOFLOW_API_KEY")
if not api_key:
    raise SystemExit("Set the ROBOFLOW_API_KEY environment variable first.")
rf = Roboflow(api_key=api_key)

project = rf.workspace("patelkrishuday2139").project("traffic-helmet-detection")
version = project.version(3)

dataset = version.download("yolov8")

print("Dataset saved at:", dataset.location)