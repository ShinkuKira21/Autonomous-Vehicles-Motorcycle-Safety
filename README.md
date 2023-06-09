# Autonomous-Vehicles-Motorcycle-Safety
## Authored by Edward Patch
Research project on dangers and safety of AVs with motorcycles. Addresses research gap by examining object classification methods, and identifying potential solutions or issues. 

## Steps ##

**Step 1:**
Create a Virtual Environment and install "requirements.txt".

**Step 2:**
Download the relevant datasets found in workbench/datasets/ and extract the files in raw/

**Step 3:**
Run workbench/notebooks/pre_processing/DS_Regroup notebook to sort the datasets.

**Step 4:**
Run workbench/notebooks/pre_processing/CSV_Creation notebook to create, filter and compile the available datasets.

**Step 5:**
Run workbench/notebooks/pre_processing/CSV_Conversion notebook to convert the preprocessed dataset into YOLO or R-CNN format.

**Step 6 (Optional for YOLO):**
Run workbench/notebooks/pre_processing/CSV_TrainTestSplit notebook to sort the data.

**Step 7:**
After dataset/model/output.yolo is created, copy the contents into your Ultralytics YOLO model and then run the train.py and the detect.py commands. (Remember to tell Ultralytics YOLO to find a selected model weight.)

**Step 8:**
Use Ultralytics YOLOv5, copy the contents of dataset/model/output.yolo to the Ultralytics YOLO directory and begin training/testing. (other ways available too!)

**Development Process**
Not optimised for Darknet anmd F_RCNN has not been focused on, so far.