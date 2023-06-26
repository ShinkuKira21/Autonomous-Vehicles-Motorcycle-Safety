import sys
import os

current_dir = os.getcwd()
workbench_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.insert(0, workbench_dir)
