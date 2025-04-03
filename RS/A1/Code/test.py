import os

scripts = ["2train_pos_per_threshold.py", "train_pos_per_extra.py"]

for script in scripts:
    print(f"Running {script}...")
    os.system(f"python3 {script}")  # 如果 Python 3，改成 `python3 {script}`
    print(f"Finished {script}.\n")