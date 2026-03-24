# Nuremberg Future Land-Cover Prediticion

This project provides a tabular machine learning framework and interactive dashboard for predicting and visualizing urban land-cover composition and temporal change in Nuremberg. Leveraging multi-temporal Sentinel-2 satellite imagery and ESA WorldCover labels , the system extracts domain-informed spatial, spectral, and temporal features mapped to a 20-meter resolution grid. The pipeline evaluates various models, ultimately utilizing a high-performing Stacking Regressor ensemble to classify regions into built-up, vegetation, or water categories and quantify their structural changes over time. To bridge the gap between technical modeling and practical application, the included web interface empowers non-expert stakeholders, such as urban planners, to interactively explore historical ground-truth data, forecast future urban expansion, and assess model confidence across the city.

## Project Report

[Project Report (PDF)](docs/report/report.pdf)

TODO: Actually add the report source, assets, and compiled PDF.

## Dashboard
### Dashboard Demo

![Dashboard Demo](docs/assets/dashboard_demo.gif)


### Running the dashboard

Follow these steps from the repository root directory.

**NOTE:** These instructions assume Python 3 is already installed on your system.

1. Pull Git LFS files before setup.

```bash
git lfs install
git lfs pull
```

**NOTE:** If Git LFS isn't installed find instructions on installing it [here (Stack Overflow)](https://stackoverflow.com/questions/63335778/how-to-install-git-lfs#:~:text=To%20install%20git%20lfs%20on%20a%20Linux,**brew%20install%20git%2Dlfs**%20*%20**git%20lfs%20install**) or [here (Official)](https://git-lfs.com/).

2. Create a virtual environment.

```bash
python -m venv .venv
```

3. Activate the virtual environment.

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

4. Upgrade pip.

```bash
python -m pip install --upgrade pip
```

5. Install the dependencies listed in pyproject.toml into the virtual environment.

```bash
python - <<'PY'
import subprocess
import sys
import tomllib

with open("pyproject.toml", "rb") as f:
	deps = tomllib.load(f)["project"]["dependencies"]

subprocess.check_call([sys.executable, "-m", "pip", "install", *deps])
PY
```

6. Run the dashboard.

```bash
python dashboard/app.py
```
