# AutoFlow Pipeline

This repository contains the implementation of AutoFlow, a medical imaging pipeline for automated flow analysis in cardiovascular imaging. The pipeline processes DICOM images through a series of steps to extract and analyze flow measurements.

## Project Status

- **Version**: 3.0.0
- **Status**: Paper resubmitted to *Radiology: Cardiothoracic Imaging*
- **Author**: Akhilesh Yeluru

## Pipeline Overview

The AutoFlow pipeline consists of several key stages:

1. **Data Preparation**
   - Unzipping and organizing DICOM files
   - Parsing DICOM metadata
   - Converting DICOM to NIFTI format
   - Patient data cataloging

2. **Localization Network (LocNet)**
   - Preprocessing for LocNet
   - Running LocNet inference
   - Reversing preprocessing
   - Extracting relevant slices

3. **Segmentation Network (SegNet)**
   - Preparing data for SegNet
   - Running SegNet inference
   - Reversing preprocessing

4. **Flow Analysis**
   - Computing flow measurements
   - Cataloging flow data
   - Measuring distances between planes

## Installation

### Using pip (Development Mode)
```bash
git clone [repository-url]
cd auto-flow_pipeline
pip install -e .
```

### Dependencies
All dependencies are specified in `environment.yml`. It is recommended to create a new conda environment using:
```bash
conda env create -f environment.yml
```

## Usage

The pipeline provides several command-line tools for each stage of processing:

### Data Preparation
```bash
unzip                    # Extract DICOM files from archives
prune                    # Clean up unnecessary files
parse_all_dicoms         # Parse DICOM metadata
generate_base_niftis     # Convert DICOM to NIFTI
catalogue_patients       # Create patient catalog
```

### Localization Network
```bash
prepare_for_locnet       # Prepare data for LocNet
run_locnet              # Run LocNet inference
reverse_locnet_preprocessing  # Reverse preprocessing
extract_from_locnet     # Extract relevant slices
```

### Segmentation Network
```bash
prepare_for_segnet      # Prepare data for SegNet
run_segnet             # Run SegNet inference
reverse_segnet_preprocessing  # Reverse preprocessing
```

### Flow Analysis
```bash
compute_flow            # Calculate flow measurements
catalogue_flow          # Catalog flow data
measure_distances_between_planes  # Measure plane distances
```

## Project Structure

```
src/auto_flow_pipeline/
├── drivers/            # Command-line interface implementations
├── data_io/            # Data input/output handling
├── preprocessing/      # Data preprocessing utilities
├── inference/          # Model inference code
├── postprocessing/     # Post-processing utilities
├── flow_calculation/   # Flow calculation algorithms
├── slice_extraction/   # Slice extraction utilities
└── visualization/      # Visualization tools
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

Yeluru ARR, Tycko A, Cazares N Jr, Masutani EM, Sankaran RR, Rushworth PM, Hall KM, Sung L, Hsiao A. Deep Learning Automated Measurement of Shunt Severity with Estimation of Uncertainty in 4D Flow MRI. Radiol Cardiothorac Imaging. 2026 Feb;8(1):e250138. doi: 10.1148/ryct.250138. PMID: 41711549; PMCID: PMC12949416.
