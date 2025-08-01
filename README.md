# SAR Geometry

A Python package for Synthetic Aperture Radar (SAR) geometry calculations.

## Installation

1. **Pre-requisites**: Make sure you have [Conda](https://docs.anaconda.com/miniconda/) installed on your system. Then, create a new environment and activate it:

```bash
conda create -n sargeom-env python=3.12
conda activate sargeom-env
```

2. **Install the package**: Next, you can install the plugin directly from GitHub using [pip](https://pypi.org/project/pip/):

```bash
pip install git+https://github.com/oleveque/sargeom.git
```

or in development mode:

```bash
git clone https://github.com/oleveque/sargeom.git
pip install -r ./sargeom/requirements.txt
pip install -e ./sargeom
```

If you want to add the package to your dependencies file, you can add the following line to your `requirements.txt` file:

```bash
sargeom @ git+https://github.com/oleveque/sargeom@main
```

## Dependency relationships

```mermaid
graph TD
    ndarray --> Cartographic
    ndarray --> Cartesian3
    Cartesian3 --> CartesianECEF
    Cartesian3 --> CartesianLocalENU
    Cartesian3 --> CartesianLocalNED
```

## Issues

If you encounter any problems, please [file an issue](https://github.com/oleveque/sargeom/issues) with a detailed description.
Your feedback is valuable in improving the package.
