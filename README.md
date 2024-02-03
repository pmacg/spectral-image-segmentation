# Image Segmentation with Spectral Clustering

This repository provides a simple python script for image segmentation with spectral clustering.

## Setup
Install the project from this Git repository with

```commandline
python -m pip install git+https://github.com/pmacg/spectral-image-segmentation.git
```

## Usage
In order to segment a given image, simply execute

```commandline
segment <inputFilename> <outputFilename> <numSegments>
```

## Reference

If you find this useful in your work, please cite the following paper.

**A Tighter Analysis of Spectral Clustering, and Beyond**, Peter Macgregor and He Sun, ICML 2022.

```bibtex
@InProceedings{pmlr-v162-macgregor22a,
  title = 	 {A Tighter Analysis of Spectral Clustering, and Beyond},
  author =       {Macgregor, Peter and Sun, He},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {14717--14742},
  year = 	 {2022},
  volume = 	 {162},
  publisher =    {PMLR},
}
```
