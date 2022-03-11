# Landmark-based segmentation using multi-atlas
This repository contains the source code of the multi-atlas baseline method used in the paper "Hybrid graph convolutional neural networks for landmark-based anatomical segmentation" (Gaggion et al., MICCAI 2021). For more information about this work, visit Nicolás Gaggion's repository on [GitHub](https://github.com/ngaggion/HybridGNet).

## Intructions
This projects uses Python 3.8.10.

### Data
- Download and preprocess the JSRT dataset. You can found the instructions [here](https://github.com/ngaggion/HybridGNet).
- Format landmarks files into txt files for SimpleElastix. Run `./00_preprocess_jsrt.sh` after installing the project environment (instructions below).

### Project environment:
1. Create and activate virtual environment: 1) `python3 -m venv env` 2) `source env/bin/activate`
2. Install required packages: `pip install -r requirements.txt`
3. Install project modules (src): `pip install -e .`
4. Install SimpleElastix toolbox following [this guide](https://gist.github.com/vfmatzkin/0fcc79a61f9bafcc2113fd83a8900937).

### Simulations:
- JSRT: `./01_test_jsrt.sh`
- Montgomery: `./02_test_montgomery.sh`
- Shenzhen: `./03_test_shenzhen.sh`

## Reference
- Gaggion, N., Mansilla, L., Milone, D. H., & Ferrante, E. (2021, September). Hybrid graph convolutional neural networks for landmark-based anatomical segmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 600-610). Springer, Cham.

## License
[MIT](https://choosealicense.com/licenses/mit/)
