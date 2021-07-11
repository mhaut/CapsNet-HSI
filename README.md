# Capsule Networks for Hyperspectral Image Classification
The Code for "Capsule Networks for Hyperspectral Image Classification". [https://ieeexplore.ieee.org/document/8509610]
```
M. E. Paoletti, J. M. Haut, R. Fernandez-Beltran, J. Plaza, A. Plaza, J. Li and F. Pla
Capsule Networks for Hyperspectral Image Classification
IEEE Transactions on Geoscience and Remote Sensing
DOI: 10.1109/TGRS.2018.2871782
vol. 57, no. 4, pp. 2145-2160, April 2019.
```

![CapsNet](https://github.com/mhaut/CapsNet-HSI/blob/master/images/capsnet.png)



## Example of use
### Download datasets

```
sh retrieveData.sh
```

### Run code

```
Without Validation
python main.py --spatialsize 11 --dataset IP

With Validation
python main.py --use_val --val_percent 0.2 --spatialsize 11 --dataset IP

```
