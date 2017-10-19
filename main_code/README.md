usage: 
```
python pstrip.py -i Input_Image [-m Mask_Name] [-o Output_Name] [-p platform] [-g gamma] [-l lambda] [-c correction] [-v] [-h]
```
PCA model for Brain Extraction: 
We provide arguments for user to specify, although the default settings are sufficient for most of the cases.

required arguments:  
  -i , --input         Input image name

optional file arguments:  
  -m, --mask           output mask name. If not specified, the mask is generated into input image folder. 
  -o, --output         brain extracted image. If not specified, image is generated into input image folder.

additional arguments:  
  -p, --platform       platform (CPU/GPU), default GPU  
  -g, --gamma          gamma for total variation term penalty, default 0.5  
  -l, --lambda         lambda for sparse term penalty, default 0.1  
  -c, --correction     number of correction (regularization) steps, default 0  
  -d, --debug {0,1,2}  Debug mode: [Tentative settings] all intermediate results will be saved in tmp_res folder. Be careful, this could occupy large space on disk if multiply images will be processed  
  -v, --version        show program's version number and exit  
  -h, --help           show this help message and exit  
