<img src='fig/sp.gif' align="right" width=225>
<br><br><br>

## Neural Implicit 3D Shapes from Single Images with Spatial Patterns

This repository provides PyTorch implementation of our paper:

[Neural Implicit 3D Shapes from Single Images with Spatial Patterns](https://arxiv.org/pdf/2106.03087.pdf)

<img src="./fig/result.png" width="700" />


### Installation
- Clone this repo:
```bash
git clone https://github.com/yixin26/SVR-SP.git
cd SVR-SP & cd code
```

#### Prerequisites
- Python 3.6
- CPU or NVIDIA GPU + CUDA CuDNN
- Pytorch > (1.4.x)

#### Install dependencies
Install via conda environment `conda env create -f environment.yml` (creates an environment called svrsp)


### Run

For a quick demo, please use the pre-trained model. Please download the model from [Google Drive](https://drive.google.com/file/d/1gLNrlg0NLG6VndslWMTRZqU6ZqV9P-ax/view?usp=sharing),
and exact the model to ```code/test/model```.
For generating all the testing samples from a category of ShapeNet Core Dataset, e.g., Chair, please use

```bash
python sdf2obj.py --category chair --ckpt 30 --batch_size 4 -g 0,1
```
The generated mesh files will be stored at  ```./test/results/30/test_objs/...```. 

We use the preprocessed dataset from [DISN](https://github.com/laughtervv/DISN), including SDF files, ShapeNetRendering files (with image and camera), and mesh obj files.
As to create meshes from the generated SDFs, we use the executable file from [DISN/isosurface](https://github.com/laughtervv/DISN/tree/master/isosurface).


