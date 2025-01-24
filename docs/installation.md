


# Installation
## :rocket: Environment

### Get started
We test the installation with
* GPU NVIDIA GeForce RTX 3090
* Ubuntu 20.04 LTS
* CUDA == 11.3, V11.3.58

```bash
$ conda env create -f environment.yml
$ conda activate cpf
```

### Install dependencies
```bash
# install open3d
# login into GPU node first
cd ~/open3d-whl
pip install open3d-0.18.0-cp310-cp310-manylinux_2_27_x86_64.whl open3d
```

```bash
# install opendr
mkdir ./whl_files
cd ./whl_files
pip download opendr
conda install menpo::osmesa
# login into GPU node
cd ~/projects/CPF/whl_files
pip install *.whl
pip install chumpy-0.70.tar.gz
pip install opendr-0.78.tar.gz
```
```bash
pip install -r requirements.txt
pip install -r requirements@git.txt
cd thirdparty
git clone https://github.com/lixiny/manotorch.git
# login into GPU node
cd manotorch
pip install .
```
```bash
# pytorch3d
pip install pytorch3d --no-index --no-cache-dir --find-links https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu113_pyt1110/download.html
```


### Install thirdparty

#### dex-ycb-toolkit
```bash
cd thirdparty

# create a __init__.py in dex_ycb_toolkit
$ touch ./dex-ycb-toolkit/dex_ycb_toolkit/__init__.py
$ pip install ./dex-ycb-toolkit

# check install success:
$ python -c "from dex_ycb_toolkit.dex_ycb import DexYCBDataset"
```
#### libmesh
```bash
$ cd thirdparty/libmesh

$ python setup.py build_ext --inplace

# check install success: (in the project root directory)
$ python -c "import thirdparty.libmesh"
```

&nbsp;
## :luggage: Assets
#### MANO hand model
Get the MANO hand model `mano_v1_2.zip` from the [MANO website](https://mano.is.tue.mpg.de).  
1. click **`Download`** on the top menu, this requires register & login.
2. on the Download page, navigate to **Models & Code** section, and click `Models & Code`, the `mano_v1_2.zip` will be downloaded automatically.
3. Unzip `mano_v1_2.zip` and copy it into an `assets` folder.

#### handobjectconsist assets
Download the contents of the [handobjectconsist/assets](https://github.com/hassony2/handobjectconsist/tree/master/assets), unzip, and put them in the `assets` folder.

the `assets` folder should have the following structure:
```shell
assets
├── anchor
│   ├── anchor_mapping_path.pkl
│   ├── anchor_weight.txt
│   ├── face_vertex_idx.txt
│   └── merged_vertex_assignment.txt
├── closed_hand
│   └── hand_mesh_close.obj
├── fhbhands_fits
│   ├── Subject_1
│   ├── ...
│   └── Subject_6
├── hand_palm_full.txt
├── mano
│   ├── fhb_skel_centeridx0.pkl
│   └── fhb_skel_centeridx9.pkl
├── mano_v1_2
│   ├── __init__.py
│   ├── LICENSE.txt
│   ├── models
│   └── webuser
```


&nbsp;
## :floppy_disk: Data Preparation

#### First-Person Hand Action Benchmark (fphab)
1. Download the fphab following the [official instructions](https://github.com/guiggh/hand_pose_action), and link it to `data/fhbhands`.  
2. Resize the original full-res images based on the [handobjectconsist/reduce_fphab.py](https://github.com/hassony2/handobjectconsist/blob/master/reduce_fphab.py).

3. Download our fhbhand supplimentary file [fhbhands_supp.tar.gz](https://huggingface.co/lixiny/CPF/resolve/main/data/fhbhands_supp.tar.gz), and link it to `data/fhbhands_supp`

The fphab-related data in the project should has the following structure:
```shell
data
├── fhbhands
│   ├── action_object_info.txt
│   ├── action_sequences_normalized
│   ├── change_log.txt
│   ├── data_split_action_recognition.txt
│   ├── file_system.jpg
│   ├── Hand_pose_annotation_v1
│   ├── Object_6D_pose_annotation_v1_1
│   ├── Object_models
│   ├── Subjects_info
│   └── Video_files_480
├── fhbhands_supp 
│   ├── Object_contact_region_annotation_v512
│   ├── Object_models
│   └── Object_models_binvox
```

#### HO3Dv2 
1. Download the [HO3D](https://www.tugraz.at/index.php?id=40231) dataset (version 2) following the [official instructions](https://github.com/shreyashampali/ho3d?) and link it to `data/HO3D`.
2. Download our YCB models supplimentary [YCB_models_supp_v2.tar.gz](https://huggingface.co/lixiny/CPF/resolve/main/data/YCB_models_supp_v2.tar.gz) and link it to `data/YCB_models_supp_v2`.
3. Download our synthetic data: [HO3D_ycba.tar.gz](https://huggingface.co/lixiny/CPF/resolve/main/data/HO3D_ycba.tar.gz) and [HO3D_syntht.tar.gz](https://huggingface.co/lixiny/CPF/resolve/main/data/HO3D_syntht.tar.gz), link them to `data/HO3D_ycba` and `data/HO3D_syntht`.

The HO3D-related data in the project should has the following structure:
```shell
data
├── HO3D
│   ├── evaluation
│   ├── evaluation.txt
│   ├── train
│   └── train.txt
├── HO3D_supp_v2
│   ├── evaluation
│   └── train
├── HO3D_syntht
│   └── train
├── HO3D_ycba
│   ├── 003_cracker_box
│   ├── 004_sugar_box
│   ├── 006_mustard_bottle
│   ├── 010_potted_meat_can
│   ├── 011_banana
│   ├── 019_pitcher_base
│   ├── 021_bleach_cleanser
│   ├── 025_mug
│   ├── 035_power_drill
│   └── 037_scissors
├── YCB_models_supp_v2
│   ├── 002_master_chef_can
│   ├── 003_cracker_box
│   └── ...
```

#### DexYCB
We use the DexYCB to aid in training the contact recovery model.  
1. Download [DexYCB](https://arxiv.org/abs/2104.04631) dataset from the [official site](https://dex-ycb.github.io), unzip and link the dataset to `data/DexYCB`. 
2. Download our DexYCB supplimentary file [DexYCB_supp.tar.gz](https://huggingface.co/lixiny/CPF/resolve/main/data/DexYCB_supp.tar.gz) and link it to `data/DexYCB_supp`.

The DexYCB-related data in the project should has the following structure:
```shell
data
├── DexYCB
│   ├── 20200709-subject-01
│   ├── ...
│   ├── 20201022-subject-10
│   ├── bop
│   ├── calibration
│   └── models
├── DexYCB_supp
│   ├── 20200709-subject-01
│   ├── ...
│   └── 20201022-subject-10
```


&nbsp;
## :snowflake: Saved Models 
Download our model ckeckpoints at [here](https://huggingface.co/lixiny/CPF/blob/main/checkpoints.tar) and put the contents in the `checkpoints` folder.   

The `legacy` folder contains the checkpoints for the hand-object pose estimation network (HoNet) and the contact recovery network (PiCR) in our ICCV 2021 conference paper.

The `verified` folder contains the checkpoints for the contact recovery network (PiCR) that is re-trained within this repository (the checkpoints used in: 
* `config/HONetPiCRPipeline_fphab.yml` 
* `config/HOPose_PiCRPipeline_ho3dv2.yml`





