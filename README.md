<img width="1750" alt="PupilSense: Detection of Depressive Episodes Through Pupillary Response in the Wild" src="https://github.com/stevenshci/PupilSense/blob/main/static/header.png">

This is the pipeline to use **PupilSense** from [PupilSense: Detection of Depressive Episodes Through Pupillary Response in the Wild](https://arxiv.org/abs/2404.14590) on SWC HPC. **PupilSense** is a deep learning-based pupillometry system. It uses eye images collected from smartphones for research in the behavior modeling domain.

<img width="1750" alt="Pupil-to-Iris Ratio (PIR) Estimation Pipeline" src="https://github.com/stevenshci/PupilSense/blob/main/static/PupilSense.png">


## Installation

To set up the project, clone this repo and navigate to the project directory by `cd PupilSense`. Then create conda environemnt and install the required packages: 
```sh
module load cuda/12.4
module load miniconda

conda env create -f environments.yml 

conda activate pupilsense
git clone https://github.com/facebookresearch/detectron2.git
pip install -e detectron2
pip install -e .
conda deactivate
```

### Setup

snakemake --profile swc-hpc/

### Dataset

This project uses a custom dataset of eye images for training and evaluation. The dataset should be organized in the following structure:

    dataset/
    ├── train/
    │   │── image1.png
    │   │── image2.png
    │   └── ...
    |
    │── train_data.json
    │    
    └── test/
    │   ├── image1.png
    │   ├── image2.png
    │   └── ...
    |
    └── test_data.json
       
Note: The annotations(test_data.json, train_data.json) should be in COCO format, with the pupil and iris regions labeled as separate categories.

### Annotations

To annotate the dataset, use tools like MakeSense.ai, Roboflow, Labelbox, LabelImg, or VIA to label pupil and iris regions on the images. Export these annotations in the COCO format, which should include necessary details for images, annotations, and categories.

The COCO format is a standard for object detection/segmentation tasks and is widely supported by many libraries and tools in the computer vision community.

### Inference

To test your images on a batch of images on the trained model:

    python scripts/inference.py



## Training
To fine-tune the Detectron2 model on your dataset, run the following command:

    python scripts/finetune.py

Click [Pretrained Models](https://github.com/stevenshci/PupilSense/releases/download/v1.0/models.zip) to download our pre-trained model for PupilSense, and unzip it into `models`.


## Reference
```
@INPROCEEDINGS{10652166,
  author={Islam, Rahul and Bae, Sang Won},
  booktitle={2024 International Conference on Activity and Behavior Computing (ABC)}, 
  title={PupilSense: Detection of Depressive Episodes through Pupillary Response in the Wild}, 
  year={2024},
  volume={},
  number={},
  pages={01-13},
  keywords={Laboratories;Mental health;Depression;Real-time systems;Wearable devices;Monitoring;Smart phones;Pupillometry;Depression;Affective computing;Machine Learning},
  doi={10.1109/ABC61795.2024.10652166}}
```