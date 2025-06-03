# PupilSense
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

## 1. Preprocessing
This folder contains scripts to compress the video from AVI to MP4 as well as to extract video frames for training the model. `cd` to this folder and run `snakemake --profile swc-hpc/`. Snakefile requires you to have three file/dirs beorehand:
- `INPUT_LIST`: a text file where you save the path to your raw eye video
- `DEST_CEPH`: a dir where you save frames for model training
- `LOG_DIR` a dir to save the log files

## 2. Training
To fine-tune the Detectron2 model on your dataset, `cd` to this folder and run `snakemake --profile swc-hpc/`.

PupilSense's defult model is available from [here](https://github.com/stevenshci/PupilSense/releases/download/v1.0/models.zip). Download and unzip it into `models/`. This gives an error message `Skip loading parameter 'roi_heads.box_predictor.cls_score.weight' to the model due to incompatible shapes: (3, 1024) in the checkpoint but (2, 1024) in the model! You might want to double check if this is expected.`.


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
       

### Annotations
To annotate the dataset, we recommend using [MakeSense.ai](https://www.makesense.ai/). Export these annotations in the COCO format, which should include necessary details for images, annotations, and categories (pupil and iris). Copy/paste them into test_data.json and train_data.json as shown above.

## 3. Inference
This folder contains scripts to predict pupil size using the trained model saved in `models/`. `cd` to this folder and run `snakemake --profile swc-hpc/`.


## Acknowledgement
We thank Dammy Onih for the support to set up this pipeline, and Ryan Shen for training the model.

The original Pupil Sense paper is below:
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

## Troubleshooting
- Here is a tip.