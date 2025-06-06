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
To fine-tune the Detectron2 model on your dataset, `cd` to this folder and run `sbatch sbatch_train_model.sh`. This requires you to have/organize a custom dataset of eye images for training in the following structure:

    dataset/
    ├── train/
    │   │── image1.jpg
    │   │── image2.jpg
    │   └── ...
    |
    └── train_data.json

This finetunes the best model you already have and it expects the following file strucure. Ask Sumiya to give you the model and save it under `models_best/`

    ├── dataset/
    │   └── ...
    ├── models/
    └── models_best/
        ├── Config/
        ├── metrics.json
        └── model_final.pth

To annotate the dataset, we recommend using [MakeSense.ai](https://www.makesense.ai/). Upload images and you are asked to create a label `pupil`. Then use polygons to annotate pupils. 

<img width="1750" alt="MakeSenseAI screenshot" src="https://github.com/sumiya-kuroda/PupilSense/blob/main/doc/gallery/makesense_2.PNG">

Export these annotations in the COCO format, which should include necessary details for images, annotations, and categories (`pupil` only for now i.e., `MODEL.ROI_HEADS.NUM_CLASSES = 1`). Copy/paste them into train_data.json as shown above.

<img width="1750" alt="MakeSenseAI screenshot" src="https://github.com/sumiya-kuroda/PupilSense/blob/main/doc/gallery/makesense_1.PNG">

## 3. Inference
This folder contains scripts to predict pupil size using the trained model saved in `models/`. `cd` to this folder and run `snakemake --profile swc-hpc/`. Snakefile requires you to have three file/dirs beorehand:
- `INPUT_LIST`: a text file where you save the path to your mp4 eye video


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
- detectron2 requires you to apply `frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)`.
- How can I speed up the inference step?: Since detectron2 natively supports only one gpu for inference (or [implement this?](https://github.com/facebookresearch/detectron2/issues/1770)), the bottle neck is likely to be video frame I/O. Single-processed OpenCV is known to be very slow ([ref](https://github.com/vujadeyoon/Fast-Video-Processing/tree/master)). This means you want to use Python's multithreading function whenever running I/O with OpenCV. Also make sure to increase `--cpus-per-gpu` of SLURM's setting.
- How can I speed up the training step?: https://github.com/facebookresearch/detectron2/issues/5314