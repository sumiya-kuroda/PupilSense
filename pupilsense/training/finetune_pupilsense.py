from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os
import torch
from pathlib import Path
from detectron2.data import MetadataCatalog
from detectron2.config.config import CfgNode
from detectron2.data.datasets import load_coco_json, register_coco_instances

# Check if CUDA is available
cuda_available = torch.cuda.is_available()

if cuda_available:
    device = 'cuda'
    ngpus_per_node = torch.cuda.device_count()
else:
    device = 'cpu'
print(f"Using device: {device}")

class Finetune:
    def __init__(self):
        """
        Initializes the Finetune class.
        """
        self._cfg = None
        self._trainer = None

    def set_config_train(self, model_path: str, train_json_path=None, train_data_path=None):
        """
        Sets the configuration for training and creates a DefaultTrainer instance.

        Args:
            model_path (str): Path to the directory where the new model will be saved.
            train_metadata (dict, optional): Metadata for the training dataset.
            train_dataset_dicts (list, optional): List of dictionaries containing training data.
        """
        self._cfg = get_cfg()
        
        self._cfg.OUTPUT_DIR = model_path
        self._cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        # self._cfg.merge_from_file(r"D:\Uni\Master\ANAT0021Dissertation\data\PupilSense\detectron2\configs\COCO-InstanceSegmentation\mask_rcnn_R_50_FPN_3x.yaml")
        
        if train_json_path is None:
            raise NotImplementedError
        else:
            register_coco_instances("pupil_train_data", {}, train_json_path, train_data_path)
        self._cfg.DATASETS.TRAIN = ("pupil_train_data",)
        self._cfg.DATASETS.TEST = () # https://github.com/facebookresearch/detectron2/issues/2012

        self._cfg.DATALOADER.NUM_WORKERS = 1
        best_model_path = Path(model_path).parent / 'models_best'
        self._cfg.MODEL.WEIGHTS = os.path.join(str(best_model_path), "model_final.pth") # FINETUNE the updated model
        # self._cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
        self._cfg.SOLVER.IMS_PER_BATCH = 4  # This is the real "batch size" commonly known to deep learning people
        self._cfg.SOLVER.BASE_LR = 0.0001  # Pick a good LR
        self._cfg.SOLVER.MAX_ITER = 2500 

        self._cfg.SOLVER.STEPS = (2000, 3000, 4000) # Add this line to specify the steps at which to decrease the learning rate
        self._cfg.SOLVER.GAMMA = 0.1  # Set the factor by which to decrease the learning rate

        # Add this line to specify the learning rate scheduler
        self._cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"

        self._cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  # Default is 512, using 256 for this dataset.
        self._cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 

        self._cfg.MODEL.DEVICE = device 
        if cuda_available:
            self._cfg.MODEL.NUM_GPUS = ngpus_per_node

        os.makedirs(self._cfg.OUTPUT_DIR, exist_ok=True)

        self._trainer = DefaultTrainer(self._cfg)  # Create an instance of DefaultTrainer with the given configuration
        self._trainer.resume_or_load(resume=False)  # Start training from scratch
    

    def save_cfg(self, cfg_path: str):
        if os.path.exists(cfg_path):
            raise FileExistsError('Are you trying to overwrite existing cofig file?')
        else:
            print("Creating cofig file")
            os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
            with open(cfg_path, "w") as f:
                f.write(self._cfg.dump())

    def train(self):
        """
        Starts the training process.
        """
        self._trainer.train()