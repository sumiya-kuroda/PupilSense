from pathlib import Path
import cv2
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from pupilsense.io import get_base_dir

class Inference:
    def __init__(self, config_path=None, model_path=None):
        """
        Initializes the Inference class.

        Args:
            config_path (str): Path to the configuration file.
            image_path (str): Path to the directory containing the images.
        """
        print('PupilSense Inference class is initialized!')
        self.config = None

        # self.device = 'cpu'
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        print(f"Using device: {self.device}")

        self.predictor = self._get_predictor(config_path, model_path)
        

    def _get_predictor(self, cfg_path: str, model_path) -> DefaultPredictor:
        if cfg_path is None:
            cfg_path = str(Path(get_base_dir()) / "models" / "Config" / "config.yaml")
        print('Config file: {}'.format(cfg_path))
        # Fetch the config from the given path
        cfg = get_cfg()
        cfg.MODEL.DEVICE = self.device #configuring the device for inference
        if self.device == 'cuda':
            cfg.MODEL.NUM_GPUS = 1 # Hacky way to avoid error. Inference does not support multi-GPUs
            # TODO: Remove MODEL.NUM_GPUS

        cfg.merge_from_file(cfg_path)

        if model_path is None:
            model_path = str(Path(get_base_dir()) / "models" / "model_final.pth")
        print('Model file: {}'.format(model_path))
        cfg.MODEL.WEIGHTS = model_path  # path to the model we just trained

        cfg.MODEL.DEVICE = self.device #configuring the device for inference
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold

        return DefaultPredictor(cfg)
    
    def predict(self, frame):
        return self.predictor(frame)

    def save_frame(self, output, frame, saving_file_location):

        class_names = ["Pupil"]

        v = Visualizer(frame[:, :, ::-1], metadata = {"thing_classes": class_names}, scale=1.0)
        out = v.draw_instance_predictions(output["instances"].to("cpu"))

        # Convert BGR to RGB
        out_rgb = cv2.cvtColor(out.get_image(), cv2.COLOR_BGR2RGB)

        # Saving the image to output directory
        cv2.imwrite(str(saving_file_location), out_rgb)
        

def get_center_and_radius(bbox):
    """
    Calculates the center and radius of a bounding box.

    Args:
        bbox (numpy.ndarray): A bounding box represented as [x1, y1, x2, y2].

    Returns:
        dict: A dictionary containing the center (xCenter, yCenter) and radius of the bounding box.
    """
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    xCenter = (bbox[2] + bbox[0]) / 2
    yCenter = (bbox[3] - height / 2)
    radius = width / 2
    height = height / 2

    return {"xCenter": xCenter, "yCenter": yCenter, "radius": radius, "height": height}