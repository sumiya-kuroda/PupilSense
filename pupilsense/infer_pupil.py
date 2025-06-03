from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import defopt
from pupilsense.inference.inference_pupilsense import Inference, get_center_and_radius

def extract_pupil(input_file_location, *, invert=False):
    """Extract pupils

    :param str input_file_location: file path to eye video.
    :param bool invert: whether you want to invert image
    """
    print('---Running pupil detection on eye video---')

    input_file_location = Path(input_file_location)

    saving_file_location = Path(input_file_location).parent / 'pupil_extracted'
    saving_file_location.mkdir(parents=False, exist_ok=True)
    _saving_file_location_predicted = saving_file_location / 'predicted'
    _saving_file_location_predicted.mkdir(parents=False, exist_ok=True)

    PSInference = Inference(config_path=None, model_path=None)
    
    # load eye video
    print('Loading eye video')
    eye_video = cv2.VideoCapture()
    eye_video.open(str(input_file_location))

    i_frame = 0
    num_total_frame = int(eye_video.get(cv2.CAP_PROP_FRAME_COUNT))
    ellipse_output = []
    for i_frame in range(1000):
    # while i_frame < num_total_frame: # while i_frame < num_total_frame:
        eye_video.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
        _, frame = eye_video.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # run prediction on video frame
        output = PSInference.predict(frame)
        instances = output["instances"]
        boxes = instances.pred_boxes.tensor.cpu().numpy()

        if len(boxes) <= 0:
            ellipse_output.append([np.nan, np.nan, np.nan,np.nan, np.nan, np.nan])
            continue

        if i_frame % 100 == 0:
            cv2.imwrite(str(saving_file_location / f'{i_frame}.jpg'), frame)
            PSInference.save_frame(output, 
                                    frame, 
                                    saving_file_location = saving_file_location / 'predicted' / f'predicted_{i_frame}.jpg')
        
        # classes = instances.pred_classes
        # scores = instances.scores

        # instances_with_scores = [(i, score) for i, score in enumerate(scores)]
        # instances_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # for index, score in instances_with_scores:
        #     if classes[index] == 0:  # 0 is Pupil
        #         pupil = boxes[index]
        #         pupil_info = get_center_and_radius(pupil)
        #         radius = int(pupil_info["radius"])
        #         height = int(pupil_info["height"])
        #         xc = int(pupil_info["xCenter"])
        #         yc = int(pupil_info["yCenter"])
        #         ellipse_output.append([i_frame, radius, height, xc, yc, float(score)])
        #         break  # Only one prediction per frame

        score = 0
        radius = 0
        print(f"Processed Frame {i_frame}: radius = {radius}, score = {score}")
        # i_frame += 1

    # write dict as csv using pd.dataframe
    pupil_est_df = pd.DataFrame(np.array(ellipse_output), columns=['frame_num','radius','height','xc','yc','score'])
    pupil_est_df.to_pickle(saving_file_location / 'eye.p')
    # print(f'total time taken: {round((time.time()-time_pre)/60,2)} mins')

if __name__ == "__main__":
    defopt.run(extract_pupil)