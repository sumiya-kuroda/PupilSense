from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import time
import defopt
from pupilsense.inference.inference_pupilsense import Inference, get_center_and_radius
from pupilsense.multithread import CapMultiThreading
import time
from datetime import datetime

def extract_pupil(input_file_location, *, invert=False):
    """Extract pupils

    :param str input_file_location: file path to eye video.
    :param bool invert: whether you want to invert image
    """
    print('---Running pupil detection on eye video---')

    input_file_location = Path(input_file_location)

    saving_file_location = Path(input_file_location).parent / 'pupil_prediction'
    saving_file_location.mkdir(parents=False, exist_ok=True)
    _saving_file_location_predicted = saving_file_location / 'predicted'
    _saving_file_location_predicted.mkdir(parents=False, exist_ok=True)

    PSInference = Inference(config_path=None, model_path=None)
    
    # load eye video
    print('Loading eye video')
    # single CPU
    # eye_video = cv2.VideoCapture()
    # eye_video.open(str(input_file_location))
    eye_video = CapMultiThreading(str(input_file_location))
    start_t = time.time()
    time_now = datetime.fromtimestamp(start_t).strftime('%Y-%m-%d %H:%M:%S')
    print(f'Starting at: {time_now}')

    i_frame = 0
    num_total_frame = int(eye_video.get_totalframe())
    print(f'Total number of frames: {num_total_frame}')
    ellipse_output = []
    for i_frame in range(num_total_frame):
        # single CPU
        # eye_video.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
        # _, frame = eye_video.read()
        ret, frame = eye_video.get_frame()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # run prediction on video frame
        output = PSInference.predict(frame)
        instances = output["instances"]
        boxes = instances.pred_boxes.tensor.cpu().numpy()

        if len(boxes) <= 0:
            ellipse_output.append([np.nan, np.nan, np.nan,np.nan, np.nan, np.nan])
            print(f"Processed Frame {i_frame}: No pupil detected")
            continue

        if i_frame % 1000 == 0:
            cv2.imwrite(str(saving_file_location / f'{i_frame}.jpg'), frame)
            PSInference.save_frame(output, 
                                    frame, 
                                    saving_file_location = saving_file_location / 'predicted' / f'predicted_{i_frame}.jpg')
        
        classes = instances.pred_classes
        scores = instances.scores
        instances_with_scores = [(i, score) for i, score in enumerate(scores)]
        instances_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        for index, score in instances_with_scores:
            if float(classes[index]) == 0:  # 0 is Pupil
                pupil = boxes[index]
                pupil_info = get_center_and_radius(pupil)
                radius = float(pupil_info["radius"])
                height = float(pupil_info["height"])
                xc = float(pupil_info["xCenter"])
                yc = float(pupil_info["yCenter"])
                ellipse_output.append([i_frame, radius, height, xc, yc, float(score)])
                break  # Only one prediction per frame

        print(f"Processed Frame {i_frame}: radius = {float(radius)}, score = {float(score)}")

    # write dict as csv using pd.dataframe
    pupil_est_df = pd.DataFrame(np.array(ellipse_output), columns=['frame_num','radius','height','xc','yc','score'])
    pupil_est_df.to_pickle(saving_file_location / 'pupilsense_output.p')
    print(f'total time taken: {round((time.time()-start_t),2)} sec')
    print(f'average FPS: {round(num_total_frame/(time.time()-start_t), 2)}')


if __name__ == "__main__":
    defopt.run(extract_pupil)