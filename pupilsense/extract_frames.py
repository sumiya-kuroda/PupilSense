import time
import numpy as np
import cv2
from pathlib import Path 
import math
import defopt

def main(input_file_location, *, num_training_img = 20, ext="jpg", log=None, output=None):
    """Extract frames from video file

    :param str input_file_location: file path to eye video.
    :param int num_training_img: Number of images generated.
    :param str ext: "png" or "jpg" etc. The ones supported by Open cv.
    :param str log: Path to save log file.
    :param str output: Path to save the extracted frames.

    """
    print('---Extracting frames from video file---')

    # ============================================================================================
    # input path -- path to the video file
    input_file_location = Path(input_file_location)

    # output path -- path to save the image files
    saving_file_location = input_file_location.parent / 'pupil_frames'
    saving_file_location.mkdir(parents=True, exist_ok=True) 
    print("Directory to save frames is created :", str(saving_file_location))
    # =============================================================================================

    cap = cv2.VideoCapture()
    cap.open(str(input_file_location))
    num_total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("\nTotal number of frames found:", num_total_frame)
    start_t = time.time()

    n_saved = 0
    i_frame = 0

    print("Extracting frames ...")
    while i_frame < num_total_frame:
        if (i_frame > 0) & (np.mod(i_frame,math.floor(num_total_frame/num_training_img)) == 0):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
            _, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            fname = "{0}_frame_{1}.{2}".format(input_file_location.stem, i_frame, ext)
            cv2.imwrite(str(saving_file_location / fname), frame)
            if output is not None:
                cv2.imwrite(str(Path(output) / fname), frame)
            print('Progress: {}%'.format(int((i_frame+1)/num_total_frame*100)))
            n_saved = n_saved + 1
        else:
            pass
        i_frame = i_frame + 1

    print("Total number of frames saved:", n_saved)
    print("Total time taken:", time.time()-start_t)

    if log is not None:
        lines = [
            f"Total number of frames found: {num_total_frame}",
            f"Total number of frames saved: {n_saved}",
            f"Total time taken : {time.time()-start_t}",
        ]

        with open(log, 'w') as f:
            for line in lines:
                f.write(line + '\n')

if __name__ == '__main__':
    defopt.run(main)