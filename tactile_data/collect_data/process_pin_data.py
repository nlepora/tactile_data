import os
import warnings
import cv2
import numpy as np
import pandas as pd

from tactile_data.utils import save_json_obj

from vsp.video_stream import CvVideoDisplay
from vsp.detector import CvBlobDetector
from vsp.detector import CvContourBlobDetector
from vsp.detector import SklDoHBlobDetector
# from vsp.detector import SkeletonizePeakDetector
from vsp.encoder import KeypointEncoder
from vsp.view import KeypointView

warnings.simplefilter('always', UserWarning)

BASE_DATA_PATH = 'temp'


def process_pin_data(path, dir_names, pin_extraction_params={}):

    if type(dir_names) is str:
        dir_names = [dir_names]

    # set keypoint detector
    if pin_extraction_params['detector_type'] == 'blob':
        detector = CvBlobDetector(**pin_extraction_params['detector_kwargs'])
    elif pin_extraction_params['detector_type'] == 'contour':
        detector = CvContourBlobDetector(**pin_extraction_params['detector_kwargs'])
    elif pin_extraction_params['detector_type'] == 'doh':
        detector = SklDoHBlobDetector(**pin_extraction_params['detector_kwargs'])
    # elif pin_extraction_params['detector_type'] == 'peak':
    #     detector = SkeletonizePeakDetector(**pin_extraction_params['detector_kwargs'])

    encoder = KeypointEncoder()
    view = KeypointView(color=(0, 255, 0))
    display = CvVideoDisplay(name='blob')
    display.open()

    # iterate over dirs
    for dir_name in dir_names:

        # paths
        image_dir = os.path.join(path, dir_name, 'images')
        if not os.path.exists(image_dir):
            image_dir = os.path.join(path, dir_name, 'processed_images')
        keypoints_dir = os.path.join(path, dir_name, 'extracted_pins')
        os.makedirs(keypoints_dir, exist_ok=True)

        # process images
        targets_df = pd.read_csv(os.path.join(path, dir_name, 'targets.csv'))
        keypoints_filenames = []
        rows_for_deletion = []
        for i, sensor_image in enumerate(targets_df.sensor_image):
            print(f'processed {image_dir}: {sensor_image}')

            image = cv2.imread(os.path.join(image_dir, sensor_image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, image_name = os.path.split(sensor_image)

            # extract keypoints from image
            extracted_keypoints = encoder.encode(detector.detect(image))
            image = view.draw(image, extracted_keypoints)
            display.write(image)

            # display features of extracted keypoints
            n_extracted_keypoints = extracted_keypoints.shape[0]
            print(f'extracted {n_extracted_keypoints} keypoints')

            if n_extracted_keypoints != pin_extraction_params['n_pins']:
                rows_for_deletion.append(i)
                warnings.warn("Couldn't extract the correct number of keypoints, skipping entry.")
                continue

            # ensure that keypoint size is consistent
            keypoints = np.zeros(shape=(pin_extraction_params['n_pins'], 2))
            keypoints[:n_extracted_keypoints, ...] = extracted_keypoints[:pin_extraction_params['n_pins'], :2]

            # sort the detected keypoints for consistency
            sorted_ind = np.lexsort((keypoints[:, 1], keypoints[:, 0]))
            keypoints = keypoints[sorted_ind]

            # normalise the keypoints by the image size
            keypoints[:, 0] = keypoints[:, 0] / image.shape[1]
            keypoints[:, 1] = keypoints[:, 1] / image.shape[0]
            # keypoints[:, 0] = keypoints[:, 0] / pin_extraction_params['image_width']
            # keypoints[:, 1] = keypoints[:, 1] / pin_extraction_params['image_height']

            # save keypoints as npy file
            keypoints_filename = f"keypoints_{os.path.splitext(image_name)[0].split('_')[1]}.npy"
            keypoints_filenames.append(keypoints_filename)
            np.save(os.path.join(keypoints_dir, keypoints_filename), keypoints)

        # drop rows where pin extraction did not work as expected
        targets_df = targets_df.drop(rows_for_deletion)

        # add keypoint names to csv
        targets_df["keypoints_filename"] = keypoints_filenames
        targets_df.to_csv(os.path.join(path, dir_name, 'new_targets.csv'), index=False)

        # save merged sensor_params and process_params
        save_json_obj(pin_extraction_params, os.path.join(path, dir_name, 'pin_extraction_params'))


if __name__ == "__main__":

    dir_names = ["data_1", "data_2"]

    pin_extraction_params = {
        'n_pins': 127,
        # 'n_pins': 331,

        'image_width': 280,
        'image_height': 280,

        # 'detector_type': 'blob',
        # 'detector_kwargs': {
        #       'min_threshold': 82,
        #       'max_threshold': 205,
        #       'filter_by_color': True,
        #       'blob_color': 255,
        #       'filter_by_area': True,
        #       'min_area': 35,
        #       'max_area': 109,
        #       'filter_by_circularity': True,
        #       'min_circularity': 0.60,
        #       'filter_by_inertia': True,
        #       'min_inertia_ratio': 0.25,
        #       'filter_by_convexity': True,
        #       'min_convexity': 0.47,
        #   }

        # 'detector_type': 'contour',
        # 'detector_kwargs': {
        #     'blur_kernel_size': 7,
        #     'thresh_block_size': 15,
        #     'thresh_constant': -16.0,
        #     'min_radius': 4,
        #     'max_radius': 7,
        # },

        'detector_type': 'doh',
        'detector_kwargs': {
            'min_sigma': 5.0,
            'max_sigma': 6.0,
            'num_sigma': 5,
            'threshold': 0.015,
        },

        # 'detector_type': 'peak',
        # 'detector_kwargs': {
        #     'blur_kernel_size': 9,
        #     'min_distance': 10,
        #     'threshold_abs': 0.4346,
        #     'num_peaks': 331,
        #     'thresh_block_size': 11,
        #     'thresh_constant': -34.0,
        # },
    }

    process_pin_data(BASE_DATA_PATH, dir_names, pin_extraction_params)
