import os
import shutil
import cv2
import glob
import numpy as np
import pandas as pd

from tactile_data.utils import save_json_obj, load_json_obj, make_dir
from tactile_image_processing.image_transforms import process_image

BASE_DATA_PATH = 'temp'


def split_data(path, dir_names, split=0.8, seed=1):

    if type(dir_names) is str:
        dir_names = [dir_names]

    if not split:
        return dir_names

    all_dirs_out = []
    for dir_name in dir_names:

        # load target df
        targets_df = pd.read_csv(os.path.join(path, dir_name, 'targets.csv'))

        # indices to split data
        np.random.seed(seed)  # make deternministic, needs to be different from collect
        inds_true = np.random.choice([True, False], size=len(targets_df), p=[split, 1-split])
        inds = [inds_true, ~inds_true]
        dirs_out = ['_'.join([out, dir_name]) for out in ["train", "val"]]

        # iterate over split
        for dir_out, ind in zip(dirs_out, inds):

            dir_out = os.path.join(path, dir_out)
            make_dir(dir_out, check=False)

            # copy over parameter files
            for file_name in glob.glob(os.path.join(path, dir_name, '*_params.json')):
                shutil.copy(file_name, dir_out)

            # create dataframe pointing to original images (to avoid copying)
            targets_df.loc[ind, 'sensor_image'] = \
                rf'../../{dir_name}/images/' + targets_df[ind].sensor_image.map(str)
            targets_df[ind].to_csv(os.path.join(dir_out, 'targets.csv'), index=False)

        all_dirs_out = [*all_dirs_out, *dirs_out]

    return all_dirs_out


def process_data(path, dir_names, process_params={}):

    if type(dir_names) is str:
        dir_names = [dir_names]

    # iterate over dirs
    for dir_name in dir_names:

        # paths
        image_dir = os.path.join(path, dir_name, 'images')
        proc_image_dir = os.path.join(path, dir_name, 'processed_images')
        os.makedirs(proc_image_dir, exist_ok=True)

        # process images (include zeroth image)
        targets_df = pd.read_csv(os.path.join(path, dir_name, 'targets.csv'))
        si_0 = targets_df.sensor_image.sort_values()[0][:-6] + '_0.png'
        si_init_0 = targets_df.sensor_image.sort_values()[0][:-8] + '_init_0.png'

        cv2.namedWindow("processed_image")
        for sensor_image in [*list(targets_df.sensor_image), si_0, si_init_0]:
            try:
                image = cv2.imread(os.path.join(image_dir, sensor_image))
                proc_image = process_image(image, **process_params)
                image_path, proc_sensor_image = os.path.split(sensor_image)
                cv2.imwrite(os.path.join(proc_image_dir, proc_sensor_image), proc_image)
                print(f'processed {dir}: {sensor_image}')
            except AttributeError:
                print(f'missing {sensor_image}')
                continue

            # show image
            cv2.imshow("processed_image", proc_image)
            k = cv2.waitKey(1)
            if k == 27:    # Esc key to stop
                exit()

        # if targets have paths remove them
        if image_path:
            targets_df.loc[:, 'sensor_image'] = \
                targets_df.sensor_image.str.split('/', expand=True).iloc[:, -1]
            targets_df.to_csv(os.path.join(path, dir_name, 'targets.csv'), index=False)

        # save merged sensor_params and process_params
        sensor_params = load_json_obj(os.path.join(path, dir_name, 'sensor_params'))
        sensor_proc_params = {**sensor_params, **process_params}

        if 'bbox' in sensor_params and 'bbox' in sensor_proc_params:
            b, pb = sensor_params['bbox'], sensor_proc_params['bbox']
            sensor_proc_params['bbox'] = [b[0]+pb[0], b[1]+pb[1], b[0]+pb[2], b[1]+pb[3]]

        save_json_obj(sensor_proc_params, os.path.join(path, dir_name, 'sensor_process_params'))


if __name__ == "__main__":

    dir_names = ["data_1", "data_2"]

    process_image_params = {
        'dims': (128, 128),
        "bbox": (12, 12, 240, 240)
    }

    # dir_names = split_data(BASE_DATA_PATH, dir_names)
    process_image_data(BASE_DATA_PATH, dir_names, process_image_params)
