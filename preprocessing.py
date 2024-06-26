#the code was modified from https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection/blob/376afb448852d4c7951458b93e171afc953500c0/2DNet/src/prepare_data.py

import PIL
import numpy as np
import pydicom as dicom
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Multiscale_net for ICH classification')

    # basic settings

    parser.add_argument('--dcm-dir', type=str)
    parser.add_argument('--img-dir', type=str)

    args = parser.parse_args()
    return args 


def get_first_of_dicom_field_as_int(x):
    if type(x) == dicom.multival.MultiValue:
        return int(x[0])
    return int(x)

def get_metadata_from_dicom(img_dicom):
    metadata = {
        "intercept": img_dicom.RescaleIntercept,
        "slope": img_dicom.RescaleSlope,
    }
    return {k: get_first_of_dicom_field_as_int(v) for k, v in metadata.items()}

def changing_window(img, window_center, window_width, intercept, slope):
    img = img * slope + intercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max
    return img

def normalize_minmax(img):
    mi, ma = img.min(), img.max()
    return (img - mi) / (ma - mi)

def preprocess_dicom(dcm_img):
    metadata = get_metadata_from_dicom(dcm_img)
    img1 = changing_window(dcm_img.pixel_array, 40, 80, **metadata)
    img1 = normalize_minmax(img1) * 255
    img2 = changing_window(dcm_img.pixel_array, 80, 200, **metadata)
    img2 = normalize_minmax(img2) * 255
    img3 = changing_window(dcm_img.pixel_array, 600, 2800, **metadata)
    img3 = normalize_minmax(img3) * 255
    img = np.dstack((img1, img2, img3))
    img = PIL.Image.fromarray(img.astype(np.int8), mode='RGB')
    return img

if __name__ == '__main__':

    args = parse_args()
    print(args)

    for f in os.listdir(arg.dcm_dir):
        png_name = f.replace('.dcm','.png')
        dcm_path = os.path.join(arg.dcm_dir,f)
        png_path = os.path.join(arg.img_dir,png_name)
        dcm_img = dicom.dcmread(dcm_path)
        png_img = preprocess_dicom(dcm_img)
        png_img.save(png_name)

