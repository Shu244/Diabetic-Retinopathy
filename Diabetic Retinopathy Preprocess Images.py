import csv
import os
import numpy as np
import cv2

'''
Much thanks to https://www.kaggle.com/ratthachat/aptos-updatedv14-preprocessing-ben-s-cropping
'''

def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        '''
        For gray images.
        '''
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        # Convert to gray image in order to apply crop technique.
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        # image is too dark so that we crop out everything,
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):
            # return original image
            return img
        else:
            # Crop each channel using the crop dimensions from the gray
            # image then put them back together to get complete image.
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def load_ben_color(path, image_name, sigmaX=10):
    image = cv2.imread(path)  # Reads image in BGR format.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = crop_image_from_gray(image)
    image = cv2.resize(image, size)

    # Highlight feature of image.
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    horizontal_img = cv2.flip(image, 0)
    adjusted = adjust_gamma(horizontal_img, gamma=gamma)
    cv2.imwrite(os.path.join(DIR, image_name), image)
    cv2.imwrite(os.path.join(DIR, '~' + image_name), adjusted)


def list_of_images(csv_file):
    '''
    listdir doesn't list in a specific order.
    '''
    file = open(csv_file, "r")
    reader = csv.reader(file)

    # Skipping header
    next(reader)

    images = []
    for line in reader:
        image = line[0] + '.png'
        images.append(image)
    return images


def save_data(csv_file, image_folder_path, portions, portion_to_save):
    if portion_to_save >= portions:
        print('Invalid portion to save. Remember, portion to save starts at 0.')
        return None

    if not os.path.isdir(DIR):
        os.mkdir(DIR)

    images = list_of_images(csv_file)
    images_per_portion = np.floor(len(images) / portions)
    start = portion_to_save * images_per_portion  # Inclusive start.
    # Exclusive end.
    if portions == portion_to_save - 1:
        end = len(images)
    else:
        end = images_per_portion * (portion_to_save + 1)
    start = int(start)
    end = int(end)
    for name_index in range(start, end):
        name = images[name_index]
        image_path = os.path.join(image_folder_path, name)
        load_ben_color(path=image_path, image_name=name)


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


portions = 1
portion_to_save = 0
size = (256, 256)
gamma = 0.5
DIR = 'C:/Users/Shuha/Desktop/Diabetic Retinopathy/Dataset/Preprocessed_Data'


image_folder = 'C:/Users/Shuha/Desktop/Diabetic Retinopathy/Dataset/train_images'
csv_file = 'C:/Users/Shuha/Desktop/Diabetic Retinopathy/Dataset/train.csv'
save_data(csv_file, image_folder, portions, portion_to_save)
print('Data Saved')
