import numpy as np
from PIL import Image
import os

IMAGE_THREE_PATH = '/mnt/zhengwen/model_synthesis/photo_from_life/three_channel'
IMAGE_FOUR_PATH = '/mnt/zhengwen/model_synthesis/photo_from_life/four_channel_0'

for img_name in sorted(os.listdir(IMAGE_THREE_PATH)):
    image = Image.open(os.path.join(IMAGE_THREE_PATH, img_name))
    image = np.asanyarray(image)
    if image.shape[2] == 4:
        image = Image.fromarray(np.uint8(image))
        image.save(os.path.join(IMAGE_FOUR_PATH, img_name))
    else:
        image = np.concatenate((image, 255 * np.zeros((64, 64, 1))), axis=-1)
        image = Image.fromarray(np.uint8(image))
        image.save(os.path.join(IMAGE_FOUR_PATH, img_name))

for img_name in sorted(os.listdir(IMAGE_FOUR_PATH)):
    image = Image.open(os.path.join(IMAGE_FOUR_PATH, img_name))
    image = np.asanyarray(image)
    print(image.shape)