import numpy as np
import os
import cv2
import time
import sys


def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:, :, 2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1


def transform_image(img, ang_range, shear_range, trans_range):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over. 

    A Random uniform distribution is used to generate different parameters for transformation

    '''
    # Rotation

    ang_rot = np.random.uniform(ang_range) - ang_range / 2
    rows, cols, ch = img.shape
    Rot_M = cv2.getRotationMatrix2D((cols / 2, rows / 2), ang_rot, 1)

    # Translation
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    tr_y = trans_range * np.random.uniform() - trans_range / 2
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])

    # Shear
    pts1 = np.float32([[5, 5], [20, 5], [5, 20]])

    pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
    pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2

    # Brightness
    pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])

    shear_M = cv2.getAffineTransform(pts1, pts2)

    img = cv2.warpAffine(img, Rot_M, (cols, rows))
    img = cv2.warpAffine(img, Trans_M, (cols, rows))
    img = cv2.warpAffine(img, shear_M, (cols, rows))

    img = augment_brightness_camera_images(img)

    return img


def transform_images(data_dir='GOT/', minimum_files_required=500):
    contents = os.listdir(data_dir)
    classes = [each for each in contents if os.path.isdir(data_dir + each)]

    for each in classes:
        print("Transforming {} images:".format(each))

        class_path = data_dir + each
        files = os.listdir(class_path)
        max = len(files) + 1

        if max < minimum_files_required:
            print('Total images before transformations:', max-1)

            for ii, file in enumerate(files, 1):

                img = cv2.imread(os.path.join(class_path, file))

                # transformed
                for h in range(12):
                    changed = transform_image(img, 20, 10, 5)

                    try:
                        cv2.imwrite(os.path.join(class_path, str(max) + ".png"), changed)
                        max += 1
                    except:
                        print('skipped..')
                        pass
            print('Total images after transformations:', max)
            print('Done!')
        else:
            print('Minimum files required found! No transformations applied.')


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        time.sleep(3)
        return

    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()


def one_hot_encode(x):
    x = [val[0] for val in x]
    n_classes = len(list(set(x)))
    class_array = np.zeros(n_classes)
    labels = sorted(list(set(x)))

    print('labels', labels)
    output = []

    for label in x:
        label_vector = np.copy(class_array)
        index = labels.index(label)
        label_vector[index] = 1
        output.append(label_vector)
    return np.array(output), n_classes


def scheduler(epoch):
    current_lr = 0.001
    epoch_step = 10
    if epoch == 0:
        updated_lr = current_lr
        print('Initial LR set to:', updated_lr)
    elif epoch % epoch_step == 0:
        dividend = epoch // epoch_step
        updated_lr = current_lr/dividend
        print('LR updated to:', updated_lr)
    else:
        updated_lr = current_lr
    return updated_lr
