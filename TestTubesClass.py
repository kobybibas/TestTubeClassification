import cv2
import numpy as np
import pandas as pd
import scipy.stats as st


class TestTubes:
    # TestTubes class.
    # contains functionality of detect test tube and classify the litmus paper of it

    def mainFunc(self, img):
        # 1. set the detection and classification parameter.
        # 2. convert image to hsv
        # 3. detect litmus paper location

        self.img = img

        # Initialize parameters for litmus paper classification
        self.red_lower_bound = np.array([170, 50, 15], dtype=float)
        self.red_upper_bound = np.array([10, 200, 220], dtype=float)
        self.blue_lower_bound = np.array([90, 20, 5], dtype=float)
        self.blue_upper_bound = np.array([115, 150, 100], dtype=float)
        self.brown_lower_bound = np.array([3, 100, 80], dtype=float)
        self.brown_upper_bound = np.array([15, 215, 245], dtype=float)

        self.red_to_blue_ratio = 1.75  # area ratio between red to blue for litmus classification
        self.litmus_start_var = 1.6  # where to start searching litmus paper
        self.litmus_height_var = 1.2  # where to finish search litmus paper
        self.litmus_width_ratio = 1.0  # litmus width ratio from cork width
        self.max_height_to_width_ratio = 1.5  # max ratio between height to width. above it splits to 2 corks
        self.total_red_ratio_threshold = 0.02

        # Convert image to HSV
        self.img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # data base which contains the test-tubes litmus paper and cork location
        self.test_tubes_db = self.detectTubesLocation()

    def setDumpPath(self, dump_path, image_num):
        # Dump path in which middle results will be saved

        self.dump_path = dump_path + '\\' + str(image_num)

    def dumpHsvImage(self):
        # Dump each channel of HSV image

        hsv_img_with_tubes = self.plotDetectedTubes(self.img_hsv)
        h, s, v = cv2.split(hsv_img_with_tubes)
        cv2.imwrite(self.dump_path + '_h.png', h)
        cv2.imwrite(self.dump_path + '_s.png', s)
        cv2.imwrite(self.dump_path + '_v.png', v)

    def dumpDetectedCorksMask(self):
        # Dump the binary results of cork detection

        cv2.imwrite(self.dump_path + '_CorcksMask.png', self.mask_corks * 255)

    def dumpDetectedTubes(self):
        # Dump image with bounding boxes on the detected tubes

        img_for_plot = self.plotDetectedTubes(self.img)
        cv2.imwrite(self.dump_path + '_detectedTubes.png', cv2.cvtColor(img_for_plot, cv2.COLOR_RGB2BGR))

    def dumpLitmusRegion_masks(self):
        # Dump binary mask of the litmus paper- red mask and blue mask

        # Keep only big red thing- test-tubes corks
        mask_red = self.filterRedOrBlueFromImg(self.img_hsv, 'red')
        mask_blue = self.filterRedOrBlueFromImg(self.img_hsv, 'blue')

        # Convert in order to save properly
        mask_red = cv2.cvtColor(mask_red, cv2.COLOR_GRAY2RGB)
        mask_blue = cv2.cvtColor(mask_blue, cv2.COLOR_GRAY2RGB)
        mask_red_with_location = self.plotDetectedTubes(mask_red)
        mask_blue_with_location = self.plotDetectedTubes(mask_blue)

        # Save
        cv2.imwrite(self.dump_path + '_redMask.png', mask_red_with_location * 255)
        cv2.imwrite(self.dump_path + '_blueMask.png', mask_blue_with_location * 255)

    def dumpLitmusRegion(self, litmus_num):
        # Dump image with bounding box of the litmus paper

        litmus_in_img = self.img[self.litmus_region[1]:self.litmus_region[1] + self.litmus_region[3],
                        self.litmus_region[0]:self.litmus_region[0] + self.litmus_region[2]]
        cv2.imwrite(self.dump_path + "_" + str(litmus_num) + ".png", cv2.cvtColor(litmus_in_img, cv2.COLOR_RGB2BGR))

    def plotDetectedTubes(self, img_for_plot):
        # Mark on input image the tubes location

        # Plotting search area for debug
        for k in range(0, self.test_tubes_db.shape[0]):
            pt1 = (int(self.test_tubes_db.loc[k, 'cork_x_min']), int(self.test_tubes_db.loc[k, 'cork_y_min']))
            pt2 = (int(self.test_tubes_db.loc[k, 'cork_x_min'] + self.test_tubes_db.loc[k, 'cork_width']),
                   int(self.test_tubes_db.loc[k, 'cork_y_min'] + self.test_tubes_db.loc[k, 'cork_height']))
            cv2.rectangle(img_for_plot, pt1, pt2, (0, 255, 0), 12, 8, 0)
            cv2.putText(img_for_plot, str(k),
                        (int(self.test_tubes_db.loc[k, 'cork_x_min'] + self.test_tubes_db.loc[k, 'cork_width'] / 3),
                         int(self.test_tubes_db.loc[k, 'cork_y_min'] + self.test_tubes_db.loc[k, 'cork_height'] / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            pt1 = (int(self.test_tubes_db.loc[k, 'litmus_x_min']), int(self.test_tubes_db.loc[k, 'litmus_y_min']))
            pt2 = (int(self.test_tubes_db.loc[k, 'litmus_x_min'] + self.test_tubes_db.loc[k, 'litmus_width']),
                   int(self.test_tubes_db.loc[k, 'litmus_y_min'] + self.test_tubes_db.loc[k, 'litmus_height']))

            # Mark on the image
            cv2.rectangle(img_for_plot, pt1, pt2, (0, 120, 120), 4, 8, 0)

        # fig = plt.figure()
        # plt.imshow(img_for_plot)
        # plt.show()

        return img_for_plot

    def getTestTubesNum(self):
        # output: number of test tubes in the data base

        return self.test_tubes_db.shape[0]

    def filterBrownFromImg(self, img_hsv):
        # Threshold the image so only the brown pixels will appear in order to detect the cork

        # Threshold the image
        lower_bound = self.brown_lower_bound
        upper_bound = self.brown_upper_bound
        mask = cv2.inRange(img_hsv, lower_bound, upper_bound)

        # Remove small red blobs
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))  # (20, 20))
        mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # makes the object in bigger
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (22, 22))  # (22, 22))
        mask_dilated = cv2.dilate(mask_opened, kernel, iterations=1)

        # remove small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        mask_closed = cv2.morphologyEx(mask_dilated, cv2.MORPH_CLOSE, kernel)

        # Threshold output in order to change class
        maskOutput = cv2.threshold(mask_closed, 60, 255, cv2.THRESH_BINARY)[1]

        # Save for dump
        self.mask_corks = maskOutput
        return maskOutput

    def filterRedOrBlueFromImg(self, img_hsv, color):
        # Input: HSV img
        # Output: mask contains only red blobs

        # Threshold the HSV image to get only red colors
        if color == 'red':
            # define range of blue color in HSV
            lower_bound = self.red_lower_bound
            upper_bound = self.red_upper_bound
        elif color == 'blue':
            # define range of blue color in HSV
            lower_bound = self.blue_lower_bound
            upper_bound = self.blue_upper_bound

        # Run mask for each channel
        mask = np.ones((img_hsv.shape[0], img_hsv.shape[1]), dtype=bool)
        for channel in range(0, 3):

            if lower_bound[channel] < upper_bound[channel]:
                mask = mask & cv2.inRange(img_hsv[:, :, channel], lower_bound[channel], upper_bound[channel])

            else:  # upper_bound[channel] < lower_bound[channel]
                mask = mask & (cv2.inRange(img_hsv[:, :, channel], lower_bound[channel], 255)
                               | cv2.inRange(img_hsv[:, :, channel], 0, upper_bound[channel]))

        return mask

    def findContours(self, mask):
        # Find blobs in mask. store their properties in data base
        # Input: mask- blobs to be search in
        # Output: contour_db- data base of the blobs

        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0]

        # Organize contour- [x_center y_center width height]
        contour_db = pd.DataFrame(0, index=np.arange(cnts.__len__()),
                                  columns=['x_center', 'y_center', 'width', 'height', 'x_min', 'y_min'])

        # Get average height
        avg_height = 0
        k = 0
        for c in cnts:
            [x_min, y_min, width, height] = cv2.boundingRect(c)
            avg_height += height
            k = k + 1
        avg_height = avg_height / (k + 0.00001)

        k = 0
        for c in cnts:
            [contour_db.loc[k, 'x_min'],
             contour_db.loc[k, 'y_min'],
             contour_db.loc[k, 'width'],
             contour_db.loc[k, 'height']] = cv2.boundingRect(c)

            contour_db.loc[k, 'x_max'] = contour_db.loc[k, 'x_min'] + contour_db.loc[k, 'width']
            contour_db.loc[k, 'y_max'] = contour_db.loc[k, 'y_min'] + contour_db.loc[k, 'height']
            # compute the center of the contour
            M = cv2.moments(c)
            contour_db.loc[k, 'x_center'] = int(M["m10"] / M["m00"])
            contour_db.loc[k, 'y_center'] = int(M["m01"] / M["m00"])

            # if too width- divide to 2 corks
            if avg_height * self.max_height_to_width_ratio < contour_db.loc[k, 'width']:
                contour_db.loc[k, 'width'] = contour_db.loc[k, 'width'] / 2
                contour_db.loc[k, 'x_center'] = contour_db.loc[k, 'x_min'] + contour_db.loc[k, 'width'] / 2
                contour_db.loc[k, 'y_center'] = contour_db.loc[k, 'y_min'] + contour_db.loc[k, 'height'] / 2

                contour_db.loc[k + 1, 'x_min'] = contour_db.loc[k, 'x_min'] + contour_db.loc[k, 'width']
                contour_db.loc[k + 1, 'y_min'] = contour_db.loc[k, 'y_min']
                contour_db.loc[k + 1, 'width'] = contour_db.loc[k, 'width']
                contour_db.loc[k + 1, 'height'] = contour_db.loc[k, 'height']
                contour_db.loc[k + 1, 'x_center'] = contour_db.loc[k + 1, 'x_min'] + contour_db.loc[k + 1, 'width'] / 2
                contour_db.loc[k + 1, 'y_center'] = contour_db.loc[k + 1, 'y_min'] + contour_db.loc[k + 1, 'height'] / 2
                k = k + 1
            k = k + 1

        return contour_db

    def buildTestTubesDB(self, contour_db):
        # building test-tube data base which include: cork location and sizes, litmus paper estimate location
        # Input: contour_db- data base which contains all the blob in the image and their location
        # Output: test_tube_db- data base which contains the cork location and the expected litmus paper location.
        #         data base format: 'tube_num', 'cork_x_min',   'cork_y_min'  , 'cork_width'  , 'cork_height'  ,
        #                                       'litmus_x_min', 'litmus_y_min', 'litmus_width', 'litmus_height'.

        # Sort by data base by y center ceter
        contour_db = contour_db.sort_values(['y_center'], ascending=True)
        contour_db = contour_db.reset_index()
        if 0:
            print(contour_db)

        # Filter blobs outside maximum min blobs
        contour_db = contour_db[contour_db['y_min'] < contour_db.loc[0, 'y_max']]

        # Sort by x location- first in the left. starting from 0
        contour_db = contour_db.sort_values(['x_center'], ascending=True)
        contour_db = contour_db.reset_index()

        # Get average width
        avg_width = contour_db['width'].mean()

        # building test tubes data base- cork and litmus paper location
        litmus_start_var = self.litmus_start_var  # where to start searching litmus paper
        litmus_height_var = self.litmus_height_var  # where to finish search litmus paper
        litmus_width_ratio = self.litmus_width_ratio
        test_tubes_db = pd.DataFrame(0, index=np.arange(contour_db.shape[0]),
                                     columns=['tube_num', 'cork_x_min', 'cork_y_min', 'cork_width', 'cork_height',
                                              'litmus_x_min', 'litmus_y_min', 'litmus_width', 'litmus_height'])
        for k in range(0, contour_db.shape[0]):
            test_tubes_db.loc[k, 'tube_num'] = k
            test_tubes_db.loc[k, 'cork_x_min'] = contour_db.loc[k, 'x_center'] - avg_width / 2
            test_tubes_db.loc[k, 'cork_y_min'] = contour_db.loc[k, 'y_center'] - avg_width / 2
            test_tubes_db.loc[k, 'cork_width'] = avg_width
            test_tubes_db.loc[k, 'cork_height'] = avg_width
            test_tubes_db.loc[k, 'litmus_center_x'] = contour_db.loc[k, 'x_center']
            test_tubes_db.loc[k, 'litmus_y_min'] = test_tubes_db.loc[k, 'cork_y_min'] + test_tubes_db.loc[
                k, 'cork_height'] + litmus_start_var * test_tubes_db.loc[k, 'cork_height']
            test_tubes_db.loc[k, 'litmus_width'] = contour_db.loc[
                                                       k, 'width'] * litmus_width_ratio  # Same width as th cork
            test_tubes_db.loc[k, 'litmus_height'] = litmus_height_var * test_tubes_db.loc[k, 'cork_height']

            test_tubes_db.loc[k, 'litmus_x_min'] = test_tubes_db.loc[k, 'litmus_center_x'] - test_tubes_db.loc[
                k, 'litmus_width'] / 2  # same x_min as th cork
        return test_tubes_db

    def detectTubesLocation(self):
        # Detect the tube location based on the cork position

        # Keep only big red thing- test-tubes corks
        mask = self.filterBrownFromImg(self.img_hsv)

        # Find contours from the filtered image
        contour_db = self.findContours(mask)

        # Build data base of the test-tubes which includes: cork and litmus paper locations and sizes
        test_tubes_db = self.buildTestTubesDB(contour_db)

        return test_tubes_db

    # Given a test-tube number: find the litmus paper color: red or blue
    def classifyTestTubeColor(self, img, test_tube_num):

        # Convert image to HSV
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Keep only big red thing- test-tubes corks
        mask_blue = self.filterRedOrBlueFromImg(img_hsv, 'blue')
        mask_red = self.filterRedOrBlueFromImg(img_hsv, 'red')

        # Crop litmus paper region from relevant testTube num
        x_min = int(self.test_tubes_db.loc[test_tube_num, 'litmus_x_min'])
        x_max = int(self.test_tubes_db.loc[test_tube_num, 'litmus_x_min'] + \
                    self.test_tubes_db.loc[test_tube_num, 'litmus_width'])
        y_min = int(self.test_tubes_db.loc[test_tube_num, 'litmus_y_min'])
        y_max = int(self.test_tubes_db.loc[test_tube_num, 'litmus_y_min'] + \
                    self.test_tubes_db.loc[test_tube_num, 'litmus_height'])

        # save for debug
        self.litmus_region = np.array([x_min, y_min, x_max - x_min, y_max - y_min])

        litmus_region_blue_mask = mask_blue[y_min:y_max, x_min:x_max]
        litmus_region_red_mask = mask_red[y_min:y_max, x_min:x_max]

        # Weight the pixels- further away from the middle, less weight
        gaussian = self.gaussianKernel(litmus_region_blue_mask.shape[1], litmus_region_blue_mask.shape[0], 1.5)
        litmus_region_blue_mask = litmus_region_blue_mask * gaussian
        litmus_region_red_mask = litmus_region_red_mask * gaussian

        # Check if Blue is more than half than Red
        red_area = litmus_region_red_mask.sum()
        blue_area = litmus_region_blue_mask.sum()
        ratio = float(red_area) / (float(blue_area) + 0.000000001)

        isBlue = False
        if self.red_to_blue_ratio >= ratio:
            isBlue = True

        total_red_ratio = float(red_area) / litmus_region_red_mask.shape[0] * litmus_region_red_mask.shape[1]
        if total_red_ratio < self.total_red_ratio_threshold:
            isBlue = True

        # Return the class of the litmus paper
        return isBlue

    def gaussianKernel(self, kernlen_x, kernlen_y, nsig=3):
        # Returns a 2D Gaussian kernel array.

        interval_x = (2 * nsig + 1.) / (kernlen_x)
        x = np.linspace(-nsig - interval_x / 2., nsig + interval_x / 2., kernlen_x + 1)
        kern1d_x = np.diff(st.norm.cdf(x))

        interval_y = (2 * nsig + 1.) / (kernlen_y)
        y = np.linspace(-nsig - interval_y / 2., nsig + interval_y / 2., kernlen_y + 1)
        kern1d_y = np.diff(st.norm.cdf(y))

        kernel_raw = np.sqrt(np.outer(kern1d_y, kern1d_x))
        kernel = kernel_raw / kernel_raw.sum()
        return kernel
