from TestTubesClass import TestTubes
import matplotlib.pyplot as plt
import cv2

# Image location
img_path = "TestImage.jpg"

# Load image. openCV load it as BlueGreenRed so converting it to RGB
rgb_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

# Plot for debug
if 0:
    cv2.imshow("image", rgb_img)
    plt.figure()
    plt.imshow(rgb_img)
    plt.show()

# Initialize TestTubes class:
# detect test tubes location and return number of detected test tubes
tubes = TestTubes()
tubes.mainFunc(rgb_img)

for tube_num in range(0, tubes.getTestTubesNum()):

    # Return True if the corresponding tube of tube-num contains blue litmus paper in the given image
    isBlue = tubes.classifyTestTubeColor(rgb_img, tube_num)

    # Print result
    if isBlue:
        labelStr = "Blue"
    else:
        labelStr = "Red"
    print("Tube num=%d , classification %s" % (tube_num, labelStr))

plt.imshow(tubes.plotDetectedTubes(rgb_img))
plt.show()
