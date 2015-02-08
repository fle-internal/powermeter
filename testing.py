from numpy import *
import cv2
import time

cap = cv2.VideoCapture(0)

# fig = figure()
# rects = bar(range(3), [190] * 3)

def scale_image(img):
    lowcells = img == -Inf
    highcells = img == Inf
    badcells = img == nan
    img[lowcells | highcells | badcells] = numpy.median(img)
    minval = numpy.min(img)
    maxval = numpy.max(img)
    img[lowcells] = minval
    img[highcells] = maxval
    rng = float(maxval - minval)
    print minval, maxval, rng
    return array((img - minval) * 255.0 / rng, dtype=uint8)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    # cv2.imshow('frame', frame)

    # 0 = blue
    # 1 = green
    # 2 = red

    red = uint16(frame[:,:,2])
    green = uint16(frame[:,:,1])
    blue = uint16(frame[:,:,0])

    veryred = red > (blue * 2)
    veryblue = blue > (red * 2)

    mostred = red > ([numpy.median(green) + numpy.std(green) * 2])
    mostblue = blue - ([numpy.median(green) + numpy.std(green) * 3])

    # heatmap = red / log(blue)
    # heatmap = (red + blue) / float32(green)
    # heatmap = 2 ** scale_image(heatmap)

    heatmap = 2 ** mostblue

    image = scale_image(heatmap)

    # image[image < 150] = 0

    # q = 4 / 0

    # for rect, h in zip(rects, [int(mean(red)), int(mean(green)), int(mean(blue))]):
    #     rect.set_height(h)

    # fig.canvas.draw()

    # print int(mean(red)), int(mean(green)), int(mean(blue))


    cv2.imshow("frame", repeat(reshape(image, shape(image) + (1,)), 3, 2))

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

