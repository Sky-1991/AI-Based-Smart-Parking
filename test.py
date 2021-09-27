# -*- coding: utf-8 -*-
"""test

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qnWalHI5jsKAHnZahVc2WAQcMCj9TbL3
"""

#test.py
import numpy as np
import cv2

# Open a sample video available in sample-videos
vcap = cv2.VideoCapture(1)
#if not vcap.isOpened():
#    print "File Cannot be Opened"

while True:
    # Capture frame-by-frame
    ret, frame = vcap.read()
    # print cap.isOpened(), ret
    if frame is not None:
        # Display the resulting frame
        cv2.imshow('video', frame)
        # Press q to close the video windows before it ends if you want
        if cv2.waitKey(22) & 0xFF == ord('q'):
            cv2.imwrite("images/2hacknitj.png", frame)
            break
    else:
        print ("Frame is None")
        break

# When everything done, release the capture
vcap.release()
cv2.destroyAllWindows()
print ("Video stop")