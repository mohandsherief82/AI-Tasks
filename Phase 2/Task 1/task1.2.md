# Report:

- First, we read the image then turn the format from cv2's BGR to HSV and apply the gaussian blur filter to decrease noise.
- Then, we create masks for red and blue colors using all their ranges in HSV.
- We apply the masks to the image.
- After that, we create the region proposals to each color and see whether either, both or none of the colors exist in the image.
- We show bounding boxes to the balls and add the class to them.
- Finally, we add findings into the file and show image.

### This whole step is repeated through all 20 photos in the balls folder.