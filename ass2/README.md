## Dependencies:
- ```tkinter```
- ```OpenCV 2.4.13```
- ```cPickle```
- ```scipy```


## Run Instructions:
For parts 1,2,3,4 run:
```$ python proc.py -s 1 imgs/Dec12.jpg imgs/Jan12.jpg```
```$ python proc.py -s 2 imgs/Dec12.jpg imgs/Jan12.jpg```
```$ python proc.py -s 3```
```$ python proc.py -s 4 imgs/Jan12.jpg```

For more details, please refer to the file ```proc.py``` and its associated docstring documentation for each function.

The observations are available in the folder out/. The archives are available in the archives/ folder.

## Description of Observations (Archives):
- manual is a Jan12 after being manually projected to the Dec12 P2 space.
- diff0, diff1, diff2, diff3, and res are outputs of the scene summarization module, the first four being the differences, and res being the resultant additive image.
- out_sift is the output after SIFT+Homography of 2 images (Jan12, Dec12)
- out_rect is the affine rectified image of the image, computed using vanishing points.
