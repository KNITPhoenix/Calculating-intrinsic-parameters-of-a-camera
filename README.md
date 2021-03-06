# Calculating-intrinsic-parameters-of-a-camera
Here we calculate intrinsic parameters of a camera to project a 3d world object to 2d image plane of camera.
Figure below shows an image of the checkerboard, where XY Z is the world coordinate and xy is marked as the image coordinate. The edge length of each grid on the checkerboard is 10mm in reality. Suppose one pixel of the image is equivalent to 1mm. You can calculate the projection matrix from world coordinate to image coordinate based on the 32 marked points on the checkerboard. From the projection matrix you can get the intrinsic matrix.

![alt text](https://github.com/KNITPhoenix/Calculating-intrinsic-parameters-of-a-camera/blob/main/checkboard.png?raw=true)
