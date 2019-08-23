Test Tube Classification
=====

Classification of litmus paper between 2 classes: red and blue.

The code is wrriten in python 3.6 with the follwoing depedencies: 
  1. cv2
  2. numpy
  3. matplotlib
  4. pandas
  5. scipy
  
  
Usage
=====
* Input: test with test tubes to be classified.  
* Output: class of each litmus paper.

Input:
<a rel="some text"><p align="center"><img src="https://i.imgur.com/cCFzT18.jpg" height="500"></p></a>

Output:
<a rel="some text"><p align="center"><img src="https://i.imgur.com/fEzzrX4.jpg" height="300"></p></a>  


Algorithem description
====================
Given an input image:

1. Search for brown blobs (test tube cork color).
2. Build data base of the cork location
3. based of the cork location crop the expected litmus paper region.
4. classify the litmus paper color based on blue to red ration.
