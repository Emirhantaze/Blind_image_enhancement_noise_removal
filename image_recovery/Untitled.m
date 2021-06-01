 clear all
I  = imread('m1.jpeg');
I=im2double(I);

PSF = fspecial('motion',24,0);
luc1 = deconvlucy(I,PSF,5);
imshow(luc1,[0 255])