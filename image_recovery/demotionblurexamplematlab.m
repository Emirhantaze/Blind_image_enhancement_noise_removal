Ioriginal = m; %imread('o1.jpeg');
figure, imshow(Ioriginal)
title('Original Image')
PSF = a;
Idouble = im2double(Ioriginal);
% blurred = imfilter(Idouble,PSF,'conv','circular');
% figure, imshow(blurred)
title('Blurred Image')  
wnr1 = deconvwnr(Idouble,PSF,0.0001);
figure, imshow(wnr1)
title('Restored Blurred Image')