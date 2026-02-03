% read and save images
img         = imread('1024_books_original.png');
%img         = imread('test_images/originals/1024_books_original.png');
img         = double(img)/255;

img(42,:)   = 0; 

figure; imshow(img);

imwrite(img,'meaning.png','png');