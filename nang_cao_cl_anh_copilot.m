% Thuật toán nâng cao chất lượng ảnh đầu vào:
clc; clear; close all;

%% ==== PART 1: Đọc ảnh giao thoa ===
addpath("C:\Users\admin\Máy tính\Lab thầy Tùng\Tài liệu a Tuân\Ảnh mẫu"); % thư mục chứa ảnh
img_name = "anh_nham_chuan.bmp";
Img_Original = imread(img_name);

% Hiển thị ảnh và cho phép chọn vùng crop bằng chuột
figure;
imshow(Img_Original);
title('Dùng chuột để chọn vùng cần crop, sau đó nhấn Enter');

% Dùng chuột chọn vùng và nhấn Enter để xác nhận
Img_Cropped = imcrop;

% Hiển thị ảnh đã crop
figure;
imshow(Img_Cropped);
title('Ảnh sau khi crop');

Img_Original = Img_Cropped;
%%
% Đọc và chuyển xám
img = Img_Cropped;
gray = rgb2gray(img);

% Tăng tương phản
gray = imadjust(gray);

% Lọc Gabor để làm nổi bật vân (có thể thử nhiều hướng)
wavelength = 8; orientation = 0;
gaborArray = gabor(wavelength, orientation);
gaborMag = imgaborfilt(gray, gaborArray);

% Chuẩn hóa và nhị phân hóa
gaborMag = mat2gray(gaborMag); % chuẩn hóa về 0-1
bw = imbinarize(gaborMag, 'adaptive');

% Làm mượt (loại bỏ nhiễu nhỏ)
bw = bwareaopen(bw, 50);

% Skeletonize (tùy chọn)
bw = bwmorph(bw, 'skel', Inf);

% Đếm số vân (đếm số đối tượng hoặc đường nối)
cc = bwconncomp(bw);
numVans = cc.NumObjects;

imshow(bw);
title(['Số vân đếm được: ', num2str(numVans)]);