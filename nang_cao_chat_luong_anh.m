%% Off-Axis Holography: Real & 3D Surface Reconstruction from Fringes
clc; clear; close all;

%% ==== PART 1: Đọc ảnh và crop ====
addpath("C:\Users\admin\Máy tính\Lab thầy Tùng\Tài liệu a Tuân\Ảnh mẫu");
img_name = "anh_nham_chuan.bmp";
Img_Original = imread(img_name);

figure; imshow(Img_Original);
title('Chọn vùng cần crop rồi nhấn Enter');
Img_Cropped = imcrop;
Img_Original = Img_Cropped;

%% ==== PART 2: Chuyển sang ảnh xám ====
if size(Img_Original, 3) == 3
    gray = rgb2gray(Img_Original);
else
    gray = Img_Original;
end
gray = im2double(gray);

%% ==== PART 3: Tăng cường vân kết hợp (CLAHE + Tách nền + Unsharp) ====
clahe_img = adapthisteq(im2uint8(gray));
gray_clahe = im2double(clahe_img);
background = imgaussfilt(gray_clahe, 15);
fringe_only = mat2gray(gray_clahe - background);  % chuẩn hóa

h_unsharp = fspecial('unsharp');
enhanced = imfilter(fringe_only, h_unsharp, 'replicate');

figure;
imshow(enhanced, []);
title("Ảnh sau khi tăng cường vân");

%% ==== PART 4: Chuyển ảnh nhị phân (KHÔNG dùng Opening & Closing) ====
BW = imbinarize(enhanced, 'adaptive', 'Sensitivity', 0.5, 'ForegroundPolarity', 'dark');

figure;
imshow(BW);
title("Ảnh nhị phân (Không Opening & Closing)");

%% ==== PART 5: Làm mảnh vân (Skeletonization) và loại râu ria ====
BW_skeleton = bwmorph(BW, 'skel', Inf);
BW_pruned = bwareaopen(BW_skeleton, 10);  % Loại bỏ nhánh < 10 px
BW = BW_pruned;
vung_chon = BW;

figure;
imshow(BW);
title("Skeletonized + loại nhánh nhỏ");

%% ==== BỔ SUNG: Loại bỏ vân râu ria bằng lọc theo độ dài và hướng ====
[H_tmp, theta_tmp, rho_tmp] = hough(BW);
P_tmp = houghpeaks(H_tmp, 10);
lines_tmp = houghlines(BW, theta_tmp, rho_tmp, P_tmp, 'FillGap', 5, 'MinLength', 20);

angles = [lines_tmp.theta];
avg_angle = mean(angles);

BW_clean = false(size(BW));
angle_threshold = 10; % độ lệch cho phép

for k = 1:length(lines_tmp)
    if abs(lines_tmp(k).theta - avg_angle) < angle_threshold
        xy = [lines_tmp(k).point1; lines_tmp(k).point2];
        BW_clean = insertShape(double(BW_clean), 'Line', [xy(1,:) xy(2,:)], ...
                               'Color', 'white', 'LineWidth', 1);
    end
end

BW_clean = imbinarize(rgb2gray(BW_clean));
BW = BW_clean;

figure;
imshow(BW);
title("Sau khi lọc vân râu ria theo hướng và độ dài");

%% ==== PART 6: Ước lượng hướng vân bằng Hough Transform ====
[H, theta, rho] = hough(vung_chon);
P = houghpeaks(H, 5);
lines = houghlines(vung_chon, theta, rho, P);

avg_angle = mean([lines.theta]);
goc_vuong_goc = avg_angle + 90;

[H_img, W_img] = size(vung_chon);
x_center = W_img / 2;
y_center = H_img / 2;
slope = tand(goc_vuong_goc);
x1 = 1; x2 = W_img;
y1 = y_center + slope * (x1 - x_center);
y2 = y_center + slope * (x2 - x_center);

% %% ==== PART 7: Hiển thị kết quả cuối ====
% angle_deg = -90 + goc_vuong_goc;
% BW_rotated = imrotate(BW, angle_deg, 'bilinear', 'crop');
% BW_rotated = BW_rotated(50:end-50, 50:end-50);  % Crop viền
% 
% figure('Name', 'Skeletonized and Rotated Fringes');
% imshow(BW_rotated); 
% title('Skeletonized Fringes (Rotated & Cropped)');
