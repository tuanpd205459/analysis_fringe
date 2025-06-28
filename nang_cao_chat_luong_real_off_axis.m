%% Off-Axis Holography Real & 3D Surface Reconstruction from Fringes
% -------------------------------------------------------------------------
% Thuật toán nâng cao chất lượng ảnh đầu vào
clc; clear; close all;

%% ==== PART 1: Đọc ảnh và crop ====
addpath("C:\Users\admin\Máy tính\Lab thầy Tùng\Tài liệu a Tuân\Ảnh mẫu");
img_name = "anh_nham_chuan.bmp";
Img_Original = imread(img_name);

figure;
imshow(Img_Original);
title('Dùng chuột để chọn vùng cần crop, sau đó nhấn Enter');
Img_Cropped = imcrop;  % Chọn vùng bằng chuột
Img_Original = Img_Cropped;

%% ==== PART 2: Chuyển sang ảnh xám ====
if size(Img_Original, 3) == 3
    gray = rgb2gray(Img_Original);
else
    gray = Img_Original;
end
gray_double = im2double(gray);

%% ==== PART 3: Thử các phương pháp tăng cường ====

% 1. Cân bằng histogram toàn cục
heq_img = histeq(gray);

% 2. Cân bằng histogram cục bộ (CLAHE)
clahe_img = adapthisteq(gray);

% 3. Tăng biên độ tần số cao (lọc unsharp)
h = fspecial('unsharp');
unsharp_img = imfilter(gray_double, h, 'replicate');

% 4. Tăng độ tương phản bằng hệ số alpha
alpha = 1.5;
contrast_img = imadjust(gray_double, stretchlim(gray_double), [], alpha);

% ==== HIỂN THỊ CÁC PHƯƠNG PHÁP ====
figure;
imshow(gray); title("Ảnh gốc");
figure; imshow(heq_img); title("Histogram Equalization");
figure; imshow(clahe_img); title("CLAHE");
figure; imshow(unsharp_img); title("Unsharp Filtering");
figure; imshow(contrast_img); title("Tăng tương phản alpha");

%% ==== PART 4: Tăng cường vân - Phương pháp kết hợp tốt nhất ====

% 1. CLAHE
clahe_img = adapthisteq(gray);
gray_clahe = im2double(clahe_img);

% 2. Tách nền bằng Gaussian blur
background = imgaussfilt(gray_clahe, 15);
fringe_only = gray_clahe - background;
fringe_only = mat2gray(fringe_only);  % chuẩn hóa 0–1

% 3. Làm sắc nét
h_unsharp = fspecial('unsharp');
final_enhanced = imfilter(fringe_only, h_unsharp, 'replicate');

% 4. Kết quả cuối
enhanced = final_enhanced;

% ==== HIỂN THỊ ẢNH KẾT QUẢ CUỐI ====
figure;
imshow(enhanced, []);
title("Ảnh sau khi tăng cường vân (CLAHE + Tách nền + Unsharp)");

% ==== (Tùy chọn) Lưu ảnh kết quả ====
% imwrite(enhanced, 'anh_tang_cuong.png');


%% ==== PART 2: Fringe Extraction & 3D Surface Reconstruction ====

grayImg = heq_img;

%% Otsu thresholding (invert so fringe = 1, background = 0)
thresh = graythresh(grayImg);
BW_Original = ~imbinarize(grayImg, thresh);

% --- 2. Skeletonization (Thinning) ---
changing = 1;
[rows, columns] = size(BW_Original);
BW_Thinned = BW_Original;

while changing
    BW_Del = ones(rows, columns); 
    changing = 0;
    % Step 1
    for i=2:rows-1
        for j = 2:columns-1
            P = [BW_Thinned(i,j) BW_Thinned(i-1,j) BW_Thinned(i-1,j+1) BW_Thinned(i,j+1) BW_Thinned(i+1,j+1) ...
                 BW_Thinned(i+1,j) BW_Thinned(i+1,j-1) BW_Thinned(i,j-1) BW_Thinned(i-1,j-1) BW_Thinned(i-1,j)];
            if (BW_Thinned(i,j) == 1 && sum(P(2:end-1))<=6 && sum(P(2:end-1)) >=2 && ...
                    P(2)*P(4)*P(6)==0 && P(4)*P(6)*P(8)==0)
                A = 0;
                for k = 2:9
                    if P(k) == 0 && P(k+1)==1
                        A = A+1;
                    end
                end
                if (A==1)
                    BW_Del(i,j)=0;
                    changing = 1;
                end
            end
        end
    end
    BW_Thinned = BW_Thinned.*BW_Del;

    % Step 2 
    BW_Del = ones(rows, columns); 
    for i=2:rows-1
        for j = 2:columns-1
            P = [BW_Thinned(i,j) BW_Thinned(i-1,j) BW_Thinned(i-1,j+1) BW_Thinned(i,j+1) BW_Thinned(i+1,j+1) ...
                 BW_Thinned(i+1,j) BW_Thinned(i+1,j-1) BW_Thinned(i,j-1) BW_Thinned(i-1,j-1) BW_Thinned(i-1,j)];
            if (BW_Thinned(i,j) == 1 && sum(P(2:end-1))<=6 && sum(P(2:end-1)) >=2 && ...
                    P(2)*P(4)*P(8)==0 && P(2)*P(6)*P(8)==0)
                A = 0;
                for k = 2:9
                    if P(k) == 0 && P(k+1)==1
                        A = A+1;
                    end
                end
                if (A==1)
                    BW_Del(i,j)=0;
                    changing = 1;
                end
            end
        end
    end
    BW_Thinned = BW_Thinned.*BW_Del;
end

BW = BW_Thinned;
vung_chon = BW;

% --- 3. Estimate fringe orientation using Hough Transform ---
[H, theta, rho] = hough(vung_chon);
P = houghpeaks(H, 5);
lines = houghlines(vung_chon, theta, rho, P);

%% Compute average fringe angle
avg_angle = mean([lines.theta]);
goc_vuong_goc = avg_angle + 90; % Perpendicular direction to fringes

[H, W] = size(vung_chon);
x_center = W / 2;
y_center = H / 2;
slope = tand(goc_vuong_goc);
x1 = 1; x2 = W;
y1 = y_center + slope * (x1 - x_center);
y2 = y_center + slope * (x2 - x_center);

%% --- 4. Visualize skeletonized and rotated fringes ---
% Rotate binary fringe image so fringes are nearly vertical
angle_deg = -90 + goc_vuong_goc;   
BW_rotated = imrotate(BW, angle_deg, 'bilinear', 'crop');
BW_rotated = BW_rotated(50:end-50, 50:end-50); % Crop border
BW_rotated = BW; %nếu không xoay
figure('Name', 'Skeletonized and Rotated Fringes');
imshow(BW_rotated); 
title('Skeletonized Fringes (Rotated, Cropped)'); 
