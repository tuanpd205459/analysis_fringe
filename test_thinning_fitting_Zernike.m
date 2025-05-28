clc; clear; close all;
addpath('./data')
Img_Original = imread('untitled2.png');

%% === 1. Chuyển ảnh về xám và nhị phân ===
grayImg = rgb2gray(Img_Original);
img = im2double(grayImg);
% figure; imshow(img); title('Ảnh xám');

% Nhị phân hóa bằng ngưỡng Otsu
thresh = graythresh(grayImg);
BW_Original = ~imbinarize(grayImg, thresh);  % Đảo màu: nền = 0, vân = 1

%% === 2. Thuật toán làm mảnh (Skeletonization) ===
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

%%
BW = BW_Thinned(50:end-50, 50:end-50);
vung_chon = BW;
%% === 4. Chọn 1 vùng để tính độ nghiêng ===
% figure;
% imshow(BW);                    
% title('Chọn vùng bằng hình chữ nhật');
% h = drawrectangle();           
% roi = h.Position;              
% x = round(roi(1));
% y = round(roi(2));
% w = round(roi(3));
% h_ = round(roi(4));
% 
% vung_chon = BW(y:y+h_-1, x:x+w-1);
% 
% figure;
% imshow(vung_chon);
% title('Vùng được chọn');

[H, theta, rho] = hough(vung_chon);
P = houghpeaks(H, 5);
lines = houghlines(vung_chon, theta, rho, P);

avg_angle = mean([lines.theta]);
goc_vuong_goc = avg_angle + 90;

[H, W] = size(vung_chon);
x_center = W / 2;
y_center = H / 2;

slope = tand(goc_vuong_goc);
x1 = 1;
x2 = W;
y1 = y_center + slope * (x1 - x_center);
y2 = y_center + slope * (x2 - x_center);

% figure;
% imshow(BW); hold on;
% plot([x1 x2], [y1 y2], 'r-', 'LineWidth', 2);
% title(['Đường vuông góc với vân (' num2str(goc_vuong_goc) '\xB0)']);
% h = drawline('Color','g','LineWidth',1);
% wait(h);

%% === 5. Hiển thị số thứ tự vân ===
figure; imshow(BW); hold on;

% B1: Xoay ảnh nhị phân để vân gần thẳng đứng
angle_deg = -90 + goc_vuong_goc;   % Ví dụ: nếu vân nghiêng 20 độ so với trục ngang
BW_rotated = imrotate(BW, angle_deg, 'bilinear', 'crop');

% Cắt biên để tránh viền đen sau khi xoay
BW_rotated = BW_rotated(50:end-50, 50:end-50);

figure; imshow(BW_rotated); title('Ảnh sau khi xoay'); hold on;

% Tìm biên các vùng sáng (vân) trong ảnh nhị phân đã làm mỏng
[B,L] = bwboundaries(BW_rotated, 'noholes');

% Labeled cũng có thể dùng để đánh nhãn vùng sáng (không bắt buộc trong đoạn này)
labeled = bwlabel(BW_rotated);

imshow(BW_rotated); hold on;
title('Đánh số từng vân');

for k = 1:length(B)
    boundary = B{k};  % mảng các điểm biên vân thứ k
    
    % Tính tọa độ trung bình (centroid x, y) của biên
    c = mean(boundary(:,2));  % trung bình cột = x
    r = mean(boundary(:,1));  % trung bình hàng = y

    % Hiển thị số thứ tự tại vị trí centroid
    text(c, r, num2str(k), 'Color','yellow','FontSize',12,'FontWeight','bold');
end



%% === 3. Tái tạo bề mặt 3D từ vân ===
lambda = 632.8e-9; % bước sóng ánh sáng
khoang_cach_van = (lambda / 2)/cos(abs(avg_angle));

BW = BW_rotated;
L = bwlabel(BW);
num_labels = max(L(:));

X = []; Y = []; Z = [];

for i = 1:num_labels
    [y, x] = find(L == i); 
    z = ones(size(x)) * (i-1) * khoang_cach_van;
    X = [X; x];
    Y = [Y; y];
    Z = [Z; z];
end

[xq, yq] = meshgrid(1:size(BW,2), 1:size(BW,1));
F = scatteredInterpolant(X, Y, Z, 'natural', 'nearest');
Zq = F(xq, yq);
Zq(~isfinite(Zq)) = 0;

figure;
surf(xq, yq, Zq, 'EdgeColor', 'none');
colormap turbo;
colorbar;
xlabel('X (px)'); ylabel('Y (px)'); zlabel('Chiều cao (m)');
title('Tái tạo bề mặt từ các đường vân');
view(3);
