clc,clear;close all;
% Đọc ảnh skeleton nhị phân
addpath("data\");
bw = imread('test_thuat_toan_noi_van.bmp');
% bw = imbinarize(I);

% Gán nhãn các đoạn vân riêng biệt
CC = bwconncomp(bw);
labeled = labelmatrix(CC);
numSegments = CC.NumObjects;

% Tính centroid từng đoạn
props = regionprops(labeled, 'Centroid');
centroids = cat(1, props.Centroid);

% Tính hướng vân (sử dụng PCA với pixel mỗi đoạn)
orientations = zeros(numSegments, 1);
for i = 1:numSegments
    coords = CC.PixelIdxList{i};
    [r, c] = ind2sub(size(bw), coords);
    X = [c, r];
    if size(X,1) > 2
        C = cov(X);
        [V, ~] = eig(C);
        dir = V(:,1);
        orientations(i) = atan2(dir(2), dir(1)); % hướng chính
    else
        orientations(i) = NaN;
    end
end

% Xây đồ thị kết nối các đoạn vân gần nhau
max_dist = 30;     % ngưỡng khoảng cách để xét nối
max_angle = pi/8;  % ngưỡng sai lệch hướng

G = graph();
G = addnode(G, numSegments);

for i = 1:numSegments
    for j = i+1:numSegments
        d = norm(centroids(i,:) - centroids(j,:));
        delta_theta = abs(orientations(i) - orientations(j));
        if d < max_dist && delta_theta < max_angle
            G = addedge(G, i, j, d);  % thêm cạnh có trọng số là khoảng cách
        end
    end
end

% Gán bậc vân từ đoạn gốc (ví dụ chọn đoạn gần tâm ảnh)
img_center = size(bw) / 2;
[~, root_idx] = min(vecnorm(centroids - img_center, 2, 2));

% Dijkstra để tính đường ngắn nhất từ đoạn gốc đến các đoạn khác
[~, D] = shortestpathtree(G, root_idx);

% Gán bậc: mỗi khoảng cách ≈ 1 bậc vân (suy từ spacing thực nghiệm)
fringe_spacing_px = 10; % giả định
fringe_order = round(D / fringe_spacing_px);

% Vẽ ảnh kết quả
order_map = zeros(size(bw));
for i = 1:numSegments
    order_map(CC.PixelIdxList{i}) = fringe_order(i);
end

figure;
imagesc(order_map);
axis image off;
colormap(jet);
colorbar;
title('Gán bậc vân bằng đồ thị (Graph-based fringe labeling)');
