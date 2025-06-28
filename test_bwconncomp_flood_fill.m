clc; clear; close all;

% Đọc ảnh nhị phân (đã tiền xử lý)
bw = imread('anh_sau_refined_cut.bmp');
if size(bw, 3) == 3
    bw = rgb2gray(bw) > 127;
end

% Cắt biên
offset = 25;
bw_crop = bw(offset:end-offset, offset:end-offset);
[H, W] = size(bw_crop);

% Phân vùng vân
cc = bwconncomp(bw_crop);
labeled_matrix = labelmatrix(cc);
stats = regionprops(cc, 'Centroid');

% Tìm group gần tâm
centroids = cat(1, stats.Centroid);
dist = vecnorm(centroids - [W/2, H/2], 2, 2);
[~, idx_center] = min(dist);

% Khởi tạo nhãn cho các nhóm
labels = nan(cc.NumObjects, 1);
labels(idx_center) = 0;

% Hàng đợi các group cần xử lý
queue = idx_center;

% Danh sách đã xét (để tránh lặp lại)
processed_groups = false(cc.NumObjects, 1);
processed_groups(idx_center) = true;

while ~isempty(queue)
    current_group = queue(1);
    queue(1) = [];

    current_label = labels(current_group);
    pixels = cc.PixelIdxList{current_group};
    [rows, cols] = ind2sub([H, W], pixels);

    visited_gid = [];  % để không xét lại cùng 1 gid trong cùng vòng

    % Duyệt tất cả pixel của group hiện tại
    for i = 1:length(rows)
        r = rows(i);
        c = cols(i);

        % Kiểm tra 4 hướng: lên, xuống, trái, phải
        neighbors = [r-1, c; r+1, c; r, c-1; r, c+1];
        for n = 1:size(neighbors, 1)
            nr = neighbors(n, 1);
            nc = neighbors(n, 2);

            % Kiểm tra trong vùng ma trận
            if nr >= 1 && nr <= H && nc >= 1 && nc <= W
                gid = labeled_matrix(nr, nc);
                if gid > 0 && ~processed_groups(gid) && ~ismember(gid, visited_gid)
                    labels(gid) = current_label + 1; % gán nhãn tăng dần
                    queue(end+1) = gid; % thêm group mới vào hàng đợi
                    processed_groups(gid) = true;
                    visited_gid(end+1) = gid;
                end
            end
        end
    end
end

% Hiển thị kết quả
figure; imshow(bw_crop); hold on;
for k = 1:cc.NumObjects
    if ~isnan(labels(k))
        c = stats(k).Centroid;
        text(c(1), c(2), num2str(labels(k)), ...
            'Color', 'yellow', 'FontSize', 12, 'FontWeight', 'bold');
    end
end
title('Gán nhãn vân bằng flood-fill lan truyền 4 hướng từ tâm');
hold off;
