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

    for i = 1:length(rows)
        r = rows(i); c = cols(i);

        % Gióng lên
        for y = r-1:-1:1
            gid = labeled_matrix(y, c);
            if gid > 0 && ~processed_groups(gid) && ~ismember(gid, visited_gid)
                labels(gid) = current_label + 1;
                queue(end+1) = gid;
                processed_groups(gid) = true;
                visited_gid(end+1) = gid;
                break;
            elseif gid > 0 && processed_groups(gid)
                break;
            end
        end

        % Gióng xuống
        for y = r+1:H
            gid = labeled_matrix(y, c);
            if gid > 0 && ~processed_groups(gid) && ~ismember(gid, visited_gid)
                labels(gid) = current_label - 1;
                queue(end+1) = gid;
                processed_groups(gid) = true;
                visited_gid(end+1) = gid;
                break;
            elseif gid > 0 && processed_groups(gid)
                break;
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
            'Color', 'yellow', 'FontSize', 9, 'FontWeight', 'bold');
    end
end
title('Gán nhãn vân bằng lan truyền theo cột từ tâm');
hold off;
