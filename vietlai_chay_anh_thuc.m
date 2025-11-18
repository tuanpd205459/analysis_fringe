clc; clear; close all;

%% load ảnh thực

addpath("C:\Users\admin\Máy tính\Lab thầy Tùng\My_TN\data 4 11 2025\sample o 6 MO60x");
hologram = imread("image_2025-11-04T18-43-20.6.bmp");
if size(hologram, 3) == 3
    hologram = rgb2gray(hologram);
    fprintf('Đã chuyển đổi ảnh RGB sang grayscale\n');
end

hologram = rot90(hologram);   % Xoay 180 độ (mỗi lần rot90 xoay 90 độ)

%% 5. Noise removal

hologram = imgaussfilt(hologram, 1);
% hologram = medfilt2(hologram, [3 3]);
% hologram = wiener2(hologram, [5 5]);

% figure;
% imshow(hologram);
% colorbar;
% title('hologram sau noise removal : ');

%% 6. Histogram equalization

hologram = adapthisteq(hologram);
figure;
imshow(hologram);
colorbar;
title('hologram sau noise adapthisteq : ');
%%
sensitive_coef = 0.6;
nei = 31;
% % Tính ngưỡng cục bộ
% T = adaptthresh(hologram, sensitive_coef, ...
%     'NeighborhoodSize', [nei nei], ...    % kích thước vùng cục bộ
%     'Statistic', 'gaussian');           % hoặc 'mean'
% 
% % Nhị phân hóa ảnh
% hologram_bin = imbinarize(hologram, T);
% 
% figure;
% imshow(hologram_bin);
% title('Adaptive threshold theo từng vùng gaussian');
% 
% % Tính ngưỡng cục bộ
% T = adaptthresh(hologram, sensitive_coef, ...
%     'NeighborhoodSize', [nei nei], ...    % kích thước vùng cục bộ
%     'Statistic', 'mean');           % hoặc 'mean'
% 
% % Nhị phân hóa ảnh
% hologram_bin = imbinarize(hologram, T);
% 
% figure;
% imshow(hologram_bin);
% title('Adaptive threshold theo từng vùng mean');

T = adaptthresh(hologram, sensitive_coef, ...
    'NeighborhoodSize', [nei nei], ...    % kích thước vùng cục bộ
    'Statistic', 'median');           % hoặc 'mean'

% Nhị phân hóa ảnh
hologram_bin = imbinarize(hologram, T);

figure;
imshow(hologram_bin);
title('Adaptive threshold theo từng vùng median');

%%
figure;
imshow(hologram_bin);
title('Cắt ngưỡng ảnh tốt');

% --- 3. Gọi hàm myDrawRec() ---
[pos, xRec, yRec, widthRec, heightRec] = myDrawRec();

% --- 4. Cắt vùng ảnh đã chọn ---
hologram_bin_crop = hologram_bin(yRec : yRec + heightRec - 1, ...
                         xRec : xRec + widthRec - 1);
hologram_bin = hologram_bin_crop;

figure();
imshow(hologram_bin, []);
title('Sau khi cat');

%%
% --- Bước 2: Skeletonize bằng Zhang-Suen ---
    fprintf('Bước 2/3: Áp dụng thuật toán Zhang-Suen...\n');
BW_Original = hologram_bin;
BW_Thinned = BW_Original;
[rows, cols] = size(BW_Thinned);
changing = true;
iteration = 0;

while changing
    iteration = iteration + 1;
    changing = false;
    BW_Del = true(rows, cols);

    % --- Step 1 của Zhang-Suen ---
    for i = 2:rows-1
        for j = 2:cols-1
            P = BW_Thinned(i-1:i+1, j-1:j+1);
            P = P(:)';
            % Sắp xếp theo thứ tự: P1(center), P2, P3, P4, P5, P6, P7, P8, P9, P2(lặp)
            P = [P(5), P(2), P(3), P(6), P(9), P(8), P(7), P(4), P(1), P(2)];

            if P(1) == 1  % Nếu pixel trung tâm là foreground
                neighbors = sum(P(2:9));  % Số lượng neighbor foreground
                transitions = sum(P(2:9) == 0 & P(3:10) == 1);  % Số transition 0->1

                % Điều kiện Zhang-Suen Step 1
                if neighbors >= 2 && neighbors <= 6 && transitions == 1 ...
                        && P(2)*P(4)*P(6) == 0 && P(4)*P(6)*P(8) == 0
                    BW_Del(i,j) = false;
                    changing = true;
                end
            end
        end
    end
    BW_Thinned = BW_Thinned & BW_Del;

    % --- Step 2 của Zhang-Suen ---
    BW_Del = true(rows, cols);
    for i = 2:rows-1
        for j = 2:cols-1
            P = BW_Thinned(i-1:i+1, j-1:j+1);
            P = P(:)';
            P = [P(5), P(2), P(3), P(6), P(9), P(8), P(7), P(4), P(1), P(2)];

            if P(1) == 1
                neighbors = sum(P(2:9));
                transitions = sum(P(2:9) == 0 & P(3:10) == 1);

                % Điều kiện Zhang-Suen Step 2
                if neighbors >= 2 && neighbors <= 6 && transitions == 1 ...
                        && P(2)*P(4)*P(8) == 0 && P(2)*P(6)*P(8) == 0
                    BW_Del(i,j) = false;
                    changing = true;
                end
            end
        end
    end
    BW_Thinned = BW_Thinned & BW_Del;

    % Tránh vòng lặp vô hạn
    if iteration > 1000
        warning('Đã đạt giới hạn iteration (1000). Dừng thuật toán.');
        break;
    end
end
%%
% --- Trả về kết quả ---
skeleton_image = BW_Thinned;
binary_image = BW_Original;

skeleton = skeleton_image;
BW = skeleton;

%
S = MZS_thinning(BW);
BW = S;

% figure;
% imshow(hologram_bin); hold on;
% imshow(BW);  title('Skeleton (MZS)');

figure;
imshowpair(BW, hologram_bin, 'falsecolor');   % Hiển thị 2 ảnh bằng 2 kênh màu khác nhau
title('Overlay bằng falsecolor');

%% loại bỏ các điểm rời rạc có kích thước nhỏ:
BW = bwmorph(BW,"spur", 10);

BW = bwmorph(BW, "clean");
figure;
imshow(BW);  title('Skeleton (MZS)- clean isolated point');

%%
CC = bwconncomp(BW, 8);   % tìm vùng liên thông theo 8-láng giềng
L = labelmatrix(CC);      % chuyển thành ma trận nhãn
RGB = label2rgb(L, 'jet', 'k', 'shuffle'); % hiển thị mỗi vùng màu khác nhau

imshow(RGB);
title('Các vùng liên thông được tô màu');

%%
% %% tim cac diem junction
branchpoints1 = bwmorph(BW,"branchpoints");
[y,x] = find(branchpoints1);

[BW, junctionMap] = removeJunctions(BW);
% BW = bwmorph(BW,"spur", 5);

figure; imshow(BW); hold on;
[row, col] = find(junctionMap);
plot(col, row, 'go', 'MarkerSize',10,'LineWidth',1);
title('Skeleton sau khi xoá junction');
hold on;
plot(x, y, 'r+', 'MarkerSize', 8, 'LineWidth', 1.5);
title('Các điểm branchpoints ban dau');
hold off;

%%
% % % Xóa các đoạn ngắn (bridge)
maxBridgeLen = 5;
BW = bwareaopen(BW, maxBridgeLen);

BW = removeSmallComponents(BW, 3);  % xoá vùng liên thông < 10 pixel
figure; imshow(BW);
title("Sau khi xoa vungf lien thong nho hon 11 pixel");

%% noois van
 % tìm vùng liên thông theo 8-láng giềng
CC = bwconncomp(BW, 8);
endPoints = bwmorph(BW, 'endpoints');
endPoints(1,:) = 0;
endPoints(end,:) = 0;
endPoints(:,1) = 0;
endPoints(:,end) = 0;

% endPoints: ảnh nhị phân (1 tại các endpoint)
[y, x] = find(endPoints);   % Tọa độ các endpoint
[h, w] = size(BW);   % Kích thước ảnh

% Bán kính nửa vùng (4x4)
r = 4;  % vì 4x4 thực ra mở rộng 2 pixel theo mỗi hướng
% Lưu danh sách các đường đã nối
allLines = {};  
count = 0;
for k = 1:length(x)
    % Tọa độ điểm hiện tại
    cx = x(k);
    cy = y(k);

    % Giới hạn vùng cắt (đảm bảo không vượt biên)
    x1 = max(cx - r, 1);
    x2 = min(cx + r, w);
    y1 = max(cy - r, 1);
    y2 = min(cy + r, h);

    % Cắt vùng 4x4 quanh điểm hiện tại
    localPatch = endPoints(y1:y2, x1:x2);

    % Nếu trong vùng có >1 endpoint => có endpoint khác
    numPoints = sum(localPatch(:));
    if numPoints > 1
        % Tìm toạ độ tương đối của endpoint khác
        [yy, xx] = find(localPatch);
        % Loại bỏ chính nó
        relIdx = find(~(xx == (cx - x1 + 1) & yy == (cy - y1 + 1)));
        if ~isempty(relIdx)
            % Tọa độ tuyệt đối của endpoint khác
            x_other = x1 + xx(relIdx(1)) - 1;
            y_other = y1 + yy(relIdx(1)) - 1;

            % Nối 2 điểm (cx,cy) và (x_other,y_other)
            [endPoints, linePix] = drawLine(endPoints, cx, cy, x_other, y_other);

            % Lưu lại line để hiển thị sau
            count = count + 1;
            allLines{count} = linePix;
        end
    end
end

% --- HIỂN THỊ TẤT CẢ ĐƯỜNG NỐI ---
figure; 
imshow(BW, []); hold on;
title('Các đường nối giữa các endpoint');

% Hiển thị từng đường nối
for i = 1:numel(allLines)
    lp = allLines{i};
    plot(lp(:,1), lp(:,2), 'r-', 'LineWidth', 1.5);
end

% % Hiển thị điểm endpoint gốc
% plot(x, y, 'go', 'MarkerSize', 4, 'MarkerFaceColor', 'g');
hold off;

% % Tham số nối
% minCompSize = 15;
% maxDist     = 10;   
% vecAlignThr = cosd(10);    % ~0.866
% vectors = fitEndpointVectors(BW, endPoints, 15);
% max_perh = 5;











%% cac ham phụ trợ

function [pos, xRec, yRec, widthRec, heightRec] = myDrawRec()
% Cho phép người dùng vẽ một hình chữ nhật (ROI) trên ảnh hiện tại.
hFig = gcf;
hROI = drawrectangle();
centerRec = [hROI.Position(1) + hROI.Position(3)/2, hROI.Position(2) + hROI.Position(4)/2];
hold on;
hMarker = plot(centerRec(1), centerRec(2), 'r+', 'MarkerSize', 10, 'LineWidth', 2);
hold off;
addlistener(hROI, 'MovingROI', @(src, evt) updateCenterRectangle(src, hMarker));

% Đợi người dùng double-click để xác nhận
wait(hROI);

pos = round(hROI.Position);
xRec = pos(1); yRec = pos(2);
widthRec = pos(3); heightRec = pos(4);

% Đóng cửa sổ sau khi đã chọn xong
if ishandle(hFig)
    close(hFig);
end
end
% -------------------------------------------------------------------------
function updateCenterRectangle(roi, centerMarker)
% Cập nhật vị trí dấu cộng ở tâm ROI khi đang di chuyển.
centerMarker.XData = roi.Position(1) + roi.Position(3)/2;
centerMarker.YData = roi.Position(2) + roi.Position(4)/2;
drawnow;
end
%%
function S = MZS_thinning(BW)
    S = BW > 0;
    prev = false(size(S));
    while true
        % Sub-iteration 1: even pixels
        marker = MZS_iteration(S, 0);
        S(marker) = 0;

        % Sub-iteration 2: odd pixels
        marker = MZS_iteration(S, 1);
        S(marker) = 0;

        if isequal(S, prev), break; end
        prev = S;
    end
end
function marker = MZS_iteration(S, parity)
    % Pad để tránh lỗi biên
    P = padarray(S, [1 1], 0, 'both');

    % Lấy lân cận (theo thứ tự ZS)
    P2 = P(1:end-2,2:end-1); % north
    P3 = P(1:end-2,3:end);   % northeast
    P4 = P(2:end-1,3:end);   % east
    P5 = P(3:end,3:end);     % southeast
    P6 = P(3:end,2:end-1);   % south
    P7 = P(3:end,1:end-2);   % southwest
    P8 = P(2:end-1,1:end-2); % west
    P9 = P(1:end-2,1:end-2); % northwest
    P1 = P(2:end-1,2:end-1); % center

    % Tổng số hàng xóm (B)
    B = P2+P3+P4+P5+P6+P7+P8+P9;

    % C(p1) theo công thức trong paper
    C = (~P2 & (P3|P4)) + (~P4 & (P5|P6)) + ...
        (~P6 & (P7|P8)) + (~P8 & (P9|P2));

    % Điều kiện chung
    cond = (C==1);

    if parity == 0   % subfield chẵn
        cond = cond & (mod(bsxfun(@plus,(1:size(S,1))',(1:size(S,2))),2)==0);
        cond = cond & (B>=2 & B<=7);
        cond = cond & (~(P2 & P4 & P6));
        cond = cond & (~(P4 & P6 & P8));
    else             % subfield lẻ
        cond = cond & (mod(bsxfun(@plus,(1:size(S,1))',(1:size(S,2))),2)==1);
        cond = cond & (B>=1 & B<=7);
        cond = cond & (~(P2 & P4 & P8));
        cond = cond & (~(P2 & P6 & P8));

        % Bổ sung điều kiện giữ pixel để bảo toàn 2x2 / diagonal
        diagNeighbors = (P3|P5|P7|P9);
        cond = cond & ~( (B==1) & diagNeighbors );
    end

    marker = P1 & cond;
end
function BW_clean = removeSmallComponents(BW, minSize)
% Xoá các vùng liên thông có kích thước < minSize pixel
%
% Input:
%   BW      - ảnh nhị phân (0/1)
%   minSize - ngưỡng số pixel nhỏ nhất giữ lại
%
% Output:
%   BW_clean - ảnh nhị phân sau khi lọc

% Tìm vùng liên thông (8-neighbors)
CC = bwconncomp(BW, 8);
% Đếm số pixel trong từng vùng
numPixels = cellfun(@numel, CC.PixelIdxList);

% Giữ lại vùng đủ lớn
BW_clean = BW;
for i = 1:CC.NumObjects
    if numPixels(i) < minSize
        BW_clean(CC.PixelIdxList{i}) = 0; % xoá vùng nhỏ
    end
end
%     BW_clean = bwmorph(BW_clean, 'spur', 1);  % loại bỏ các nhánh nhỏ lẻ

end
function [BW_clean, junction] = removeJunctions(BW)
% REMOVEJUNCTIONS - Phát hiện và loại bỏ junction pixels trong skeleton
%
% Syntax:
%   [BW_clean, junction] = removeJunctions(BW)
%
% Input:
%   BW - ảnh nhị phân skeleton
%
% Output:
%   BW_clean - skeleton sau khi xoá junction
%   junction - ảnh nhị phân, 1 tại vị trí junction pixel
%
% Đặc điểm:
%   - Junction pixel: có >=3 hàng xóm trong 8 hướng
%   - Bỏ qua các pixel sát biên (4 pixel)

% --- 1. Đếm số hàng xóm ---
kernel = ones(3,3); kernel(2,2) = 0;  % 8-neighborhood
neighborCount = conv2(double(BW), kernel, 'same');

% --- 2. Junction: skeleton pixel có >= 3 hàng xóm ---
junction = (BW == 1) & (neighborCount >= 3);

% --- 3. Không xét biên (4 pixel) ---
junction(1,:)       = 0;
junction(end,:) = 0;
junction(:,1)       = 0;
junction(:,end) = 0;

% --- 4. Xóa junction ---
BW_clean = BW;
BW_clean(junction) = 0;
end

function [BW, linePix] = drawLine(BW, x1, y1, x2, y2)
% Vẽ line nối từ (x1,y1) đến (x2,y2) bằng thuật toán Bresenham
[h, w] = size(BW);
[lineX, lineY] = bresenham(x1, y1, x2, y2);

linePix = [lineX(:), lineY(:)];

for k = 1:length(lineX)
    cx = lineX(k);
    cy = lineY(k);
    if cx >= 1 && cx <= w && cy >= 1 && cy <= h
        BW(cy, cx) = 1;
    end
end

end

function [x, y] = bresenham(x1, y1, x2, y2)

% Thuật toán Bresenham
x1 = round(x1); y1 = round(y1);
x2 = round(x2); y2 = round(y2);

dx = abs(x2 - x1);
dy = abs(y2 - y1);

sx = sign(x2 - x1);
sy = sign(y2 - y1);

err = dx - dy;

x = []; y = [];
while true
    x(end+1) = x1;
    y(end+1) = y1;
    if x1 == x2 && y1 == y2
        break;
    end
    e2 = 2 * err;
    if e2 > -dy
        err = err - dy;
        x1 = x1 + sx;
    end
    if e2 < dx
        err = err + dx;
        y1 = y1 + sy;
    end
end
end
