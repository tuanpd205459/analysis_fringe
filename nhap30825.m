clc, clear, close all;
%%
load("BW.mat"); % load anh binary

%% Xoa vung nho le
BW = removeSmallComponents(BW, 5);  % xoá vùng liên thông < 10 pixel

%% Xoá junction
[BW, junctionMap] = removeJunctions(BW);
BW = bwmorph(BW,"spur",2);

% figure; imshow(BW); hold on;
% [row, col] = find(junctionMap);
% plot(col, row, 'go', 'MarkerSize',10,'LineWidth',1);
% title('Skeleton sau khi xoá junction');
%
% % Xóa các đoạn ngắn (bridge)
maxBridgeLen = 8;
BW = bwareaopen(BW, maxBridgeLen);
% figure; imshow(BW); hold on;
% [row, col] = find(junctionMap);
% plot(col, row, 'go', 'MarkerSize',10,'LineWidth',1);
% title('Skeleton sau khi sau khi xoá nối');

%% --- 4. Nối endpoint theo vòng lặp thử ---
nLoop = 1; % số vòng nối
for count = 1:nLoop
    CC = bwconncomp(BW, 8);
    endPoints = bwmorph(BW, 'endpoints');
    endPoints(1,:) = 0;
    endPoints(end,:) = 0;
    endPoints(:,1) = 0;
    endPoints(:,end-1) = 0;
    if count == 1
        fprintf('--> Vòng nối %d\n', count);

        % Tham số nối thử
        minCompSize = 15;
        maxDist     = 10;    % vòng sau cho phép nối xa hơn
        vecAlignThr = cosd(10);    % ~0.866
        vectors = fitEndpointVectors(BW, endPoints, 35);

    end
    if  count == 2
        fprintf('--> Vòng nối %d ----\n', count);

        % Tham số nối thử
        minCompSize = 12;
        maxDist     = 20;    % vòng sau cho phép nối xa hơn
        vecAlignThr = cosd(15);    % ~0.866

        vectors = fitEndpointVectors(BW, endPoints, 35);

    end
    if  count == 3
        fprintf('--> Vòng nối %d ----\n', count);

        % Tham số nối thử
        minCompSize = 8;
        maxDist     = 25;    % vòng sau cho phép nối xa hơn
        vecAlignThr = cosd(15);    % ~0.866

        vectors = fitEndpointVectors(BW, endPoints, 35);

    end

    if  count == 4
        fprintf('--> Vòng nối %d ----\n', count);

        % Tham số nối thử
        minCompSize = 5;
        maxDist     = 30;    % vòng sau cho phép nối xa hơn
        vecAlignThr = cosd(20);    % ~0.866

        vectors = fitEndpointVectors(BW, endPoints, 10);

    end
    if  count == 5
        fprintf('--> Vòng nối %d ----\n', count);

        % Tham số nối thử
        minCompSize = 5;
        maxDist     = 35;    % vòng sau cho phép nối xa hơn
        vecAlignThr = cosd(30);    % ~0.866

        vectors = fitEndpointVectors(BW, endPoints, 10);
        [BW, linesConnected] = connectEndpoints_v3(BW, vectors, CC, minCompSize, maxDist, vecAlignThr, 5);

    end

    if  count == 6
        fprintf('--> Vòng nối %d ----\n', count);

        % Tham số nối thử
        minCompSize = 5;
        maxDist     = 40;    % vòng sau cho phép nối xa hơn
        vecAlignThr = cosd(35);    % ~0.866

        vectors = fitEndpointVectors(BW, endPoints, 10);
    end

    if  count == 7
        fprintf('--> Vòng nối %d ----\n', count);

        % Tham số nối thử
        minCompSize = 5;
        maxDist     = 50;    % vòng sau cho phép nối xa hơn
        vecAlignThr = cosd(50);    % ~0.866

        vectors = fitEndpointVectors(BW, endPoints, 10);
        [BW, linesConnected] = connectEndpoints_v3(BW, vectors, CC, minCompSize, maxDist, vecAlignThr, 10);

    end
        if  count == 8
        fprintf('--> Vòng nối %d ----\n', count);

        % Tham số nối thử
        minCompSize = 5;
        maxDist     = 100;    % vòng sau cho phép nối xa hơn
        vecAlignThr = cosd(60);    % ~0.866

        vectors = fitEndpointVectors(BW, endPoints, 10);
        [BW, linesConnected] = connectEndpoints_v3(BW, vectors, CC, minCompSize, maxDist, vecAlignThr, 20);

    end
    [BW, linesConnected] = connectEndpoints_v3(BW, vectors, CC, minCompSize, maxDist, vecAlignThr, 5);

    % Hiển thị skeleton sau vòng nối
    figure;
    imshow(BW); hold on;
    if ~isempty(endPoints)
        plot(endPoints(:,1), endPoints(:,2), 'ro', 'MarkerSize',8,'LineWidth',2);
    end
    title(sprintf('Skeleton sau vòng nối %d', count));


    % Hiển thị các đường nối và vector tại endpoint
    if ~isempty(linesConnected)
        figure;
        imshow(BW); hold on;

        % --- Vẽ các đường nối ---
        for k = 1:numel(linesConnected)
            lineXY = linesConnected{k};  % [x y] pixel trên đoạn nối
            plot(lineXY(:,1), lineXY(:,2), 'g-', 'LineWidth', 2);
        end

        % --- Vẽ endpoint và vector hướng ---
        for i = 1:size(vectors,1)
            cx = vectors(i,1);
            cy = vectors(i,2);
            vx = vectors(i,3);
            vy = vectors(i,4);

            % Vẽ điểm endpoint
            plot(cx, cy, 'ro', 'MarkerFaceColor','r', 'MarkerSize',5);

            % Vẽ vector hướng tại endpoint
            quiver(cx, cy, 10*vx, 10*vy, 'r', 'LineWidth',2, 'MaxHeadSize',2);
        end

        % --- Tiêu đề ---
        title(sprintf('Đường nối vòng %d (%d đường)', count, numel(linesConnected)));
    end

end
BW_NEW = BW;
% save("BW_NEW.mat","BW_NEW");
%%
% --- Tìm endpoint ---
endPoints = bwmorph(BW,'endpoints');
[yEP,xEP] = find(endPoints);

% --- (Ví dụ) Tạo vector hướng tại endpoint: dùng PCA 15 px lân cận
Nfit = 15;
vectors = fitEndpointVectors_simple(BW,endPoints,Nfit);

% --- Tham số nối ---
margin = 20;          % chỉ xét endpoint cách biên <= 10 px
stopOnSkeleton = true; % dừng nếu chạm skeleton hiện có
thickness = 1;         % bề dày nét nối

[BW_new, linesOut] = extendEndpointsToBorder(BW,endPoints,vectors,margin,stopOnSkeleton,thickness);

figure; imshow(BW); title('Skeleton gốc');
figure; imshow(BW_new); title('Sau khi nối ra biên');

% Vẽ các line đã nối (màu khác) để kiểm tra trực quan
hold on;
for k = 1:numel(linesOut)
    if ~isempty(linesOut{k})
        plot(linesOut{k}(:,1), linesOut{k}(:,2), 'LineWidth', 1.5);
    end
end
hold off;

figure;
imshow(BW); 
title("Anh finalll");


%% Ham
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

function comp_id = findComponent(CC, p)
% p = [row, col]
comp_id = 0;
idx = sub2ind(CC.ImageSize, p(1), p(2));
for c = 1:CC.NumObjects
    if ismember(idx, CC.PixelIdxList{c})
        comp_id = c;
        return;
    end
end
end



%%
function vectors = fitEndpointVectors(BW, endPoints, Nfit)
% fitEndpointVectors - Tính vector hướng tại endpoint của skeleton
%
% Cú pháp:
%   vectors = fitEndpointVectors(BW, endPoints, Nfit)
%
% Input:
%   BW        - ảnh nhị phân skeleton
%   endPoints - ảnh nhị phân endpoint (1 tại endpoint)
%   Nfit      - số pixel dùng để fit PCA (ví dụ: 30)
%
% Output:
%   vectors - ma trận [N x 4], mỗi hàng:
%             [cx cy vx vy]
%             (cx, cy) = tọa độ endpoint
%             (vx, vy) = vector đơn vị hướng ra ngoài

[y_idx, x_idx] = find(endPoints);  % tọa độ endpoints
CC = bwconncomp(BW, 8);           % tìm các component
vectors = [];

for k = 1:length(x_idx)
    cx = x_idx(k);
    cy = y_idx(k);

    % Kiểm tra endpoint thuộc component nào
    comp_id = 0;
    for c = 1:CC.NumObjects
        if ismember(sub2ind(size(BW), cy, cx), CC.PixelIdxList{c})
            comp_id = c;
            break;
        end
    end

    if comp_id == 0, continue; end  % endpoint không thuộc component nào

    % Lấy tọa độ tất cả pixel trong component
    [yy, xx] = ind2sub(size(BW), CC.PixelIdxList{comp_id});

    % Tính khoảng cách từ endpoint
    dist2 = (xx - cx).^2 + (yy - cy).^2;
    [~, idx] = sort(dist2);
    idxN = idx(1:min(Nfit, numel(idx)));

    X = xx(idxN);
    Y = yy(idxN);

    if numel(X) > 1
        % --- Fit hướng bằng PCA ---
        Xc = X - mean(X);
        Yc = Y - mean(Y);
        D = [Xc(:) Yc(:)];
        [~,~,V] = svd(D,'econ');
        v = V(:,1);  % vector chính (cột đầu tiên)
        v = v / norm(v);

        % --- Xác định hướng "ra ngoài" ---
        centroid = [mean(X); mean(Y)];
        c = centroid - [cx; cy];  % vector từ endpoint vào trong component
        if dot(v, c) > 0
            v = -v; % đảo dấu để hướng ra ngoài
        end
    else
        v = [0;0];
    end

    vectors = [vectors; cx cy v(1) v(2)];
end
end
function [BW_new, linesConnected] = connectEndpoints_fast(BW, vectors, CC, minCompSize, maxDist, vecAlignThr)
% CONNECTENDPOINTS_FAST - Nối các endpoint skeleton từ điểm nằm sâu 3 pixel
% trong cùng component.
%
% INPUT:
%   BW : ảnh nhị phân skeleton
%   vectors : [cx cy vx vy] từ hàm computeEndpointVectors (dùng để check hướng)
%   CC : bwconncomp(BW,8)
%   minCompSize : kích thước tối thiểu của component để nối
%   maxDist : khoảng cách tối đa giữa 2 endpoint
%   vecAlignThr : ngưỡng cosine giữa vector hướng và đường nối
%
% OUTPUT:
%   BW_new : skeleton sau khi nối
%   linesConnected: cell lưu danh sách pixel của các đoạn nối

BW_new = BW;
linesConnected = {};
used = false(size(vectors,1),1);
N = size(vectors,1);

for i = 1:N-1
    if used(i), continue; end

    cx1 = vectors(i,1); cy1 = vectors(i,2);
    v1 = [vectors(i,3), vectors(i,4)];
    comp_id1 = findComponent(CC, [cy1,cx1]);
    if comp_id1==0 || numel(CC.PixelIdxList{comp_id1}) < minCompSize
        continue;
    end

    for j = i+1:N
        if used(j), continue; end

        cx2 = vectors(j,1); cy2 = vectors(j,2);
        v2 = [vectors(j,3), vectors(j,4)];
        comp_id2 = findComponent(CC, [cy2,cx2]);
        if comp_id2==0 || numel(CC.PixelIdxList{comp_id2}) < minCompSize
            continue;
        end

        % chỉ nối nếu cùng component
        if comp_id1 ~= comp_id2
            continue;
        end

        % khoảng cách giữa endpoint
        dist = hypot(cx1-cx2, cy1-cy2);
        if dist > maxDist, continue; end

        % kiểm tra vector hướng
        dir12 = [cx2-cx1, cy2-cy1];
        dir12 = dir12 / (norm(dir12)+eps);
        cond1 = dot(v1, dir12) > vecAlignThr;
        cond2 = dot(v2, -dir12) > vecAlignThr;
        if ~(cond1 && cond2), continue; end

        % đi vào skeleton 3 pixel
        connect_p1 = walkInSkeleton(BW, [cx1,cy1], 3);
        connect_p2 = walkInSkeleton(BW, [cx2,cy2], 3);
        if isempty(connect_p1) || isempty(connect_p2)
            continue;
        end

        % nối từ điểm cách endpoint 3 pixel
        [BW_new, linePixels] = drawLine(BW_new, connect_p1(1), connect_p1(2), ...
            connect_p2(1), connect_p2(2));
        linesConnected{end+1} = linePixels; %#ok<AGROW>

        used(i) = true;
        used(j) = true;
        break;
    end
end
end

function pt = walkInSkeleton(BW, start_pt, nStep)
% WALKINSKELETON - đi từ endpoint vào trong skeleton nStep pixel
% start_pt : [x y] endpoint (theo hệ toạ độ cột-x, hàng-y)
% pt : [x y] sau nStep, [] nếu không đi đủ bước

x = start_pt(1);
y = start_pt(2);
prev = start_pt;

for k = 1:nStep
    neigh = [];
    for dx=-1:1
        for dy=-1:1
            if dx==0 && dy==0, continue; end
            nx = x+dx; ny = y+dy;
            if nx<1 || ny<1 || nx>size(BW,2) || ny>size(BW,1)
                continue;
            end
            if BW(ny,nx)==1
                neigh(end+1,:) = [nx,ny]; %#ok<AGROW>
            end
        end
    end

    % bỏ pixel vừa đi
    neigh = neigh(~(neigh(:,1)==prev(1) & neigh(:,2)==prev(2)), :);
    if isempty(neigh)
        pt = [];
        return;
    end

    % chọn pixel kế tiếp (thường 1 neighbor, trừ junction)
    next = neigh(1,:);
    prev = [x,y];
    x = next(1); y = next(2);
end

pt = [x,y];
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
junction(1:2,:)       = 0;
junction(end-1:end,:) = 0;
junction(:,1:2)       = 0;
junction(:,end-1:end) = 0;

% --- 4. Xóa junction ---
BW_clean = BW;
BW_clean(junction) = 0;
end
function [BW_new, linesConnected] = connectEndpoints(BW, vectors, CC, minCompSize, maxDist, vecAlignThr)
% BW           : skeleton binary
% vectors      : [cx cy vx vy] từ hàm computeEndpointVectors
% CC           : bwconncomp(BW,8)
% minCompSize  : kích thước tối thiểu của vân
% maxDist      : khoảng cách tối đa cho phép nối
% vecAlignThr  : ngưỡng cos(angle) hướng (ví dụ 0.7 ~ >45°)

BW_new = BW; % copy để cập nhật nối
linesConnected = {}; % cell lưu danh sách các đoạn đã nối

for i = 1:size(vectors,1)-1
    cx1 = vectors(i,1); cy1 = vectors(i,2);
    v1  = [vectors(i,3), vectors(i,4)];

    % kiểm tra component của endpoint i
    comp_id1 = findComponent(CC, [cy1,cx1]);
    if comp_id1==0 || numel(CC.PixelIdxList{comp_id1}) < minCompSize
        continue;
    end

    for j = i+1:size(vectors,1)
        cx2 = vectors(j,1); cy2 = vectors(j,2);
        v2  = [vectors(j,3), vectors(j,4)];

        % kiểm tra component j
        comp_id2 = findComponent(CC, [cy2,cx2]);
        if comp_id2==0 || numel(CC.PixelIdxList{comp_id2}) < minCompSize
            continue;
        end

        % --- khoảng cách Euclidean giữa 2 endpoint ---
        d = hypot(cx1-cx2, cy1-cy2);
        if d > maxDist, continue; end

        % --- kiểm tra hướng vector (cùng hướng nối) ---
        dir12 = [cx2-cx1, cy2-cy1];
        dir12 = dir12 / (norm(dir12)+eps);

        cond1 = dot(v1, dir12) > vecAlignThr;    % v1 hướng về P2
        cond2 = dot(v2, -dir12) > vecAlignThr;   % v2 hướng về P1

        if ~(cond1 && cond2), continue; end

        % --- kiểm tra thêm khoảng cách vuông góc ---
        % đường thẳng qua P2 với vector v2
        a = -v2(2);
        b =  v2(1);
        c =  v2(2)*cx2 - v2(1)*cy2;
        d_perp = abs(a*cx1 + b*cy1 + c) / sqrt(a^2 + b^2);

        if d_perp > 5, continue; end

        % --- nối 2 endpoint ---
        [BW_new, linePixels] = drawLine(BW_new, cx1, cy1, cx2, cy2);
        linesConnected{end+1} = linePixels; %#ok<AGROW>
    end
end
end
function [BW_new, linesConnected] = connectEndpoints_v2(BW, vectors, CC, minCompSize, maxDist, vecAlignThr)
% BW           : skeleton binary
% vectors      : [cx cy vx vy] từ hàm computeEndpointVectors
% CC           : bwconncomp(BW,8)
% minCompSize  : kích thước tối thiểu của vân
% maxDist      : khoảng cách tối đa cho phép nối
% vecAlignThr  : ngưỡng cos(angle) hướng (ví dụ 0.7 ~ >45°)

BW_new = BW; % copy để cập nhật nối
linesConnected = {}; % cell lưu danh sách các đoạn đã nối

used = false(size(vectors,1),1); % đánh dấu endpoint đã được nối

for i = 1:size(vectors,1)-1
    if used(i), continue; end  % bỏ qua nếu endpoint i đã nối

    cx1 = vectors(i,1); cy1 = vectors(i,2);
    v1  = [vectors(i,3), vectors(i,4)];

    % kiểm tra component của endpoint i
    comp_id1 = findComponent(CC, [cy1,cx1]);
    if comp_id1==0 || numel(CC.PixelIdxList{comp_id1}) < minCompSize
        continue;
    end

    for j = i+1:size(vectors,1)
        if used(j), continue; end  % bỏ qua nếu endpoint j đã nối

        cx2 = vectors(j,1); cy2 = vectors(j,2);
        v2  = [vectors(j,3), vectors(j,4)];

        % kiểm tra component j
        comp_id2 = findComponent(CC, [cy2,cx2]);
        if comp_id2==0 || numel(CC.PixelIdxList{comp_id2}) < minCompSize
            continue;
        end

        % --- khoảng cách Euclidean giữa 2 endpoint ---
        d = hypot(cx1-cx2, cy1-cy2);
        if d > maxDist, continue; end

        % --- kiểm tra hướng vector (cùng hướng nối) ---
        dir12 = [cx2-cx1, cy2-cy1];
        dir12 = dir12 / (norm(dir12)+eps);

        cond1 = dot(v1, dir12) > vecAlignThr;    % v1 hướng về P2
        cond2 = dot(v2, -dir12) > vecAlignThr;   % v2 hướng về P1

        if ~(cond1 && cond2), continue; end

        % --- kiểm tra thêm khoảng cách vuông góc ---
        a = -v2(2);
        b =  v2(1);
        c =  v2(2)*cx2 - v2(1)*cy2;
        d_perp = abs(a*cx1 + b*cy1 + c) / sqrt(a^2 + b^2);

        if d_perp > 10, continue; end

        % --- nối 2 endpoint ---
        [BW_new, linePixels] = drawLine(BW_new, cx1, cy1, cx2, cy2);
        linesConnected{end+1} = linePixels; %#ok<AGROW>

        % đánh dấu endpoint đã nối
        used([i j]) = true;

        break; % thoát vòng j, không nối thêm endpoint khác cho i
    end
end
end


function [BW_new, connected] = propagateConnect(BW, vectors, maxStep, maxDist)
% PROPAGATECONNECT - Nối các endpoint theo vector hướng
%
% Input:
%   BW       - ảnh nhị phân skeleton
%   vectors  - [x, y, vx, vy] cho mỗi endpoint
%   maxStep  - số pixel bước nhảy mỗi lần (ví dụ 2)
%   maxDist  - khoảng cách tối đa để nối
%
% Output:
%   BW_new   - ảnh skeleton đã nối
%   connected - danh sách các đoạn nối (cell array)

BW_new = BW;
connected = {};

N = size(vectors,1);

for i = 1:N
    cx = vectors(i,1);
    cy = vectors(i,2);
    vx = vectors(i,3);
    vy = vectors(i,4);

    % Đi theo hướng vector
    for step = 1:maxDist
        cx_new = round(cx + step * vx * maxStep);
        cy_new = round(cy + step * vy * maxStep);

        % Nếu ra ngoài ảnh thì dừng
        if cx_new < 1 || cx_new > size(BW,2) || cy_new < 1 || cy_new > size(BW,1)
            break;
        end

        % Nếu gặp skeleton khác thì nối
        if BW_new(cy_new, cx_new) == 1
            [BW_new, linePix] = drawLine(BW_new, cx, cy, cx_new, cy_new);
            connected{end+1} = linePix;
            break;
        end
    end
end
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

function [BW_new, connected] = propagateConnect_line_ao(BW, vectors, maxDist)
% PROPAGATECONNECT - Nối endpoint nếu line giả định cắt nhau hoặc gặp skeleton
%
% Input:
%   BW       - ảnh nhị phân skeleton
%   vectors  - [x, y, vx, vy] cho mỗi endpoint
%   maxDist  - chiều dài tối đa của line giả định
%
% Output:
%   BW_new   - ảnh skeleton đã nối
%   connected - danh sách đoạn nối

BW_new = BW;
connected = {};
N = size(vectors,1);

fakeLines = cell(N,1);

% --- 1. Vẽ line giả định cho tất cả endpoint ---
for i = 1:N
    cx = vectors(i,1);
    cy = vectors(i,2);
    vx = vectors(i,3);
    vy = vectors(i,4);

    cx_end = round(cx + vx * maxDist);
    cy_end = round(cy + vy * maxDist);

    [~, linePix] = drawLineMask(size(BW), cx, cy, cx_end, cy_end);
    fakeLines{i} = linePix;
end

% --- 2. Kiểm tra cắt nhau giữa line giả định ---
for i = 1:N-1
    for j = i+1:N
        li = fakeLines{i};
        lj = fakeLines{j};

        % Tìm điểm giao
        [~, ia, ib] = intersect(li, lj, 'rows');
        if ~isempty(ia)
            % Nối endpoint i và j
            cx1 = vectors(i,1); cy1 = vectors(i,2);
            cx2 = vectors(j,1); cy2 = vectors(j,2);

            [BW_new, linePixReal] = drawLine(BW_new, cx1, cy1, cx2, cy2);
            connected{end+1} = linePixReal;
        end
    end
end

% --- 3. Kiểm tra line giả định cắt skeleton thật ---
for i = 1:N
    li = fakeLines{i};
    for k = 1:size(li,1)
        x = li(k,1); y = li(k,2);
        if BW_new(y,x) == 1
            % Nối endpoint i tới điểm skeleton
            cx = vectors(i,1); cy = vectors(i,2);
            [BW_new, linePixReal] = drawLine(BW_new, cx, cy, x, y);
            connected{end+1} = linePixReal;
            break; % chỉ cần điểm gần nhất
        end
    end
end

end


function [BW_new, linePix] = drawLineMask(BW, x1, y1, x2, y2)
% DRAWLINE - Nối thật skeleton giữa 2 điểm
[x, y] = bresenham_line(x1, y1, x2, y2);

% Giới hạn trong ảnh
sz = size(BW);
valid = x>=1 & x<=sz(2) & y>=1 & y<=sz(1);
x = x(valid); y = y(valid);

idx = sub2ind(sz, y, x);
BW_new = BW;
BW_new(idx) = 1;

linePix = [x(:), y(:)];
end

function [x, y] = bresenham_line(x1, y1, x2, y2)
% BRESENHAM_LINE - Thuật toán Bresenham tạo line pixel
x1 = round(x1); y1 = round(y1);
x2 = round(x2); y2 = round(y2);

dx = abs(x2 - x1);
dy = abs(y2 - y1);
sx = sign(x2 - x1);
sy = sign(y2 - y1);
err = dx - dy;

x = []; y = [];
x_cur = x1; y_cur = y1;

while true
    x(end+1) = x_cur;
    y(end+1) = y_cur;
    if x_cur == x2 && y_cur == y2
        break;
    end
    e2 = 2*err;
    if e2 > -dy
        err = err - dy;
        x_cur = x_cur + sx;
    end
    if e2 < dx
        err = err + dx;
        y_cur = y_cur + sy;
    end
end
end


function BW_new = connectLongFringes(BW, Nfit, maxGap)
% CONNECTLONGFRINGES - Nối các vân cong bị đứt đoạn dài
%
% Syntax:
%   BW_new = connectLongFringes(BW, Nfit, maxGap)
%
% Input:
%   BW     - skeleton nhị phân (1 pixel wide)
%   Nfit   - số pixel dùng để fit vector hướng tại endpoint
%   maxGap - khoảng cách tối đa giữa 2 endpoint để nối
%
% Output:
%   BW_new - skeleton đã nối

BW_new = BW;

% 1. Tìm endpoint
endPoints = bwmorph(BW_new,'endpoints');
[y_end,x_end] = find(endPoints);
nEP = length(x_end);

% 2. Duyệt từng cặp endpoint
for i = 1:nEP-1
    p1 = [x_end(i), y_end(i)];
    for j = i+1:nEP
        p2 = [x_end(j), y_end(j)];

        % 2a. Kiểm tra khoảng cách
        if norm(p1-p2) > maxGap
            continue
        end

        % 2b. Ước lượng hướng tại endpoint
        v1 = estimateTangent(BW_new, p1, Nfit);
        v2 = estimateTangent(BW_new, p2, Nfit);

        % 2c. Sinh Bézier curve
        curve = generateBezierCurve(p1,p2,v1,v2,50); % 50 điểm


        x = round(curve(:,1));
        y = round(curve(:,2));

        % Clamp tọa độ
        x = max(1, min(size(BW_new,2), x));
        y = max(1, min(size(BW_new,1), y));

        if ~any(BW_new(sub2ind(size(BW_new), y, x)))
            % nối vân
        end

        % 2d. Kiểm tra va chạm
        if ~any(BW_new(sub2ind(size(BW_new), round(curve(:,2)), round(curve(:,1)))))
            % 2e. Nối vân
            for k = 1:size(curve,1)
                BW_new(round(curve(k,2)), round(curve(k,1))) = 1;
            end
        end
    end
end

end

%% ----- Hàm phụ trợ -----
function v = estimateTangent(BW, p, N)
% Lấy N pixel gần p dọc skeleton, fit polyline → vector hướng
[y_skel, x_skel] = find(BW);
dist = sqrt((x_skel - p(1)).^2 + (y_skel - p(2)).^2);
[~, idx] = sort(dist);
pts = [x_skel(idx(1:N)), y_skel(idx(1:N))];
p_fit = polyfit(pts(:,1), pts(:,2), 1); % slope
v = [1, p_fit(1)];
v = v / norm(v);
end

function curve = generateBezierCurve(p1,p2,v1,v2,nPoints)
% Tạo cubic Bézier curve nối p1 và p2 theo hướng v1,v2
L = norm(p2-p1)/3; % khoảng cách điều khiển
c1 = p1 + L*v1;
c2 = p2 - L*v2;

t = linspace(0,1,nPoints)';
curve = (1-t).^3*p1 + 3*(1-t).^2.*t.*c1 + 3*(1-t).*t.^2.*c2 + t.^3*p2;
end
%% connect khi xét qua toàn bộ endpoint và tìm điểm phù hợp nhất
function [BW_new, linesConnected] = connectEndpoints_v3(BW, vectors, CC, minCompSize, maxDist, vecAlignThr,maxPerp)
% BW           : skeleton binary
% vectors      : [cx cy vx vy] từ hàm computeEndpointVectors
% CC           : bwconncomp(BW,8)
% minCompSize  : kích thước tối thiểu của vân
% maxDist      : khoảng cách tối đa cho phép nối
% vecAlignThr  : ngưỡng cos(angle) hướng (ví dụ 0.7 ~ >45°)

BW_new = BW; % copy để cập nhật nối
linesConnected = {}; % cell lưu danh sách các đoạn đã nối

used = false(size(vectors,1),1); % đánh dấu endpoint đã được nối

for i = 1:size(vectors,1)-1
    if used(i), continue; end  % bỏ qua nếu endpoint i đã nối

    cx1 = vectors(i,1); cy1 = vectors(i,2);
    v1  = [vectors(i,3), vectors(i,4)];

    % kiểm tra component của endpoint i
    comp_id1 = findComponent(CC, [cy1,cx1]);
    if comp_id1==0 || numel(CC.PixelIdxList{comp_id1}) < minCompSize
        continue;
    end

    best_j   = 0;
    bestCost = inf;

    for j = i+1:size(vectors,1)
        if used(j), continue; end

        cx2 = vectors(j,1); cy2 = vectors(j,2);
        v2  = [vectors(j,3), vectors(j,4)];

        % kiểm tra component j
        comp_id2 = findComponent(CC, [cy2,cx2]);
        if comp_id2==0 || numel(CC.PixelIdxList{comp_id2}) < minCompSize
            continue;
        end

        % --- khoảng cách Euclidean ---
        d = hypot(cx1-cx2, cy1-cy2);
        if d > maxDist, continue; end

        % --- hướng vector ---
        dir12 = [cx2-cx1, cy2-cy1];
        dir12 = dir12 / (norm(dir12)+eps);

        cond1 = dot(v1, dir12) > vecAlignThr;
        cond2 = dot(v2, -dir12) > vecAlignThr;
        if ~(cond1 && cond2), continue; end

        % --- khoảng cách vuông góc ---
        a = -v2(2);
        b =  v2(1);
        c =  v2(2)*cx2 - v2(1)*cy2;
        d_perp = abs(a*cx1 + b*cy1 + c) / sqrt(a^2 + b^2);

        if d_perp > maxPerp, continue; end

        % --- tính "cost" để chọn ứng viên tốt nhất ---
        % --- khoảng cách chuẩn hóa ---
        d_norm = d / maxDist;  % [0,1]

        % --- sai lệch góc ---
        ang1 = acos(dot(v1, dir12));     % góc v1 với hướng nối
        ang2 = acos(dot(v2, -dir12));    % góc v2 với hướng nối ngược
        ang_err = (ang1 + ang2)/2;       % sai số góc trung bình (rad)

        ang_norm = ang_err / (pi/2);     % chuẩn hóa [0,1], 0 tốt, 1 tệ

        % --- khoảng cách vuông góc (cũng chuẩn hóa) ---
        d_perp_norm = min(d_perp/10,1);  % giới hạn max =1
        w1=0.2; w2=0.7; w3=0.1;
        % --- cost tổng hợp ---
        cost = w1*d_norm + w2*ang_norm + w3*d_perp_norm;
        if cost < bestCost
            bestCost = cost;
            best_j   = j;
        end
    end

    % Sau khi duyệt hết, nếu tìm thấy ứng viên tốt nhất thì nối
    if best_j > 0
        cx2 = vectors(best_j,1); cy2 = vectors(best_j,2);
        [BW_new, linePixels] = drawLine(BW_new, cx1, cy1, cx2, cy2);
        linesConnected{end+1} = linePixels; %#ok<AGROW>
        used([i best_j]) = true;
    end

end
end




function pixels = bresenhamLine(x0, y0, x1, y1)
    % Thuật toán Bresenham để vẽ đường thẳng pixel-perfect
    
    x0 = round(x0); y0 = round(y0);
    x1 = round(x1); y1 = round(y1);
    
    dx = abs(x1 - x0);
    dy = abs(y1 - y0);
    
    if x0 < x1
        sx = 1;
    else
        sx = -1;
    end
    
    if y0 < y1
        sy = 1;
    else
        sy = -1;
    end
    
    err = dx - dy;
    
    pixels = [];
    x = x0;
    y = y0;
    
    while true
        pixels = [pixels; x, y];
        
        if x == x1 && y == y1
            break;
        end
        
        e2 = 2 * err;
        
        if e2 > -dy
            err = err - dy;
            x = x + sx;
        end
        
        if e2 < dx
            err = err + dx;
            y = y + sy;
        end
    end
end


function linePixels = bresenhamLine2(x1, y1, x2, y2)
% Bresenham line algorithm, trả về [x y] pixel từ (x1,y1) đến (x2,y2)

x1 = round(x1); y1 = round(y1);
x2 = round(x2); y2 = round(y2);

dx = abs(x2 - x1);
dy = abs(y2 - y1);
sx = sign(x2 - x1);
sy = sign(y2 - y1);

err = dx - dy;
x = x1; y = y1;

linePixels = [];
while true
    linePixels = [linePixels; x y];
    if x == x2 && y == y2, break; end
    e2 = 2*err;
    if e2 > -dy, err = err - dy; x = x + sx; end
    if e2 < dx,  err = err + dx; y = y + sy; end
end

end



%% ================= Core Function =================
function [BW_new, linesOut] = extendEndpointsToBorder(BW, endPoints, vectors, margin, stopOnSkeleton, thickness)
% extendEndpointsToBorder - Nối từ endpoint ra tới biên theo hướng "ra ngoài".
%
% Cú pháp:
%   [BW_new, linesOut] = extendEndpointsToBorder(BW, endPoints, vectors, margin, stopOnSkeleton, thickness)
%
% Input:
%   BW             - ảnh nhị phân skeleton (logical)
%   endPoints      - ảnh nhị phân endpoint (1 tại endpoint)
%   vectors        - Mx4 [x, y, vx, vy] tại mỗi endpoint (tọa độ theo cột=x, hàng=y)
%   margin         - chỉ xét endpoint có khoảng cách tới biên <= margin
%   stopOnSkeleton - true: dừng nếu chạm pixel 1 của BW trong quá trình nối
%   thickness      - bề dày nét vẽ (>=1)
%
% Output:
%   BW_new   - ảnh skeleton sau khi nối
%   linesOut - cell, mỗi phần tử là Nx2 toạ độ [x,y] của đoạn nối (để debug/hiển thị)

BW_new = BW;
[H,W] = size(BW);
linesOut = {};

% Lấy danh sách endpoint từ mask
[yE,xE] = find(endPoints);

% Tạo tra cứu index cho vectors theo (x,y)
% (giả định vectors có cùng số endpoint; nếu không, sẽ ghép bằng gần nhất)
XYv = vectors(:,1:2);

for i = 1:numel(xE)
    x0 = xE(i); y0 = yE(i);

    % Bỏ qua endpoint không có vector khớp
    vidx = find(XYv(:,1)==x0 & XYv(:,2)==y0, 1, 'first');
    if isempty(vidx), continue; end

    % Chỉ xét endpoint gần biên
    if min([x0-1, y0-1, W-x0, H-y0]) > margin
        continue;
    end

    vx = vectors(vidx,3); vy = vectors(vidx,4);
    if vx==0 && vy==0, continue; end

    % Chọn hướng "ra ngoài": hướng nào chạm biên SỚM HƠN
    [t1, xB1, yB1] = ray_to_image_border(x0,y0, vx, vy, W, H);
    [t2, xB2, yB2] = ray_to_image_border(x0,y0,-vx,-vy, W, H);

    if isempty(t1) && isempty(t2)
        continue; % không bắn được ra biên (hiếm)
    end

    if isempty(t2) || (~isempty(t1) && t1 <= t2)
        x1 = xB1; y1 = yB1;  % dùng (vx,vy)
    else
        x1 = xB2; y1 = yB2;  % dùng (-vx,-vy)
    end

    % Lấy các pixel trên đoạn nối (bao gồm điểm đầu và cuối)
    [xs, ys] = rasterize_line(x0,y0, x1,y1);

    % Nếu cần dừng khi gặp skeleton hiện có, cắt tại điểm chạm đầu tiên (bỏ (x0,y0))
    if stopOnSkeleton
        hitIdx = [];
        for k = 2:numel(xs) % bắt đầu từ pixel kế endpoint
            if BW_new(ys(k), xs(k))
                hitIdx = k; break;
            end
        end
        if ~isempty(hitIdx)
            xs = xs(1:hitIdx); ys = ys(1:hitIdx);
        end
    end

    % Vẽ lên BW_new với bề dày "thickness"
    BW_new = drawLineMask(BW_new, xs, ys, thickness);

    % Lưu để hiển thị/debug
    linesOut{end+1} = [xs(:), ys(:)]; %#ok<AGROW>
end
end

%% ----- Utility: tìm giao điểm tia với biên ảnh -----
function [tHit, xB, yB] = ray_to_image_border(x0,y0, dx,dy, W,H)
% Trả về tham số tHit >= 0 và toạ độ (xB,yB) khi tia (x0,y0)+t*[dx,dy] chạm biên ảnh
% Nếu không có giao điểm hợp lệ trong ảnh, trả về []

epsv = 1e-12; tCand = [];
% Các biên: x=1, x=W, y=1, y=H
if abs(dx) > epsv
    t = (1 - x0)/dx;  if t>=0, tCand(end+1) = t; end %#ok<AGROW>
    t = (W - x0)/dx;  if t>=0, tCand(end+1) = t; end %#ok<AGROW>
end
if abs(dy) > epsv
    t = (1 - y0)/dy;  if t>=0, tCand(end+1) = t; end %#ok<AGROW>
    t = (H - y0)/dy;  if t>=0, tCand(end+1) = t; end %#ok<AGROW>
end

% Lọc những ứng viên thực sự nằm trong hộp [1..W]x[1..H]
ok = false(size(tCand));
xBv = zeros(size(tCand)); yBv = zeros(size(tCand));
for i = 1:numel(tCand)
    xBv(i) = x0 + tCand(i)*dx;
    yBv(i) = y0 + tCand(i)*dy;
    if xBv(i) >= 1-1e-9 && xBv(i) <= W+1e-9 && yBv(i) >= 1-1e-9 && yBv(i) <= H+1e-9
        ok(i) = true;
    end
end

if ~any(ok)
    tHit = []; xB = []; yB = [];
    return;
end

% Lấy t nhỏ nhất (chạm biên sớm nhất)
[tHit, idx] = min(tCand(ok));
xB = xBv(ok); xB = xB(idx);
yB = yBv(ok); yB = yB(idx);

% Clamp vào biên nguyên (pixel)
xB = min(max(xB,1),W); yB = min(max(yB,1),H);
end

%% ----- Utility: rasterize đoạn thẳng thành dãy pixel -----
function [xs, ys] = rasterize_line(x0,y0, x1,y1)
% Lấy N điểm nguyên theo bước 1 pixel dựa trên độ dài Chebyshev
N = max(abs(round(x1)-round(x0)), abs(round(y1)-round(y0))) + 1;
xs = round(linspace(x0, x1, N));
ys = round(linspace(y0, y1, N));
end

%% ----- (Tuỳ chọn) Vector hướng đơn giản bằng PCA lân cận -----
function vectors = fitEndpointVectors_simple(BW, endPoints, Nfit)
% Trả về [x, y, vx, vy] cho mỗi endpoint dựa trên PCA các pixel skeleton lân cận.
[yE, xE] = find(endPoints);
coords = [xE, yE];
[H,W] = size(BW);

vectors = zeros(numel(xE),4);
for i = 1:numel(xE)
    x0 = xE(i); y0 = yE(i);
    % Lấy vùng lân cận hình vuông (2*Nfit+1)
    x1 = max(1, x0-Nfit); x2 = min(W, x0+Nfit);
    y1 = max(1, y0-Nfit); y2 = min(H, y0+Nfit);
    [yy,xx] = find(BW(y1:y2, x1:x2));
    if numel(xx) < 3
        continue;
    end
    xx = xx + x1 - 1; yy = yy + y1 - 1;

    % PCA
    X = [double(xx(:))-x0, double(yy(:))-y0];
    C = cov(X);
    [V,~] = eig(C);
    v = V(:,2); % vector riêng lớn nhất (cột 2)
    vx = v(1); vy = v(2);
    nrm = hypot(vx,vy); if nrm>0, vx=vx/nrm; vy=vy/nrm; end

    vectors(i,:) = [x0, y0, vx, vy];
end
end
