clc, clear, close all;

%% --- 1. CẤU HÌNH THAM SỐ (PARAMETERS) ---
% Thông số đường dẫn
baseFolder = "C:\Users\admin\Máy tính\Lab thầy Tùng\My_TN\data 12 11 25\60x o thu 6\251112\anh oke";
fileName = "image_2025-11-12T18-33-12.732 - Copy.bmp";

% Thông số xử lý ảnh
SENSITIVITY_COEF = 0.65;   % Hệ số nhạy adaptive threshold
NEIGHBORHOOD_SIZE = 51;    % Kích thước vùng lân cận (phải là số lẻ)
MIN_BRANCH_LENGTH = 8;    % Ngưỡng độ dài nhánh thừa cần xóa (thay cho distThresh cũ)
GAUSS_SIGMA = 1;           % Độ làm mượt ảnh
distThresh = 6;
%% --- 2. ĐỌC VÀ TIỀN XỬ LÝ ẢNH ---

imgPath = fullfile(baseFolder, fileName);

if ~isfile(imgPath)
    % Fallback: Nếu không tìm thấy file cứng, mở hộp thoại chọn
    fprintf('Không tìm thấy file mặc định. Vui lòng chọn ảnh...\n');
    [fileName, baseFolder] = uigetfile({'*.bmp;*.png;*.jpg'}, 'Chọn file ảnh');
    if isequal(fileName, 0), return; end
    imgPath = fullfile(baseFolder, fileName);
end

hologram = imread(imgPath);
if size(hologram, 3) == 3
    hologram = rgb2gray(hologram);
end

% Xoay ảnh (Lưu ý: rot90(img, k). k=1 là 90 độ, k=2 là 180 độ)
% Theo comment cũ của bạn là 180 độ, nên tôi để k=2. Nếu muốn 90 thì sửa thành 1.
hologram = rot90(hologram, 1); 

% 2.1. Lọc nhiễu và cân bằng histogram
hologram = imgaussfilt(hologram, GAUSS_SIGMA);

hologram = adapthisteq(hologram);

%% --- 3. NHỊ PHÂN HÓA (BINARIZATION) ---
% Tính ngưỡng thích nghi
T = adaptthresh(hologram, SENSITIVITY_COEF, ...
    'NeighborhoodSize', [NEIGHBORHOOD_SIZE NEIGHBORHOOD_SIZE], ...
    'Statistic', 'median');

BW = imbinarize(hologram, T);
fprintf('Đang thực hiện Skeletonize và xóa nhánh nhỏ...\n');

se_cut = strel('disk', 1); 
BW_open = imopen(BW, se_cut);

figure; 
imshow(BW_open); 
title(' anh bw sau khi imopen');

% --- CÁCH MỚI (Tối ưu tốc độ và hiệu quả) ---
% bwskel: Hàm này nhanh hơn và mạnh hơn bwmorph.
% 'MinBranchLength': Tự động xóa các nhánh cụt ngắn hơn giá trị này (thay thế vòng lặp geodesic)
BW_skel_clean = bwskel(BW_open, 'MinBranchLength', MIN_BRANCH_LENGTH);

% Làm sạch thêm các điểm cô lập (nếu cần)
BW_skel_clean = bwmorph(BW_skel_clean, 'clean');

%% --- 5. HIỂN THỊ KẾT QUẢ (VISUALIZATION) ---
% Hình 5: So sánh chồng ảnh (Overlay)
figure; 
imshowpair(BW_skel_clean, BW_open, 'falsecolor'); 
title('5. Skeleton Overlay');

% Hình 6: Kết quả Skeleton cuối cùng
figure(); imshow(BW_skel_clean); 
title(['6. Final Skeleton (Đã xóa nhánh < ', num2str(MIN_BRANCH_LENGTH), 'px)']);
%%
branchpoints = bwmorph(BW_skel_clean, 'branchpoints');

[rows, cols] = find(branchpoints); 
endpoints = bwmorph(BW_skel_clean, "endpoints");
[r,c] = find(endpoints);

figure;
imshow(BW_skel_clean);
hold on;
plot(cols, rows, "go", 'MarkerSize', 10, 'LineWidth',1.5);
plot(c, r, "r+", 'MarkerSize', 10, 'LineWidth',1.5);

hold off;

%% cắt cách điểm endpoint và branchpoints gần nhau
BW = BW_skel_clean;
MIN_BRANCH_LENGTH = 15;
BW = bwskel(BW, 'MinBranchLength', MIN_BRANCH_LENGTH);

%%
BW = bwfill(BW,'holes');
BW = bwskel(BW, 'MinBranchLength', MIN_BRANCH_LENGTH);


%% ngawts ket noi H-brigde 
% 1. Tìm các điểm nút (Branchpoints)
B = bwmorph(BW, 'branchpoints');
R = 4; 
se_disk = strel('disk', R);

% 3. Thực hiện phép Phình to (Dilation)
B = imdilate(B, se_disk);

% 2. Tìm các điểm đầu cuối (Endpoints) - Để tham khảo xem có bao nhiêu
E = bwmorph(BW, 'endpoints');

% 3. "Tháo khớp": Xóa điểm nút khỏi khung xương để tách các đoạn
% Đoạn xương rời rạc = Xương gốc - Điểm nút
BW = BW & ~B;

% 4. Lọc bỏ các đoạn ngắn (Gai hoặc Cầu nối sai)
min_len = 30; % Ngưỡng độ dài (pixel)
BW = bwareaopen(BW, min_len);

% 5. "Lắp lại": Cộng lại các điểm nút vào những đoạn đã giữ
% Lưu ý: Khi cộng lại có thể dư ra 1 chút mấu nhỏ, cần thin lại 1 lần
BW = BW | B;
BW = bwmorph(BW, 'thin', Inf); 

% --- HIỂN THỊ KẾT QUẢ ---
figure('Name', 'Clean Spurs', 'Color', 'w');
imshow(BW); title('1. Skeleton Gốc (Có gai)');

%%
%% --- Tìm endpoint ---
BW = bwmorph(BW,"bridge",Inf);
BW = bwmorph(BW,"diag", Inf);
BW = bwmorph(BW,"skeleton", Inf);
BW = bwmorph(BW,'spur',5);

endPoints = bwmorph(BW, 'endpoints');

% --- Tính hướng vector tại endpoint ---
Nfit = 12;  % số pixel dùng để fit PCA
vectors = fitEndpointVectors(BW, endPoints, Nfit);

% --- Hiển thị kết quả ---
figure;
imshow(BW); hold on;
title('Hướng tại các Endpoint');
for i = 1:size(vectors, 1)
    cx = vectors(i,1);
    cy = vectors(i,2);
    vx = vectors(i,3);
    vy = vectors(i,4);

    % Vẽ vector hướng (mũi tên)
    quiver(cx, cy, vx*10, vy*10, 'r', 'LineWidth', 1.5, 'MaxHeadSize', 2);
end
hold off;

%% noois diem
n = 8;
for count =1:n
    % Timf endPoint
    % Kernel để đếm số hàng xóm (8-neighbors)
    endPoints = bwmorph(BW, 'endpoints');
    figure; imshow(BW); title('Skeleton gốc');
    [row, col] = find(endPoints);
    hold on; plot(col, row, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
    %
    fprintf('--> Bước 2b: Ước lượng vector hướng theo đoạn liên thông\n');
    vectors = fitEndpointVectors(BW, endPoints, 12);

    imshow(BW); hold on;
    for i = 1:size(vectors,1)
        cx = vectors(i,1);
        cy = vectors(i,2);
        vx = vectors(i,3);
        vy = vectors(i,4);
        quiver(cx, cy, 10*vx, 10*vy, 'r', 'LineWidth',2, 'MaxHeadSize',2);
    end
    title('Vector hướng tại các endpoint');

    %%
    fprintf('--> Bước 2c: Nối các endpoint theo vector hướng\n');
    if count == 1 %nối vân dài + góc lệch nhỏ + khoảng cách nhỏ
        fprintf('--> lan chay dau tien\n');
        minCompSize = 12;   % chỉ nối nếu component đủ dài
        maxDist     = 6;   % khoảng cách tối đa giữa 2 endpoint
        vecAlignThr = cosd(15);  % = 0.866 ~ hướng lệch <= 30°
    end
    if count ==2 % Nối vân dài + góc lệch lớn + khoảng cách lớn
        fprintf('--> Bước 2d: Nối các endpoint theo vector hướng\n');
        fprintf('--> lan chay lan 2\n');

        minCompSize = 12;   % chỉ nối nếu component đủ dài
        maxDist     = 12;   % khoảng cách tối đa giữa 2 endpoint
        vecAlignThr = cosd(15);  % = 0.866 ~ hướng lệch <= 30°
    end
    if count == 3 % Nối vân dài + góc lệch nhỏ + khoảng cách lớn hơn
        fprintf('--> Bước 2d: Nối các endpoint theo vector hướng\n');
        fprintf('--> lan chay lan 3\n');

        minCompSize = 12;   % chỉ nối nếu component đủ dài
        maxDist     = 25;   % khoảng cách tối đa giữa 2 endpoint
        vecAlignThr = cosd(30);  % = 0.866 ~ hướng lệch <= 30°
    end
    if count == 4 % Nối vân ngắn + góc lệch lớn
        fprintf('--> Bước 2d: Nối các endpoint theo vector hướng\n');
        fprintf('--> lan chay lan 4\n');

        minCompSize = 5;   % chỉ nối nếu component đủ dài
        maxDist     = 50;   % khoảng cách tối đa giữa 2 endpoint
        vecAlignThr = cosd(40);  % = 0.866 ~ hướng lệch <= 30°
    end
    if count == 5 % Nối vân ngắn + góc lệch lớn
        fprintf('--> Bước 2d: Nối các endpoint theo vector hướng\n');
        fprintf('--> lan chay lan 5\n');

        minCompSize = 20;   % chỉ nối nếu component đủ dài
        maxDist     = 50;   % khoảng cách tối đa giữa 2 endpoint
        vecAlignThr = cosd(15);  % = 0.866 ~ hướng lệch <= 30°
    end
    if count == 6 % Nối vân ngắn + góc lệch lớn
        fprintf('--> Bước 2d: Nối các endpoint theo vector hướng\n');
        fprintf('--> lan chay lan 6\n');

        minCompSize = 20;   % chỉ nối nếu component đủ dài
        maxDist     = 50;   % khoảng cách tối đa giữa 2 endpoint
        vecAlignThr = cosd(30);  % = 0.866 ~ hướng lệch <= 30°
    end
    if count == 7 % Nối vân ngắn + góc lệch lớn
        fprintf('--> Bước 2d: Nối các endpoint theo vector hướng\n');
        fprintf('--> lan chay lan 7\n');
        minCompSize = 20;   % chỉ nối nếu component đủ dài
        maxDist     = 50;   % khoảng cách tối đa giữa 2 endpoint
        vecAlignThr = cosd(30);  % = 0.866 ~ hướng lệch <= 30°
    end
    if count == 8 % Nối vân ngắn + góc lệch lớn
        fprintf('--> Bước 2d: Nối các endpoint theo vector hướng\n');
        fprintf('--> lan chay lan 7\n');
        minCompSize = 20;   % chỉ nối nếu component đủ dài
        maxDist     = 60;   % khoảng cách tối đa giữa 2 endpoint
        vecAlignThr = cosd(80);  % = 0.866 ~ hướng lệch <= 30°
    end

    CC = bwconncomp(BW, 8);

    [BW, linesConnected] = connectEndpoints(BW, vectors, CC, minCompSize, maxDist, vecAlignThr);
    % xoá vùng nhỏ lẻ

    figure; imshow(BW); hold on;
    for k = 1:numel(linesConnected)
        lineXY = linesConnected{k};
        plot(lineXY(:,1), lineXY(:,2), 'g-', 'LineWidth', 2);
    end
    title('Skeleton sau khi nối endpoint (màu xanh)');

end
maxLen = 150;    % độ dài tối đa kéo dài (pixel)
step = 1;        % bước mẫu dọc theo ray (pixel)
connectThresh = 3; % ngưỡng khoảng cách để coi là "gặp" (pixel)

% Gọi hàm chính
[BW_new, connections] = extend_and_connect(BW, endpoints, vectors, maxLen, step, connectThresh);

% Hiển thị kết quả
figure; imshow(BW_new); title('Kết quả: Đoạn gốc (trắng) + kết nối (trắng)');
hold on;
% vẽ endpoints nguyên thủy
plot(endpoints(:,1), endpoints(:,2), 'ro', 'MarkerSize',6, 'LineWidth',1.5);
for k=1:size(connections,1)
    p1 = connections{k,1}; p2 = connections{k,2};
    plot([p1(1), p2(1)], [p1(2), p2(2)], 'g-', 'LineWidth',1.5);
end
hold off;


skel_repaired = connect_endpoints_by_ray_intersection(BW);

% Hiển thị kết quả cuối cùng
figure; imshow(skel_repaired); title('Skeleton sau khi nối');

%%
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



%% ========== Hàm chính ==========
function [BW_out, connections] = extend_and_connect(BW, endpoints, vectors, maxLen, step, connectThresh)
% EXTEND_AND_CONNECT Extend rays from endpoints and connect if two rays meet/near.
% inputs:
%   BW         - binary image (H x W) của các đoạn ban đầu (logical)
%   endpoints  - Nx2 matrix [x y] (pixel coordinates)
%   vectors    - Nx2 matrix hướng tương ứng (không cần chuẩn hóa)
%   maxLen     - chiều dài tối đa kéo dài (pixel)
%   step       - bước mẫu dọc ray (pixel)
%   connectThresh - ngưỡng khoảng cách (pixel) để nối
% outputs:
%   BW_out     - binary image cập nhật (kèm đường nối)
%   connections - cell array Mx2; mỗi hàng là hai điểm gần nhau nối (x y)

    assert(size(endpoints,2) == 2, 'endpoints must be Nx2 [x y]');
    assert(size(vectors,1) == size(endpoints,1), 'vectors must match endpoints count');

    [H,W] = size(BW);
    N = size(endpoints,1);

    % Tạo tập điểm dọc theo mỗi ray
    rays = cell(N,1);
    for i=1:N
        v = vectors(i,:);
        if all(v==0)
            rays{i} = endpoints(i,:); continue;
        end
        vn = v / norm(v);
        ts = (0:step:maxLen)'; % bao gồm điểm gốc
        pts = endpoints(i,:) + ts * vn; % kích thước length(ts) x 2
        % Loại bỏ điểm ra ngoài ảnh
        valid = pts(:,1) >= 1 & pts(:,1) <= W & pts(:,2) >= 1 & pts(:,2) <= H;
        rays{i} = round(pts(valid,:)); % làm tròn về pixel
        % Loại bỏ điểm trùng nhau liên tiếp
        rays{i} = unique(rays{i}, 'rows', 'stable');
    end

    % Kiểm tra cặp ray -> ray
    connections = {};
    paired = false(N); % đánh dấu đã nối cặp nào
    for i=1:N-1
        for j=i+1:N
            if isempty(rays{i}) || isempty(rays{j}), continue; end
            % Tính khoảng cách giữa mọi cặp điểm (cẩn thận về hiệu năng)
            % nếu m*n quá lớn, giới hạn bằng downsample hoặc kiểm tra theo block
            P = double(rays{i});
            Q = double(rays{j});
            % Thực hiện pdist2 (fast enough cho điểm ray ~ vài trăm)
            D = pdist2(P, Q, 'euclidean');
            [minVal, idx] = min(D(:));
            if minVal <= connectThresh
                [ia, jb] = ind2sub(size(D), idx);
                p_closest = P(ia,:);
                q_closest = Q(jb,:);
                connections(end+1,1:2) = {p_closest, q_closest}; %#ok<AGROW>
                paired(i,j) = true;
                paired(j,i) = true;
            end
        end
    end

    % Tạo ảnh mới và vẽ các đường nối (giữa điểm gần nhất trên hai ray)
    BW_out = BW;
    for k=1:size(connections,1)
        p = connections{k,1};
        q = connections{k,2};
        BW_out = drawLineBW(BW_out, p, q);
    end
end

