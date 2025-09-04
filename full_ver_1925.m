clc; clear; close all;

% --- Skeleton giả định (test) ---
%% 1. KHỞI TẠO
fprintf('Bắt đầu quy trình mô phỏng và tái tạo...\n');

%% 2. MÔ PHỎNG HOLOGRAM
fprintf('--> Bước 1: Mô phỏng Hologram...\n');
% --- Thiết lập thông số ---
% M = 512; % Kích thước ảnh (chiều cao)
% N = 512; % Kích thước ảnh (chiều rộng)
% snr = 13;
% 
% auto_fft = 0;
% 
% % 
% % % nhiễu - phương sai: sigma
% % sigma = pi/5;
% % noise_level = 0;
% % noise = noise_level * randn(N, N) .* sigma;
% 
% 
% [X, Y] = meshgrid(linspace(-1,1,N), linspace(-1,1,M));
% object_phase_without_noise = 2 * peaks(3*X, 3*Y);



M = 512; % Kích thước ảnh (chiều cao)
N = 512; % Kích thước ảnh (chiều rộng)
snr = 30;
% fprintf('\n====================================================\n');
fprintf('ĐANG CHẠY MÔ PHỎNG VỚI SNR = %.1f dB\n', snr);
% fprintf('====================================================\n');
auto_fft = 0;

% 
% % nhiễu - phương sai: sigma
% sigma = pi/5;
% noise_level = 0;
% noise = noise_level * randn(N, N) .* sigma;

% --- Tạo lưới toạ độ ---
[x, y] = meshgrid(1:N, 1:M);

% --- Thông số sóng sin ---
freq_x = 4;   % số chu kỳ theo trục x
freq_y = 4;   % số chu kỳ theo trục y
amp    = 10; % biên độ pha (rad)

% --- Pha object dạng hình sin ---
object_phase_without_noise = amp * sin(2*pi*freq_x*x/N);
%%
% Thêm nhiễu vào pha đối tượng
object_phase = awgn(object_phase_without_noise, snr, 'measured', 'db');
% figure;
% surf(object_phase,"EdgeColor","none");
% title("doi tuong co nhieu- groundtruth");

% --- Hiển thị pha gốc (không nhiễu) ---

% figure;
% surf(object_phase_without_noise, "EdgeColor", "none");
% colorbar;
% title('Đối tượng pha (không nhiễu): ');

% 3. TẠO HOLOGRAM
fprintf('--> Bước 2: Tạo Hologram...\n');

% --- Thiết lập thông số sóng mang ---
fx = 20 / N; % Tần số sóng mang
fy = -30 / M;
[X, Y] = meshgrid(1:N, 1:M);

% Cường độ nền và điều biến
a = 1.0; % Background intensity
b = 0.8; % Modulation depth

% Sóng mang phẳng (plane wave carrier)
carrier = 2 * pi * (fx * X + fy * Y);

% --- Tạo hologram (ảnh giao thoa) theo công thức mới ---
hologram = a + b .* cos(carrier + object_phase);

% --- Hiển thị Hologram ---
figure;
imshow(hologram, []);
title('Ảnh Hologram (Giao thoa) có nhiễu');
%% 3. Tạo bề mặt interferogram
hologram = mat2gray(hologram);
imwrite(hologram, 'hologram.bmp');

%% 5. Noise removal
hologram = imgaussfilt(hologram, 1);
% hologram = medfilt2(hologram, [3 3]);
% hologram = wiener2(hologram, [5 5]);
figure;
imshow(hologram);
colorbar;
title('hologram sau noise removal : ');
%% 6. Histogram equalization
% hologram = adapthisteq(hologram);
% figure;
% imshow(hologram);
% colorbar;
% title('hologram sau equaliztion histogram : ');

input_image = hologram;
%% 7. ƯỚC LƯỢNG PHA BẰNG PHƯƠNG PHÁP PHÂN TÍCH VÂN
fprintf('--> Bước 3: Ước lượng pha thô bằng phân tích vân...\n');
% Làm mảnh và gán bậc vân
% % Chuyển đổi sang ảnh xám nếu cần
if size(input_image, 3) == 3
    input_image = rgb2gray(input_image);
    fprintf('Đã chuyển đổi ảnh RGB sang grayscale\n');
end

    fprintf('Bắt đầu quá trình skeletonization...\n');

    % --- Bước 1: Nhị phân hóa ảnh bằng Otsu ---
    fprintf('Bước 1/3: Nhị phân hóa ảnh bằng phương pháp Otsu...\n');
    thresh = graythresh(input_image);
    BW_Original = imbinarize(input_image, thresh);

    fprintf('Ngưỡng Otsu: %.4f\n', thresh);
    fprintf('Số pixel foreground: %d\n', sum(BW_Original(:)));

    % --- Bước 2: Skeletonize bằng Zhang-Suen ---
    fprintf('Bước 2/3: Áp dụng thuật toán Zhang-Suen...\n');
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

        % Hiển thị tiến trình mỗi 10 iterations
        if mod(iteration, 10) == 0
            fprintf('  Iteration %d: %d pixels còn lại\n', iteration, sum(BW_Thinned(:)));
        end

        % Tránh vòng lặp vô hạn
        if iteration > 1000
            warning('Đã đạt giới hạn iteration (1000). Dừng thuật toán.');
            break;
        end
    end

    fprintf('Hoàn thành sau %d iterations\n', iteration);
    fprintf('Số pixel skeleton: %d\n', sum(BW_Thinned(:)));

        fprintf('Bước 3/3: Hiển thị kết quả...\n');

        figure('Name', 'Kết quả Skeletonization Zhang-Suen', 'NumberTitle', 'off');

        % Hiển thị so sánh
        subplot(1, 3, 1);
        imshow(input_image);
        title('Ảnh gốc', 'FontSize', 12);

        subplot(1, 3, 2);
        imshow(BW_Original);
        title('Ảnh nhị phân (Otsu)', 'FontSize', 12);

        subplot(1, 3, 3);
        imshow(BW_Thinned);
        title('Skeleton (Zhang-Suen)', 'FontSize', 12);

        % Điều chỉnh layout
        sgtitle('Quá trình Skeletonization', 'FontSize', 14, 'FontWeight', 'bold');
    

    % --- Trả về kết quả ---
    skeleton_image = BW_Thinned;
    binary_image = BW_Original;

skeleton = skeleton_image;

%%
BW = skeleton;
fprintf('Running Modified ZS (MZS) thinning...\n');
S = MZS_thinning(BW);

figure;
subplot(1,2,1); imshow(BW); title('Input binary');
subplot(1,2,2); imshow(S);  title('Skeleton (MZS)');

BW = S;


main_loop = 1;
for count_main_loop = 1:main_loop
%% Xoa vung nho le
BW = removeSmallComponents(BW, 5);  % xoá vùng liên thông < 10 pixel

%% Xoá junction

[BW, junctionMap] = removeJunctions(BW);
BW = bwmorph(BW,"spur", 3);

% figure; imshow(BW); hold on;
% [row, col] = find(junctionMap);
% plot(col, row, 'go', 'MarkerSize',10,'LineWidth',1);
% title('Skeleton sau khi xoá junction');
%
% % Xóa các đoạn ngắn (bridge)
maxBridgeLen = 5;
BW = bwareaopen(BW, maxBridgeLen);
BW = BW(2:end-1, 2:end-1);
%% --- 4. Nối endpoint theo vòng lặp thử ---
nLoop = 8; % số vòng nối
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
        vectors = fitEndpointVectors(BW, endPoints, 15);
        max_perh = 5;

    end
    if  count == 2
        fprintf('--> Vòng nối %d ----\n', count);

        % Tham số nối thử
        minCompSize = 12;
        maxDist     = 20;    % vòng sau cho phép nối xa hơn
        vecAlignThr = cosd(15);    % ~0.866
        max_perh = 5;
        vectors = fitEndpointVectors(BW, endPoints, 15);

    end
    if  count == 3
        fprintf('--> Vòng nối %d ----\n', count);

        % Tham số nối thử
        minCompSize = 8;
        maxDist     = 25;    % vòng sau cho phép nối xa hơn
        vecAlignThr = cosd(15);    % ~0.866
        max_perh = 5;

        vectors = fitEndpointVectors(BW, endPoints, 15);

    end

    if  count == 4
        fprintf('--> Vòng nối %d ----\n', count);

        % Tham số nối thử
        minCompSize = 5;
        maxDist     = 30;    % vòng sau cho phép nối xa hơn
        vecAlignThr = cosd(20);    % ~0.866
        max_perh = 5;

        vectors = fitEndpointVectors(BW, endPoints, 15);

    end
    if  count == 5
        fprintf('--> Vòng nối %d ----\n', count);

        % Tham số nối thử
        minCompSize = 5;
        maxDist     = 35;    % vòng sau cho phép nối xa hơn
        vecAlignThr = cosd(30);    % ~0.866

        vectors = fitEndpointVectors(BW, endPoints, 10);
        max_perh = 5;

    end

    if  count == 6
        fprintf('--> Vòng nối %d ----\n', count);

        % Tham số nối thử
        minCompSize = 5;
        maxDist     = 40;    % vòng sau cho phép nối xa hơn
        vecAlignThr = cosd(35);    % ~0.866
        max_perh = 5;

        vectors = fitEndpointVectors(BW, endPoints, 10);
    end

    if  count == 7
        fprintf('--> Vòng nối %d ----\n', count);

        % Tham số nối thử
        minCompSize = 5;
        maxDist     = 50;    % vòng sau cho phép nối xa hơn
        vecAlignThr = cosd(50);    % ~0.866

        vectors = fitEndpointVectors(BW, endPoints, 10);
        max_perh = 10;

    end
    if  count == 8
        fprintf('--> Vòng nối %d ----\n', count);

        % Tham số nối thử
        minCompSize = 5;
        maxDist     = 100;    % vòng sau cho phép nối xa hơn
        vecAlignThr = cosd(50);    % ~0.866

        vectors = fitEndpointVectors(BW, endPoints, 10);
        max_perh = 25;
    end

    [BW, linesConnected] = connectEndpoints_v3(BW, vectors, CC, minCompSize, maxDist, vecAlignThr, max_perh);

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

%% Nối vân ở biên
% --- Tìm endpoint ---
endPoints = bwmorph(BW,'endpoints');

vectors = fitEndpointVectors(BW, endPoints, 5);

margin = 15;
extendLength = 15;
% --- Tham số nối ---
BW = extendLineNearBorder(BW, vectors, extendLength, margin);

figure; imshow(BW,[]);
hold on; plot(vectors(:,1), vectors(:,2),'ro')
title("Nối vân ở biên");


end

save("after_estimate.mat");


% Hoàn thành


%% Hàm phụ trợ
%% --- FUNCTION MZS ---
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

%% ----- Hàm phụ trợ -----

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

%% helper: estimate local direction via PCA on nearest pixels in same component
function BW_out = extendLineNearBorder(BW, vectors, extendLen, margin)
% extendLineNearBorder - Nối dài endpoint ra ngoài NẾU nó gần biên ảnh
%
% Input:
%   BW        - ảnh nhị phân
%   vectors   - [cx, cy, vx, vy] cho mỗi endpoint
%   extendLen - số pixel muốn nối dài thêm
%   margin    - ngưỡng khoảng cách từ biên (ví dụ 5)
%
% Output:
%   BW_out    - ảnh nhị phân sau khi vẽ đoạn thẳng nối dài

[H,W] = size(BW);
BW_out = BW;

for i = 1:size(vectors,1)
    cx = vectors(i,1);
    cy = vectors(i,2);
    vx = vectors(i,3);
    vy = vectors(i,4);

    % --- CHỈ vẽ nếu endpoint gần biên ---
    if ~(cx <= margin || cx >= W-margin || cy <= margin || cy >= H-margin)
        continue; % bỏ qua nếu không gần biên
    end

    % Tính điểm mới (C = B + extendLen*v)
    x3 = cx + extendLen*vx;
    y3 = cy + extendLen*vy;

    % Bresenham từ (cx,cy) đến (x3,y3)
    [xLine, yLine] = bresenham2(round(cx), round(cy), round(x3), round(y3));

    % Loại pixel ngoài biên
    mask = xLine>=1 & xLine<=W & yLine>=1 & yLine<=H;
    xLine = xLine(mask);
    yLine = yLine(mask);

    % Vẽ vào ảnh
    BW_out(sub2ind([H,W], yLine, xLine)) = 1;
end

end

%% --- Hàm Bresenham ---
function [x,y] = bresenham2(x1,y1,x2,y2)
x1=round(x1); y1=round(y1);
x2=round(x2); y2=round(y2);

dx=abs(x2-x1); dy=abs(y2-y1);
sx=sign(x2-x1); sy=sign(y2-y1);

x=x1; y=y1;
xx=[]; yy=[];

if dx > dy
    err = dx/2;
    while x ~= x2
        xx(end+1)=x; yy(end+1)=y;
        x = x + sx;
        err = err - dy;
        if err < 0
            y = y + sy;
            err = err + dx;
        end
    end
else
    err = dy/2;
    while y ~= y2
        xx(end+1)=x; yy(end+1)=y;
        y = y + sy;
        err = err - dx;
        if err < 0
            x = x + sx;
            err = err + dy;
        end
    end
end
xx(end+1)=x2; yy(end+1)=y2;
x=xx; y=yy;
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