clc; clear; close all;

% --- Skeleton giả định (test) ---
%% 1. KHỞI TẠO
clc, clear, close all;
fprintf('Bắt đầu quy trình mô phỏng và tái tạo...\n');

%% 2. MÔ PHỎNG HOLOGRAM
fprintf('--> Bước 1: Mô phỏng Hologram...\n');
% --- Thiết lập thông số ---
M = 512; % Kích thước ảnh (chiều cao)
N = 512; % Kích thước ảnh (chiều rộng)
snr = 15;
fx = 40 / N; % Tần số sóng mang
fy = -60 / M;

auto_fft = 0;


% nhiễu - phương sai: sigma
sigma = pi/5;
noise_level = 0;
noise = noise_level * randn(N, N) .* sigma;


[X, Y] = meshgrid(linspace(-1,1,N), linspace(-1,1,M));
object_phase_without_noise = 2 * peaks(3*X, 3*Y);


%%
% Thêm nhiễu vào pha đối tượng
object_phase = awgn(object_phase_without_noise, snr, 'measured', 'db');
figure;
surf(object_phase,"EdgeColor","none");
title("doi tuong co nhieu- groundtruth");
% --- Hiển thị pha gốc (không nhiễu) ---
figure;
surf(object_phase_without_noise, "EdgeColor", "none");
colorbar;
title('Đối tượng pha (không nhiễu): ');

% 3. TẠO HOLOGRAM
fprintf('--> Bước 2: Tạo Hologram...\n');

% --- Thiết lập thông số sóng mang ---
fx = 40 / N; % Tần số sóng mang
fy = -60 / M;
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
hologram = medfilt2(hologram, [3 3]);
hologram = wiener2(hologram, [5 5]);
figure;
imshow(hologram);
colorbar;
title('hologram sau noise removal : ');
%% 6. Histogram equalization
hologram = adapthisteq(hologram);
figure;
imshow(hologram);
colorbar;
title('hologram sau equaliztion histogram : ');

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
close all;
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
BW =skeleton;
fprintf('Running Modified ZS (MZS) thinning...\n');
S = MZS_thinning(BW);

figure;
subplot(1,2,1); imshow(BW); title('Input binary');
subplot(1,2,2); imshow(S);  title('Skeleton (MZS)');



%%
BW = S;
figure; imshow(BW); title('Skeleton gốc');
kernel = ones(3,3);
kernel(2,2) = 0;
neighborCount = conv2(double(BW), kernel, 'same');
% Junction: pixel skeleton có >= 3 hàng xóm
junction = (BW == 1) & (neighborCount >= 3);
% Không xét biên
% Loại bỏ toàn bộ biên 4 cạnh
junction(1:4,:)   = 0;   % dòng đầu
junction(end-4:end,:) = 0;   % dòng cuối
junction(:,1:4)   = 0;   % cột đầu
junction(:,end-4:end) = 0;   % cột cuối


[row, col] = find(junction);
hold on; plot(col, row, 'go', 'MarkerSize', 10, 'LineWidth', 1);

BW(junction) = 0; %xoá junction

figure;
imshow(BW); title('sau khi xoa junction');
figure; imshow(BW); title('sau khi xoa junctionnn');
[row, col] = find(junction);
hold on; plot(col, row, 'go', 'MarkerSize', 10, 'LineWidth', 1);
%%
BW_clean = removeSmallComponents(BW, 5);  % xoá vùng liên thông < 10 pixel

figure;
subplot(1,2,1); imshow(BW);        title('Skeleton gốc');
subplot(1,2,2); imshow(BW_clean); title('Skeleton sau khi xoá vùng nhỏ');


%% Tìm endpoint
BW = BW_clean;


%% 4. ƯỚC LƯỢNG VECTOR HƯỚNG TẠI ENDPOINTS (theo component 30 pixel)
n = 2;
for count =1:n 
    % Timf endPoint
    % Kernel để đếm số hàng xóm (8-neighbors)
kernel = ones(3,3);
kernel(2,2) = 0;
neighborCount = conv2(double(BW), kernel, 'same');

% Endpoint: pixel skeleton có đúng 1 hàng xóm
endPoints = (BW == 1) & (neighborCount == 1);

% Không xét biên
endPoints(1:3,:)   = 0;
endPoints(end-3:end,:) = 0;
endPoints(:,1:3)   = 0;
endPoints(:,end-3:end) = 0;


figure; imshow(BW); title('Skeleton gốc');

[row, col] = find(endPoints);
hold on; plot(col, row, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
%
fprintf('--> Bước 2b: Ước lượng vector hướng theo đoạn liên thông\n');

Nfit = 30; % số pixel để fit
[y_idx, x_idx] = find(endPoints);  % toạ độ endpoint

% Tìm các component trong mask hợp lệ
CC = bwconncomp(BW, 8);

vectors = [];

for k = 1:length(x_idx)
    cx = x_idx(k); cy = y_idx(k);

    % Kiểm tra endpoint thuộc component nào
    comp_id = 0;
    for c = 1:CC.NumObjects
        if ismember(sub2ind(size(BW), cy, cx), CC.PixelIdxList{c})
            comp_id = c; break;
        end
    end
    
    if comp_id == 0, continue; end  % endpoint không thuộc component nào

    % Lấy toạ độ tất cả điểm trong component
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
        c = centroid - [cx; cy];  % vector từ endpoint đến trong component
        if dot(v, c) > 0
            v = -v; % đảo dấu để hướng ra ngoài
        end
    else
        v = [0;0];
    end

    vectors = [vectors; cx cy v(1) v(2)];
end

% --- hiển thị kết quả ---
figure; imshow(BW,[]); hold on;
quiver(vectors(:,1), vectors(:,2), 15*vectors(:,3), 15*vectors(:,4), ...
       0, 'g','LineWidth',0.5,'MaxHeadSize',0.5);
title('Vectors hướng tại endpoints (ra ngoài)');

% 
% % Hiển thị kết quả
% figure; imshow(BW,[]); hold on;
% plot(x_idx, y_idx,'ro','MarkerSize',6,'LineWidth',1.5); % endpoint
% quiver(vectors(:,1), vectors(:,2), 10*vectors(:,3), 10*vectors(:,4), ...
%        'y','LineWidth',1.5); % vector hướng
% title('Endpoint (đỏ) với vector hướng nội suy từ 30 pixel trong component');

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
if count == 3 % Nối vân ngắn + góc lệch nhỏ
            fprintf('--> Bước 2d: Nối các endpoint theo vector hướng\n');
            fprintf('--> lan chay lan 3\n');

            minCompSize = 12;   % chỉ nối nếu component đủ dài
            maxDist     = 25;   % khoảng cách tối đa giữa 2 endpoint
            vecAlignThr = cosd(15);  % = 0.866 ~ hướng lệch <= 30°
end
if count == 4 % Nối vân ngắn + góc lệch lớn
            fprintf('--> Bước 2d: Nối các endpoint theo vector hướng\n');
            fprintf('--> lan chay lan 4\n');

            minCompSize = 5;   % chỉ nối nếu component đủ dài
            maxDist     = 20;   % khoảng cách tối đa giữa 2 endpoint
            vecAlignThr = cosd(15);  % = 0.866 ~ hướng lệch <= 30°
end

BW_new = BW; % copy để cập nhật nối

% --- Lặp qua các cặp endpoint ---
for i = 1:size(vectors,1)-1
    cx1 = vectors(i,1); cy1 = vectors(i,2);
    v1  = [vectors(i,3), vectors(i,4)];
    
    % kiểm tra component của endpoint i
    comp_id1 = 0;
    for c = 1:CC.NumObjects
        if ismember(sub2ind(size(BW), cy1, cx1), CC.PixelIdxList{c})
            comp_id1 = c; break;
        end
    end
    if comp_id1==0 || numel(CC.PixelIdxList{comp_id1}) < minCompSize
        continue;
    end
    
    for j = i+1:size(vectors,1)
        cx2 = vectors(j,1); cy2 = vectors(j,2);
        v2  = [vectors(j,3), vectors(j,4)];

        % kiểm tra component j
        comp_id2 = 0;
        for c = 1:CC.NumObjects
            if ismember(sub2ind(size(BW), cy2, cx2), CC.PixelIdxList{c})
                comp_id2 = c; break;
            end
        end
        if comp_id2==0 || numel(CC.PixelIdxList{comp_id2}) < minCompSize
            continue;
        end
        
        % --- kiểm tra khoảng cách ---
        d = sqrt((cx1-cx2)^2 + (cy1-cy2)^2);
        if d > maxDist, continue; end

        % --- kiểm tra hướng đối nhau ---
        dir12 = [cx2-cx1, cy2-cy1];
        dir12 = dir12 / norm(dir12+eps);
       if dot(v1, dir12) > vecAlignThr && dot(v2, -dir12) > vecAlignThr
            % nối 2 điểm bằng line
            BW_new = drawLine(BW_new, cx1, cy1, cx2, cy2);

        end
    end
end

% --- hiển thị ---
figure; imshow(BW_new,[]); hold on;
plot(vectors(:,1), vectors(:,2), 'ro');
title(sprintf('Kết quả sau khi nối endpoints - lần thứ %d', count));
% quiver(vectors(:,1), vectors(:,2), 15*vectors(:,3), 15*vectors(:,4), ...
%        0, 'g','LineWidth',0.5,'MaxHeadSize',0.5);
%% Tìm các endpoint mới
BW = BW_new;
end

kernel = ones(3,3);
kernel(2,2) = 0;
neighborCount = conv2(double(BW), kernel, 'same');

% Endpoint: pixel skeleton có đúng 1 hàng xóm
endPoints = (BW == 1) & (neighborCount == 1);

% Không xét biên
endPoints(1:3,:)   = 0;
endPoints(end-3:end,:) = 0;
endPoints(:,1:3)   = 0;
endPoints(:,end-3:end) = 0;


figure; imshow(BW); title('Skeleton gốc');

[row, col] = find(endPoints);
hold on; plot(col, row, 'ro', 'MarkerSize', 10, 'LineWidth', 2);

%% Nối tiếp
minLen   = 30;   % ngưỡng chiều dài vân
probeLen = 20;   % độ dài đoạn line giả định

BW_new = BW;

for i = 1:size(vectors,1)
    cx = vectors(i,1); cy = vectors(i,2);
    v  = [vectors(i,3), vectors(i,4)];

    % tìm component chứa endpoint này
    comp_id = 0;
    for c = 1:CC.NumObjects
        if ismember(sub2ind(size(BW), cy, cx), CC.PixelIdxList{c})
            comp_id = c; break;
        end
    end
    if comp_id==0 || numel(CC.PixelIdxList{comp_id}) < minLen
        continue; % bỏ qua vân ngắn
    end

    % tạo điểm "giả định" cách endpoint 10 pixel theo hướng vector
    x2 = round(cx + probeLen * v(1));
    y2 = round(cy + probeLen * v(2));

    % ép toạ độ nằm trong khung ảnh
    x2 = max(1, min(size(BW,2), x2));
    y2 = max(1, min(size(BW,1), y2));

    % vẽ đoạn line 10 pixel
    probeLine = drawLine(false(size(BW)), cx, cy, x2, y2);

    % kiểm tra xem line có giao vân nào khác không
    overlap = probeLine & BW;
    overlap(cy, cx) = 0; % loại endpoint gốc
    if any(overlap(:))
        % nếu có giao, nối endpoint tới điểm giao đầu tiên
        [ry, cx_] = find(overlap, 1, 'first');
        BW_new = drawLine(BW_new, cx, cy, cx_, ry);
    end
end

figure; imshow(BW_new,[]);
title('Kết quả nối endpoint > 50 pixels bằng line giả định 10 px');























% Kết thúc
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

    % Không xét biên của ảnh gốc
    marker(1:3,:)   = 0;
    marker(end-3:end,:) = 0;
    marker(:,1:3)   = 0;
    marker(:,end-3:end) = 0;
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
    BW_clean = bwmorph(BW_clean, 'spur', 1);  % loại bỏ các nhánh nhỏ lẻ

end
function BW_lines = connect_endpoints_fringe(BW_edge, BW_endpoints, N, maxDist)
% Kết nối endpoints: ưu tiên khoảng cách gần và hướng đối nhau
% BW_edge: ảnh nhị phân cạnh vân
% BW_endpoints: ảnh nhị phân, pixel=1 là endpoint
% N: số điểm lân cận để fit vector hướng (vd N=30)
% maxDist: ngưỡng tối đa khoảng cách để nối (vd 80)

if nargin < 3, N = 30; end
if nargin < 4, maxDist = 80; end

[H,W] = size(BW_edge);
BW_lines = false(H,W);

% --- B0. Lấy tọa độ endpoints từ ảnh nhị phân
[y,x] = find(BW_endpoints);
endpoints = [y,x];
nPts = size(endpoints,1);

vectors = zeros(nPts,2);

%% B1. Xây dựng vector hướng cho từng endpoint
for k = 1:nPts
    y0 = endpoints(k,1); x0 = endpoints(k,2);

    % tìm N pixel lân cận thuộc cạnh vân
    [yy,xx] = find(BW_edge);
    d = hypot(xx-x0, yy-y0);
    [~,idx] = sort(d);
    idx = idx(2:min(N+1,length(idx))); % bỏ chính nó
    X = xx(idx); Y = yy(idx);

    if numel(X) < 2
        vectors(k,:) = [1,0]; % fallback
    else
        % fit đường thẳng y = ax+b
        p = polyfit(X,Y,1);
        v = [1, p(1)];
        v = v / norm(v);
        vectors(k,:) = v;
    end
end

%% B2. Ghép cặp endpoint
used = false(nPts,1);

for i = 1:nPts
    if used(i), continue; end
    xi = endpoints(i,2); yi = endpoints(i,1);
    vi = vectors(i,:);

    best_j = -1;
    best_dist = Inf;

    for j = 1:nPts
        if i==j || used(j), continue; end
        xj = endpoints(j,2); yj = endpoints(j,1);
        vj = vectors(j,:);

        % khoảng cách
        dist = hypot(xi-xj, yi-yj);

        % hướng có đối nhau không? (dot < 0)
        if dist < maxDist && dot(vi,vj) < 0
            if dist < best_dist
                best_dist = dist;
                best_j = j;
            end
        end
    end

    % Nếu tìm thấy cặp hợp lệ thì nối
    if best_j > 0
        line_pixels = bresenham_line(xi,yi,endpoints(best_j,2),endpoints(best_j,1));
        valid = line_pixels(:,1)>=1 & line_pixels(:,1)<=W & ...
                line_pixels(:,2)>=1 & line_pixels(:,2)<=H;
        lp = line_pixels(valid,:);
        ind = sub2ind([H,W], lp(:,2), lp(:,1));
        BW_lines(ind) = true;

        used(i) = true;
        used(best_j) = true;
    end
end

end

%% ------------------
% Hàm con: Bresenham line
function pts = bresenham_line(x1,y1,x2,y2)
x1 = round(x1); y1 = round(y1); x2 = round(x2); y2 = round(y2);
dx = abs(x2 - x1); dy = abs(y2 - y1);
sx = sign(x2 - x1); sy = sign(y2 - y1);
if dy <= dx
    err = dx/2; y = y1;
    pts = zeros(dx+1,2); k=1;
    for x = x1:sx:x2
        pts(k,:) = [x,y]; k=k+1;
        err = err - dy;
        if err < 0, y = y + sy; err = err + dx; end
    end
else
    err = dy/2; x = x1;
    pts = zeros(dy+1,2); k=1;
    for y = y1:sy:y2
        pts(k,:) = [x,y]; k=k+1;
        err = err - dx;
        if err < 0, x = x + sx; err = err + dy; end
    end
end
end

function BW = drawLine(BW, x1, y1, x2, y2)
    % Vẽ đoạn thẳng từ (x1,y1) -> (x2,y2) trên ảnh BW nhị phân
    [rr, cc] = bresenham(y1, x1, y2, x2); 
    idx = sub2ind(size(BW), rr, cc);
    BW(idx) = 1;
end

function [rr,cc] = bresenham(y1, x1, y2, x2)
    % Thuật toán Bresenham (integer line rasterization)
    x1=round(x1); x2=round(x2);
    y1=round(y1); y2=round(y2);
    dx = abs(x2-x1);
    dy = abs(y2-y1);
    steep = abs(dy)>abs(dx);
    if steep
        [x1,y1] = deal(y1,x1);
        [x2,y2] = deal(y2,x2);
        [dx,dy] = deal(dy,dx);
    end
    if x1 > x2
        [x1,x2] = deal(x2,x1);
        [y1,y2] = deal(y2,y1);
    end
    derr = 2*dy; err = 0;
    y = y1;
    if y2>y1, ystep=1; else, ystep=-1; end
    rr=[]; cc=[];
    for x=x1:x2
        if steep
            rr=[rr; x]; cc=[cc; y];
        else
            rr=[rr; y]; cc=[cc; x];
        end
        err = err + derr;
        if err > dx
            y = y + ystep;
            err = err - 2*dx;
        end
    end
end
