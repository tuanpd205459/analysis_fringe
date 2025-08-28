clc; clear; close all;
fprintf('=== MÔ PHỎNG & TÁI TẠO PHA (Step 1 & 2) ===\n');

%% 1. KHỞI TẠO THÔNG SỐ & ĐỐI TƯỢNG PHA
M = 512; N = 512;              % kích thước ảnh
snr = 100;                      % SNR thêm nhiễu

[X, Y] = meshgrid(linspace(-1,1,N), linspace(-1,1,M));
phi_true = 2 * peaks(3*X, 3*Y);            % pha gốc (groundtruth)
object_phase = awgn(phi_true, snr, 'measured'); % thêm nhiễu
wrapped_phase = wrapToPi(object_phase);    % pha quấn [-pi,pi]

% Hiển thị đối tượng
figure; surf(object_phase,"EdgeColor","none"); title("Đối tượng pha (có nhiễu)");
figure; surf(wrapped_phase,"EdgeColor","none"); title("Wrapped phase");

% Scale 0–255 giống trong bài báo
G_gray = uint8(255 * (wrapped_phase - min(wrapped_phase(:))) / (2*pi));

%% 2. FRINGE EDGE DETECTION (Eq.1)
fprintf('--> Bước 1: Phát hiện biên fringe (Eq.1)\n');

% Đạo hàm theo x và y
Ex = double(G_gray(:,1:end-1)) - double(G_gray(:,2:end));
Ey = double(G_gray(1:end-1,:)) - double(G_gray(2:end,:));

% Hàm biên E(x,y)
E = zeros(size(G_gray));
E(:,1:end-1)   = E(:,1:end-1)   + Ex.^2;
E(1:end-1,:)   = E(1:end-1,:)   + Ey.^2;
E = sqrt(E);

% Ngưỡng (150–220 theo bài báo)
pass_mark = 200;
edge_map = E > pass_mark;

figure; imshow(edge_map,[]); title('Fringe edges');

%% 3. ENDPOINT DETECTION (Eq.2)
fprintf('--> Bước 2: Xác định endpoint (Eq.2)\n');

% Đếm số láng giềng 8-connect
kernel = ones(3); kernel(2,2) = 0;
neighbor_count = conv2(double(edge_map), kernel, 'same');

% Endpoint nếu weight = 1 hoặc 2
endpoints = edge_map & (neighbor_count <= 1);

figure; imshow(endpoints,[]); title('Detected endpoints');

%% 4. BROKEN FRINGE ESTIMATION (Vector hướng LS fitting)
fprintf('--> Bước 2b: Ước lượng vector hướng tại endpoints\n');

Nfit = 30; % số điểm edge lân cận để fit
[y_idx, x_idx] = find(endpoints);
vectors = [];

for k = 1:length(x_idx)
    cx = x_idx(k); cy = y_idx(k);

    % Lấy N điểm edge gần nhất
    [yy,xx] = find(edge_map);
    dist2 = (xx-cx).^2 + (yy-cy).^2;
    [~,idx] = sort(dist2);
    idxN = idx(1:min(Nfit,end));

    X = xx(idxN); Y = yy(idxN);

    % Fit đường thẳng y = ax + b (least squares)
    p = polyfit(X,Y,1);
    a = p(1);
    v = [1 a]; v = v/norm(v); % vector hướng chuẩn hóa

    vectors = [vectors; cx cy v];
end

% Hiển thị kết quả
figure; imshow(edge_map,[]); hold on;
plot(x_idx, y_idx,'ro','MarkerSize',6,'LineWidth',1.5);
quiver(vectors(:,1), vectors(:,2), 10*vectors(:,3), 10*vectors(:,4), ...
       'y','LineWidth',1.5);
title('Endpoints với vector hướng');
