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

