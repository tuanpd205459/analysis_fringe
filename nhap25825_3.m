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

% --- Tham số ---
win = 20; half = floor(win/2);
[rows, cols] = size(skeleton);
skeleton_refined = skeleton;



figure; imshow(skeleton); title('Skeleton gốc');
% --- Hiển thị kết quả ---


% --- Vòng lặp kiểm tra từng điểm skeleton ---
for i = 1+half:rows-half
    for j = 1+half:cols-half
        if skeleton(i,j) == 1
            % Lấy cửa sổ quanh điểm (i,j)
            W = skeleton(i-half:i+half, j-half:j+half);
            [yy, xx] = find(W);
            if numel(xx) < 10, continue; end
            
            % PCA để tìm hướng chính
            xx = xx - mean(xx); yy = yy - mean(yy);
            C = cov([xx yy]);
            [V,D] = eig(C);
            [~,idx] = max(diag(D));
            v = V(:,idx);
            theta_OF = atan2(v(2), v(1));
            
            % Hướng skeleton cục bộ (neighbor)
            [yn,xn] = find(skeleton(i-1:i+1, j-1:j+1));
            xn = xn-2; yn = yn-2;
            idx2 = (xn~=0 | yn~=0);
            if any(idx2)
                theta_sk = atan2(mean(yn(idx2)), mean(xn(idx2)));
                dtheta = abs(mod(theta_OF-theta_sk+pi,2*pi)-pi);
                if dtheta > pi/2   % lệch > 70°
                    skeleton_refined(i,j) = 0;
                end
            end
        end
    end
end

% --- Hiển thị kết quả ---
figure;
imshowpair(skeleton, skeleton_refined, 'montage');
title('Trái: Skeleton gốc | Phải: Sau refine bằng Orientation Field');


