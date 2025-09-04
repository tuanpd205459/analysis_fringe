% === SCRIPT CHÍNH ĐỂ DEMO ===
% Chạy toàn bộ quá trình từ tạo hologram đến tái tạo pha tự động.

clear;
close all;
clc;

fprintf('=== Demo Tái tạo Pha từ Hologram với Lựa chọn Tự động ===\n');

% --- 1. Thiết lập tham số mô phỏng ---
M = 512; % Chiều cao ảnh
N = 512; % Chiều rộng ảnh

% Tần số sóng mang (carrier frequency)
% Đặt fy < 0 để đảm bảo phổ bậc +1 nằm ở nửa trên của miền tần số
fx = 40 / N; 
fy = -60 / M;

% Tạo một vật thể pha mẫu (sử dụng hàm 'peaks' của MATLAB)
[X, Y] = meshgrid(linspace(-3, 3, N), linspace(-3, 3, M));
phase_object = 2 * peaks(X, Y);
% phase_object = (phase_object - min(phase_object(:))) / (max(phase_object(:)) - min(phase_object(:))) * 2 * pi; % Chuẩn hóa về [0, 2*pi]

% --- 2. Tạo hologram mẫu ---
hologram = generate_test_hologram(M, N, fx, fy, phase_object);

% --- 3. Thiết lập tham số cho việc tái tạo ---
params.filter_radius = 25; % Bán kính bộ lọc có thể điều chỉnh
params.dc_suppression_radius = 20; % Bán kính vùng DC cần loại bỏ

% --- 4. Gọi hàm tái tạo pha tự động ---
fprintf('Bắt đầu tái tạo pha...\n');
[reconstructed_phase, debug_info] = reconstruct_phase_auto(hologram, params);
fprintf('Tái tạo hoàn tất.\n');

% --- 5. Hiển thị kết quả ---
figure('Name', 'Quá trình tái tạo pha tự động', 'Position', [100, 100, 1500, 700]);
sgtitle('Demo Tái tạo Pha từ Hologram', 'FontSize', 16, 'FontWeight', 'bold');

% Ảnh 1: Pha gốc (ground truth)
subplot(2, 3, 1);
imagesc(phase_object);
axis image; colorbar;
title('1. Vật thể Pha Gốc');
xlabel('x'); ylabel('y');

% Ảnh 2: Hologram đầu vào
subplot(2, 3, 2);
imagesc(hologram);
axis image; colormap(gca, 'gray');
title('2. Hologram Đầu vào');
xlabel('x'); ylabel('y');

% Ảnh 3: Phổ Fourier và vùng được chọn
subplot(2, 3, 3);
imagesc(log(1 + debug_info.spectrumMagnitude));
axis image; colormap(gca, 'parula');
hold on;
% Vẽ vòng tròn tại vị trí đã tìm thấy
theta = 0:0.01:2*pi;
x_circle = params.filter_radius * cos(theta) + debug_info.u_max;
y_circle = params.filter_radius * sin(theta) + debug_info.v_max;
plot(x_circle, y_circle, 'g', 'LineWidth', 2);
plot(debug_info.u_max, debug_info.v_max, 'g+', 'MarkerSize', 10, 'LineWidth', 2);
hold off;
title({'3. Phổ Fourier', ['Đỉnh tự động chọn tại (', num2str(debug_info.u_max), ', ', num2str(debug_info.v_max), ')']});
xlabel('Tần số u'); ylabel('Tần số v');

% Ảnh 4: Phổ sau khi lọc và dịch chuyển
subplot(2, 3, 4);
imagesc(log(1 + abs(debug_info.filteredSpectrum)));
axis image;
title('4. Phổ Bậc +1 (Sau khi lọc và dịch tâm)');
xlabel('Tần số u'); ylabel('Tần số v');

% Ảnh 5: Biên độ tái tạo
subplot(2, 3, 5);
surf(reconstructed_phase,"EdgeColor","none");
 colorbar;
title('5. Ảnh 3D');
xlabel('x'); ylabel('y');

% Ảnh 6: Pha tái tạo
subplot(2, 3, 6);
imagesc(reconstructed_phase);
axis image; colorbar;
title('6. Pha Tái tạo (Wrapped)');
xlabel('x'); ylabel('y');


% === CÁC HÀM HỖ TRỢ ===

function [wrappedPhase, debug_info, params] = reconstruct_phase_auto(hologram, params)
% Tái tạo pha từ hologram bằng cách lọc trong miền tần số với lựa chọn tự động.
%
% Chức năng sẽ tự động tìm phổ bậc +1 ở nửa trên của miền tần số,
% tạo một bộ lọc tròn và tiến hành tái tạo pha.
%
% Chức năng này được thiết kế để không tự vẽ hình, thay vào đó trả về
% thông tin trong 'debug_info' để script chính xử lý việc hiển thị.

    % --- Kiểm tra và đặt giá trị mặc định cho params ---
    if ~exist('params', 'var')
        params = struct();
    end
    if ~isfield(params, 'filter_radius')
        params.filter_radius = 50; % Bán kính của bộ lọc tròn
    end
    if ~isfield(params, 'dc_suppression_radius')
        params.dc_suppression_radius = 25; % Bán kính vùng trung tâm để loại bỏ
    end
    
    % --- Xử lý ban đầu ---
    if size(hologram, 3) > 1
        hologramGray = rgb2gray(hologram);
    else
        hologramGray = hologram;
    end
    
    [numRows, numCols] = size(hologramGray);
    fourierTransform = fftshift(fft2(hologramGray));
    spectrumMagnitude = abs(fourierTransform);
    
    % --- Tự động tìm kiếm phổ bậc +1 ---
    
    % Tọa độ tâm của phổ
    u0 = floor(numCols / 2) + 1;
    v0 = floor(numRows / 2) + 1;
    
    % Tạo một bản sao của phổ cường độ để tìm kiếm
    searchSpectrum = spectrumMagnitude;
    
    % Loại bỏ thành phần DC (bậc 0) để tránh chọn nhầm.
    % Tạo một mask tròn ở tâm và đặt giá trị trong vùng đó bằng 0.
    [U, V] = meshgrid(1:numCols, 1:numRows);
    dist_from_center = sqrt((U - u0).^2 + (V - v0).^2);
    searchSpectrum(dist_from_center <= params.dc_suppression_radius) = 0;
    
    % Chỉ tìm kiếm ở nửa trên của phổ (nơi thường chứa phổ bậc +1)
    upperHalfSpectrum = searchSpectrum(1:v0-1, :);
    
    % Tìm tọa độ của điểm có cường độ lớn nhất
    [~, maxIdx] = max(upperHalfSpectrum(:));
    [v_max, u_max] = ind2sub(size(upperHalfSpectrum), maxIdx);
    % (v_max, u_max) là tọa độ của tâm vùng ROI được chọn tự động
    
    % --- Tạo bộ lọc và trích xuất phổ ---
    
    % Tạo một bộ lọc (mask) hình tròn tại vị trí (u_max, v_max)
    % Meshgrid (U, V) đã được tạo ở trên
    roi_mask = sqrt((U - u_max).^2 + (V - v_max).^2) <= params.filter_radius;
    
    % Áp dụng mask để chỉ giữ lại phổ bậc +1
    filteredContent = fourierTransform .* roi_mask;
    
    % Dịch chuyển vùng phổ đã chọn về lại tâm của ma trận
    % Tính toán độ dịch chuyển cần thiết
    v_shift = v0 - v_max;
    u_shift = u0 - u_max;
    
    % Dùng circshift để dịch chuyển.
    filteredSpectrum = circshift(filteredContent, [v_shift, u_shift]);
    
    % --- Tái tạo trường sóng phức và lấy pha ---
    finalPhaseComplex = ifft2(ifftshift(filteredSpectrum));
    
    % Lấy pha từ trường phức (kết quả là pha bị Wrapped trong khoảng [-pi, pi])
    wrappedPhase = angle(finalPhaseComplex);

    % --- Gói thông tin gỡ lỗi để trả về ---
    debug_info.spectrumMagnitude = spectrumMagnitude;
    debug_info.u_max = u_max;
    debug_info.v_max = v_max;
    debug_info.filteredSpectrum = filteredSpectrum;
    debug_info.reconstructedAmplitude = abs(finalPhaseComplex);
end


function hologram = generate_test_hologram(M, N, fx, fy, phase_object)
% Tạo ra một hologram nhiễu xạ Fresnel đơn giản.
%
% Input:
%   M, N: Kích thước của hologram
%   fx, fy: Tần số sóng mang theo hai chiều x và y
%   phase_object: Ma trận 2D đại diện cho pha của vật thể
%
% Output:
%   hologram: Ma trận 2D của hologram được tạo ra

    [X, Y] = meshgrid(1:N, 1:M);
    
    % Cường độ nền và điều biến
    a = 1.0; % Background intensity
    b = 0.8; % Modulation depth
    
    % Sóng mang phẳng (plane wave carrier)
    carrier = 2 * pi * (fx * X + fy * Y);
    
    % Công thức tạo hologram
    % g = a + b * cos(sóng_mang + pha_vật)
    hologram = a + b .* cos(carrier + phase_object);
    
    % Thêm một chút nhiễu Gaussian để thực tế hơn
    hologram = hologram + 0 * randn(M, N);
end