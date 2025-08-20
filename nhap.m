% === SCRIPT CHÍNH ĐỂ DEMO (PHIÊN BẢN LINH HOẠT) ===
clear;
close all;
clc;
fprintf('=== Demo Tái tạo Pha Linh hoạt ===\n');

% --- 1. CÁC TÙY CHỌN ĐIỀU KHIỂN ---
% Thay đổi các giá trị true/false ở đây để thử nghiệm
params.enable_zp = false;               % true = Bật Zero Padding, false = Tắt
params.enable_ramp_correction = false;  % true = Bật bù trừ pha (sửa lỗi nghiêng), false = Tắt

% --- 2. Thiết lập tham số mô phỏng ---
params.zp_factor = 8;      % Hệ số Zero Padding (chỉ có tác dụng khi enable_zp = true)
params.filter_radius = 25; % Bán kính bộ lọc 
params.dc_suppression_radius = 40; % Bán kính vùng DC cần loại bỏ

M = 512; % Chiều cao ảnh
N = 512; % Chiều rộng ảnh
% Tần số sóng mang (dùng số lẻ để thấy rõ ưu điểm của ZP và bù trừ pha)
fx = 40  / N; 
fy = -60 / M; 

% Tạo một vật thể pha mẫu (không chuẩn hóa để thấy phase wrapping)
[X, Y] = meshgrid(linspace(-3, 3, N), linspace(-3, 3, M));
phase_object = 5 * peaks(X, Y);

% Tạo hologram mẫu
hologram = generate_test_hologram(M, N, fx, fy, phase_object);

% --- 3. Gọi hàm tái tạo pha linh hoạt ---
[reconstructed_phase, debug_info] = reconstruct_phase_flexible(hologram, M, N, params);
fprintf('Tái tạo hoàn tất.\n');

% --- 4. Hiển thị kết quả ---
figure('Name', 'Quá trình tái tạo pha', 'Position', [50, 50, 1600, 800]);
sgtitle('Demo Tái tạo Pha', 'FontSize', 16, 'FontWeight', 'bold');

% Cột 1: Dữ liệu gốc
subplot(2, 4, 1);
surf(phase_object, "EdgeColor", "none"); colorbar;
title('1. Vật thể Pha Gốc (dạng 3D)');
subplot(2, 4, 5);
imagesc(hologram); axis image; colormap(gca, 'gray'); title('2. Hologram Đầu vào');

% Cột 2: Phân tích phổ
subplot(2, 4, 2);
imagesc(log(1 + debug_info.spectrumMagnitude)); axis image; 
title({'3. Phổ Gốc'});
subplot(2, 4, 6);
if isfield(debug_info, 'spectrumMagnitude_zp')
    imagesc(log(1 + debug_info.spectrumMagnitude_zp)); axis image;
    hold on;
    plot(debug_info.u_max_zp, debug_info.v_max_zp, 'r+', 'MarkerSize', 10, 'LineWidth', 2);
    hold off;
    title({'4. Phổ Zero-Padded', 'và Đỉnh được phát hiện'});
    legend('Đỉnh chính xác');
else
    title('4. Phổ Gốc (Không ZP)');
end

% Cột 3: Lọc và tái tạo
subplot(2, 4, 3);
imagesc(log(1 + abs(debug_info.filteredSpectrum))); axis image;
title('5. Phổ Bậc +1 (Đã lọc và dịch tâm)');
subplot(2, 4, 7);
imagesc(debug_info.reconstructedAmplitude); axis image; colorbar;
title('6. Biên độ Tái tạo');

% Cột 4: Kết quả cuối cùng
subplot(2, 4, 4);
surf(reconstructed_phase, "EdgeColor", "none"); 
colorbar;
title('7. Pha Tái tạo (dạng 3D)');

subplot(2, 4, 8);
phase_error = reconstructed_phase - phase_object;
phase_error = phase_error - mean(phase_error(:)); % Bỏ offset
imagesc(phase_error); axis image; colorbar;
title(sprintf('8. Sai số Pha (RMS: %.4f rad)', sqrt(mean(phase_error(:).^2))));


% === CÁC HÀM HỖ TRỢ ===

function [final_phase, debug_info] = reconstruct_phase_flexible(hologram, M, N, params)
% Tái tạo pha từ hologram với các tùy chọn linh hoạt.
% Logic chính được giữ nguyên theo yêu cầu của bạn.

    % --- Xử lý tùy chọn ---
    if ~isfield(params, 'enable_zp'), params.enable_zp = false; end
    if ~isfield(params, 'enable_ramp_correction'), params.enable_ramp_correction = false; end
    if ~isfield(params, 'zp_factor'), params.zp_factor = 1; end
    if ~isfield(params, 'filter_radius'), params.filter_radius = 50; end
    if ~isfield(params, 'dc_suppression_radius'), params.dc_suppression_radius = 25; end

    % Quyết định hệ số ZP sẽ sử dụng
    if params.enable_zp
        current_zp_factor = params.zp_factor;
        fprintf('Chế độ: Bật Zero Padding (hệ số %d).\n', current_zp_factor);
    else
        current_zp_factor = 1;
        fprintf('Chế độ: Tắt Zero Padding (hệ số = 1).\n');
    end

    % --- Xử lý ban đầu ---
    if size(hologram, 3) > 1, hologramGray = rgb2gray(hologram); else, hologramGray = hologram; end
    
    % --- Bước 1: Dùng Zero Padding để tìm đỉnh chính xác ---
    zp_Rows = M * current_zp_factor;
    zp_Cols = N * current_zp_factor;
    
    hologram_zp = zeros(zp_Rows, zp_Cols, 'like', hologramGray);
    hologram_zp(1:M, 1:N) = hologramGray;
    
    ft_zp = fftshift(fft2(hologram_zp));
    spec_zp_mag = abs(ft_zp);
    
    u0_zp = floor(zp_Cols / 2) + 1;
    v0_zp = floor(zp_Rows / 2) + 1;
    
    searchSpectrum_zp = spec_zp_mag;
    [U_zp, V_zp] = meshgrid(1:zp_Cols, 1:zp_Rows);
    dist_from_center_zp = sqrt((U_zp - u0_zp).^2 + (V_zp - v0_zp).^2);
    dc_radius_zp = params.dc_suppression_radius * current_zp_factor;
    searchSpectrum_zp(dist_from_center_zp <= dc_radius_zp) = 0;
    
    upperHalf_zp = searchSpectrum_zp(1:v0_zp-1, :);
    [~, maxIdx] = max(upperHalf_zp(:));
    [v_max_zp, u_max_zp] = ind2sub(size(upperHalf_zp), maxIdx);
    
    % --- Bước 2: Quy đổi tọa độ đỉnh về phổ gốc ---
    u0 = floor(N / 2) + 1;
    v0 = floor(M / 2) + 1;
    u_offset_zp = u_max_zp - u0_zp;
    v_offset_zp = v_max_zp - v0_zp;
    u_offset = u_offset_zp / current_zp_factor;
    v_offset = v_offset_zp / current_zp_factor;
    u_max_precise = u0 + u_offset;
    v_max_precise = v0 + v_offset;
    
    % --- Bước 3: Lọc và dịch chuyển trên phổ gốc ---
    fourierTransform = fftshift(fft2(hologramGray));
    [U, V] = meshgrid(1:N, 1:M);
    roi_mask = sqrt((U - u_max_precise).^2 + (V - v_max_precise).^2) <= params.filter_radius;
    filteredContent = fourierTransform .* roi_mask;
    
    % --- Bước 4: Xử lý dịch chuyển ---
    v_shift = v0 - v_max_precise;
    u_shift = u0 - u_max_precise;
    v_shift_int = round(v_shift);
    u_shift_int = round(u_shift);
    shiftedContent_int = circshift(filteredContent, [v_shift_int, u_shift_int]);
    
    % --- Bước 5: Tái tạo và tùy chọn bù trừ pha ---
    finalPhaseComplex = ifft2(ifftshift(shiftedContent_int));
    phase_before_correction = angle(finalPhaseComplex);
    
    if params.enable_ramp_correction
        fprintf('Thực hiện bù trừ pha dốc (Ramp Correction).\n');
        v_shift_frac = v_shift - v_shift_int;
        u_shift_frac = u_shift - u_shift_int;
        [x, y] = meshgrid(0:N-1, 0:M-1);
        % Lưu ý: Dấu trừ là đúng để bù lại pha do dịch chuyển
        residual_phase_ramp = 2 * pi * (u_shift_frac * x / N + v_shift_frac * y / M);
        final_phase = phase_before_correction - residual_phase_ramp;
    else
        fprintf('Bỏ qua bù trừ pha dốc (Ramp Correction).\n');
        final_phase = phase_before_correction;
    end
    
    % --- Gói thông tin gỡ lỗi để trả về ---
    debug_info.spectrumMagnitude = abs(fourierTransform);
    if params.enable_zp
        debug_info.spectrumMagnitude_zp = spec_zp_mag;
        debug_info.u_max_zp = u_max_zp;
        debug_info.v_max_zp = v_max_zp;
    end
    debug_info.filteredSpectrum = shiftedContent_int;
    debug_info.reconstructedAmplitude = abs(finalPhaseComplex);
end

function hologram = generate_test_hologram(M, N, fx, fy, phase_object)
% Tạo ra một hologram nhiễu xạ Fresnel đơn giản.
    [X, Y] = meshgrid(1:N, 1:M);
    a = 1.0; b = 0.8; 
    carrier = 2 * pi * (fx * X + fy * Y);
    hologram = a + b .* cos(carrier + phase_object);
    hologram = hologram + 0.05 * randn(M, N);
end