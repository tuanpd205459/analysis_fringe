
function demo_phase_unwrapping()
% Demo sử dụng hàm unwrapping với ramp correction
    
    fprintf('=== DEMO PHASE UNWRAPPING VỚI RAMP CORRECTION ===\n\n');
    
    % Tạo dữ liệu test
    [rows, cols] = deal(100, 100);
    [X, Y] = meshgrid(1:cols, 1:rows);
    
    % Pha thực với mặt phẳng nghiêng
    phi_true = 0.05*X + 0.03*Y + sin(0.2*X) + cos(0.15*Y);
    
    % Pha wrapped (mô phỏng dữ liệu đo)
    phi_wrapped = angle(exp(1i * phi_true));
    
    % Pha estimate thiếu mặt phẳng nghiêng
    phi_est = sin(0.2*X) + cos(0.15*Y);  % Thiếu phần 0.05*X + 0.03*Y
    
    % Phân tích gradient
    analyze_phase_gradient(phi_wrapped, phi_est);
    
    % Unwrapping với hiệu chỉnh
    phi_unwrapped = phase_unwrap_with_ramp_correction(phi_wrapped, phi_est);
    
    % Đánh giá kết quả
    error_rms = sqrt(mean((phi_unwrapped(:) - phi_true(:)).^2));
    fprintf('\nRMS Error: %.6f rad\n', error_rms);
    
    % Hiển thị kết quả
    visualize_unwrapping_results(phi_wrapped, phi_est, phi_unwrapped);
    
    fprintf('\nDemo hoàn thành!\n');
end

function phi_unwrapped = phase_unwrap_with_ramp_correction(phi_wrapped, phi_est, mask)
% PHASE_UNWRAP_WITH_RAMP_CORRECTION - Unwrap pha với hiệu chỉnh mặt phẳng nghiêng
%
% Inputs:
%   phi_wrapped - Pha wrapped (rad)
%   phi_est     - Pha ước lượng ban đầu (rad) 
%   mask        - Mask (optional, 1=valid, 0=invalid)
%
% Output:
%   phi_unwrapped - Pha unwrapped đã hiệu chỉnh (rad)

    if nargin < 3 || isempty(mask)
        mask = ones(size(phi_wrapped));
    end
    
    % 1. Ước lượng mặt phẳng nghiêng từ pha wrapped
    ramp_wrapped = estimate_ramp_plane(phi_wrapped, mask);
    
    % 2. Ước lượng mặt phẳng nghiêng từ pha estimate 
    ramp_est = estimate_ramp_plane(phi_est, mask);
    
    % 3. Hiệu chỉnh phi_est để có cùng xu hướng nghiêng
    phi_est_corrected = phi_est + (ramp_wrapped - ramp_est);
    
    % 4. Tính số bậc k với phi_est đã hiệu chỉnh
    k = round((phi_wrapped - phi_est_corrected) / (2 * pi));
    
    % 5. Unwrap cuối cùng
    phi_unwrapped = phi_wrapped + 2 * pi * k;
    
    fprintf('Hoàn thành phase unwrapping với ramp correction\n');
end

function ramp = estimate_ramp_plane(phi, mask)
% Ước lượng mặt phẳng nghiêng: phi = a*x + b*y + c
    
    [rows, cols] = size(phi);
    [X, Y] = meshgrid(1:cols, 1:rows);
    
    % Chỉ sử dụng các điểm hợp lệ
    valid_idx = mask == 1;
    
    if sum(valid_idx(:)) < 3
        error('Không đủ điểm hợp lệ để fit mặt phẳng');
    end
    
    % Tạo ma trận A cho least squares: [x y 1] * [a; b; c] = phi
    x_valid = X(valid_idx);
    y_valid = Y(valid_idx);
    phi_valid = phi(valid_idx);
    
    A = [x_valid(:), y_valid(:), ones(length(x_valid(:)), 1)];
    
    % Giải least squares
    coeffs = A \ phi_valid(:);
    
    % Tạo mặt phẳng nghiêng cho toàn bộ ảnh
    ramp = coeffs(1) * X + coeffs(2) * Y + coeffs(3);
    
    fprintf('Hệ số mặt phẳng - a: %.6f, b: %.6f, c: %.6f\n', ...
            coeffs(1), coeffs(2), coeffs(3));
end

% ================ HÀM KIỂM TRA VÀ VISUALIZATION ================

function analyze_phase_gradient(phi_wrapped, phi_est)
% Phân tích gradient để phát hiện mặt phẳng nghiêng
    
    [grad_y_wrapped, grad_x_wrapped] = gradient(phi_wrapped);
    [grad_y_est, grad_x_est] = gradient(phi_est);
    
    % Gradient trung bình
    mean_grad_x_wrapped = mean(grad_x_wrapped(:), 'omitnan');
    mean_grad_y_wrapped = mean(grad_y_wrapped(:), 'omitnan');
    mean_grad_x_est = mean(grad_x_est(:), 'omitnan');
    mean_grad_y_est = mean(grad_y_est(:), 'omitnan');
    
    fprintf('\n=== PHÂN TÍCH GRADIENT ===\n');
    fprintf('Pha wrapped - Grad X: %.6f, Grad Y: %.6f\n', ...
            mean_grad_x_wrapped, mean_grad_y_wrapped);
    fprintf('Pha estimate - Grad X: %.6f, Grad Y: %.6f\n', ...
            mean_grad_x_est, mean_grad_y_est);
    
    % Chênh lệch gradient
    diff_grad_x = mean_grad_x_wrapped - mean_grad_x_est;
    diff_grad_y = mean_grad_y_wrapped - mean_grad_y_est;
    
    fprintf('Chênh lệch gradient - X: %.6f, Y: %.6f\n', ...
            diff_grad_x, diff_grad_y);
    
    % Cảnh báo nếu có mặt phẳng nghiêng lớn
    threshold = 0.1;
    if abs(diff_grad_x) > threshold || abs(diff_grad_y) > threshold
        fprintf('⚠️  PHÁT HIỆN MẶT PHẲNG NGHIÊNG LỚN! Cần hiệu chỉnh.\n');
    else
        fprintf('✅ Mặt phẳng nghiêng nhỏ, có thể bỏ qua.\n');
    end
end

function visualize_unwrapping_results(phi_wrapped, phi_est, phi_unwrapped)
% Hiển thị kết quả unwrapping
    
    figure('Position', [100, 100, 1200, 800]);
    
    % Pha wrapped
    subplot(2,3,1);
    imagesc(phi_wrapped);
    colorbar; title('Pha Wrapped');
    axis equal tight;
    
    % Pha estimate
    subplot(2,3,2);
    imagesc(phi_est);
    colorbar; title('Pha Estimate');
    axis equal tight;
    
    % Pha unwrapped
    subplot(2,3,3);
    imagesc(phi_unwrapped);
    colorbar; title('Pha Unwrapped');
    axis equal tight;
    
    % Chênh lệch trước unwrap
    subplot(2,3,4);
    diff_before = phi_wrapped - phi_est;
    imagesc(diff_before);
    colorbar; title('Chênh lệch trước unwrap');
    axis equal tight;
    
    % Số bậc k
    subplot(2,3,5);
    k = round((phi_wrapped - phi_est) / (2*pi));
    imagesc(k);
    colorbar; title('Số bậc K');
    axis equal tight;
    
    % Histogram chênh lệch
    subplot(2,3,6);
    histogram(diff_before(:), 50);
    title('Histogram chênh lệch');
    xlabel('Chênh lệch (rad)');
    ylabel('Tần suất');
    grid on;
end

% ================ VÍ DỤ SỬ DỤNG ================
