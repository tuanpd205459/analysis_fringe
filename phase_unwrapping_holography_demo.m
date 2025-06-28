% PHASE UNWRAPPING HOLOGRAPHY DEMO
% Thuật toán kết hợp phase unwrapping cho giao thoa holography
% Tác giả: Demo Code
% Ngày: 2025

clc; clear; close all;

%% Phần 1: Mô phỏng dữ liệu
fprintf('=== MÔ PHỎNG DỮ LIỆU HOLOGRAPHIC INTERFEROMETRY ===\n');

% Tham số mô phỏng
N = 256;                    % Kích thước ảnh
[X, Y] = meshgrid(1:N, 1:N);

% Tạo pha thực (true phase) - dạng bề mặt phức tạp
true_phase = create_complex_phase_surface(X, Y, N);

% Mô phỏng pha wrapped từ FFT hologram
phase_wrapped = wrap_phase(true_phase);

% THÊM BƯỚC LỌC PHA WRAPPED MỚI Ở ĐÂY
fprintf('Thực hiện lọc pha wrapped...\n');
phase_wrapped_filtered = filter_wrapped_phase(phase_wrapped);
% Gán pha đã lọc trở lại biến phase_wrapped để các hàm sau sử dụng
phase_wrapped = phase_wrapped_filtered; 
fprintf('Lọc pha wrapped hoàn tất.\n');

% Mô phỏng pha ước lượng từ vân giao thoa (có sai lệch)
phase_estimate = simulate_fringe_phase_estimate(true_phase, X, Y);

% Hiển thị dữ liệu mô phỏng
display_simulation_data(true_phase, phase_wrapped, phase_estimate);

%% Phần 2: Thuật toán Phase Unwrapping kết hợp
fprintf('\n=== THỰC HIỆN PHASE UNWRAPPING KẾT HỢP ===\n');

% Bước 1: Phân tích chất lượng
reliability_map = calculate_reliability_map(phase_wrapped, phase_estimate);

% Bước 2: Ước lượng số lần nhảy 2π
jump_count = estimate_jump_count(phase_wrapped, phase_estimate);

% Bước 3: Phase unwrapping kết hợp
phase_unwrapped = hybrid_phase_unwrapping(phase_wrapped, phase_estimate, ...
                                         reliability_map, jump_count);

% Bước 4: Tinh chỉnh toàn cục
phase_final = global_refinement(phase_unwrapped, reliability_map);

%% Phần 3: Đánh giá kết quả
fprintf('\n=== ĐÁNH GIÁ KẾT QUẢ ===\n');
evaluate_results(true_phase, phase_wrapped, phase_estimate, phase_final);
% Hiển thị kết quả cuối cùng
display_final_results(true_phase, phase_wrapped, phase_estimate, ...
    phase_final, reliability_map);
% Hiển thị kết quả dưới dạng 3D
fprintf('\n=== HIỂN THỊ KẾT QUẢ 3D ===\n');
display_3d_views(true_phase, phase_wrapped, phase_final);

%% FUNCTIONS
function true_phase = create_complex_phase_surface(X, Y, N)
    % Tạo bề mặt pha phức tạp mô phỏng đo biến dạng thực tế
    
    % Thành phần chính - dạng sóng
    component1 = 2 * sin(2*pi*X/50) .* cos(2*pi*Y/40);
    
    % Thành phần biến dạng cục bộ
    component2 = 2 * exp(-((X-N/2).^2 + (Y-N/2).^2)/(N/8)^2);
    
    % Thành phần nhiễu có cấu trúc
    component3 = 0.2 * sin(2*pi*X/20) + 0.3 * cos(2*pi*Y/30);
    
    % Gradient tuyến tính
    component4 = 0.1 * (X + Y);
    
    true_phase = component1 + component2 + component3 + component4;
    
    fprintf('Tạo pha thực với range: %.2f đến %.2f rad\n', ...
            min(true_phase(:)), max(true_phase(:)));
end

function phase_wrapped = wrap_phase(phase)
    % Wrap pha về khoảng [-π, π]
    phase_wrapped = angle(exp(1i * phase));
end

function phase_wrapped_filtered = filter_wrapped_phase(phase_wrapped_input)
    % Hàm mới: Lọc pha wrapped để giảm nhiễu
    % Sử dụng bộ lọc trung vị (median filter) kích thước 3x3
    % Có thể thay đổi loại bộ lọc (ví dụ: Gaussian) và kích thước tùy theo yêu cầu
    
    filter_size = [3 3]; % Kích thước bộ lọc
    phase_wrapped_filtered = medfilt2(phase_wrapped_input, filter_size);
end

function phase_estimate = simulate_fringe_phase_estimate(true_phase, X, Y)
    % Mô phỏng pha ước lượng từ phân tích vân giao thoa
    
    % Thêm nhiễu Gaussian
    noise_level = 0.3;
    gaussian_noise = noise_level * randn(size(true_phase));
    
    % Thêm sai lệch hệ thống (systematic bias)
    bias = 0.2 * sin(2*pi*X/100) + 0.05 * cos(2*pi*Y/120);
    
    % Làm mờ để mô phỏng độ phân giải thấp hơn
    h = fspecial('gaussian', [5 5], 1.5);
    phase_smoothed = imfilter(true_phase, h, 'replicate');
    
    % Kết hợp các yếu tố
    phase_estimate = phase_smoothed + gaussian_noise + bias;
    
    % Thêm một số điểm outlier
    outlier_mask = rand(size(true_phase)) < 0.02;
    phase_estimate(outlier_mask) = phase_estimate(outlier_mask) + ...
                                   5 * randn(sum(outlier_mask(:)), 1);
    
    fprintf('Tạo pha ước lượng với RMSE: %.3f rad\n', ...
            sqrt(mean((phase_estimate(:) - true_phase(:)).^2)));
end

function reliability_map = calculate_reliability_map(phase_wrapped, phase_estimate)
    % Tính bản đồ độ tin cậy
    
    % Độ tin cậy dựa trên gradient
    [gx_w, gy_w] = gradient(phase_wrapped);
    [gx_e, gy_e] = gradient(phase_estimate);
    
    grad_consistency = exp(-abs(gx_w - gx_e) - abs(gy_w - gy_e));
    
    % Độ tin cậy dựa trên local variance
    h = ones(5,5)/25;
    local_var_wrapped = imfilter(phase_wrapped.^2, h) - ...
                       (imfilter(phase_wrapped, h)).^2;
    local_var_estimate = imfilter(phase_estimate.^2, h) - ...
                        (imfilter(phase_estimate, h)).^2;
    
    var_reliability = exp(-local_var_wrapped - local_var_estimate);
    
    % Kết hợp các yếu tố
    reliability_map = 0.6 * grad_consistency + 0.4 * var_reliability;
    
    % Làm mượt bản đồ độ tin cậy
    h_smooth = fspecial('gaussian', [3 3], 0.8);
    reliability_map = imfilter(reliability_map, h_smooth, 'replicate');
    
    fprintf('Tính toán bản đồ độ tin cậy hoàn tất\n');
end

function jump_count = estimate_jump_count(phase_wrapped, phase_estimate)
    % Ước lượng số lần nhảy 2π tại mỗi pixel
    
    phase_diff = phase_estimate - phase_wrapped;
    jump_count = round(phase_diff / (2*pi));
    
    % Làm mượt jump count để tránh lỗi
    jump_count = medfilt2(jump_count, [3 3]);
    
    fprintf('Ước lượng jump count: min=%d, max=%d\n', ...
            min(jump_count(:)), max(jump_count(:)));
end

function phase_unwrapped = hybrid_phase_unwrapping(phase_wrapped, phase_estimate, ...
                                                  reliability_map, jump_count)
    % Thuật toán phase unwrapping kết hợp
    
    % Bước 1: Unwrap cơ bản dựa trên jump count
    phase_unwrapped = phase_wrapped + 2*pi * jump_count;
    
    % Bước 2: Kết hợp có trọng số dựa trên độ tin cậy
    threshold = 0.5;
    
    % Vùng độ tin cậy thấp: sử dụng nhiều thông tin từ phase_estimate
    low_reliability = reliability_map < threshold;
    
    weight_estimate = (1 - reliability_map) .* low_reliability;
    weight_wrapped = reliability_map + (1 - low_reliability);
    
    % Normalize weights
    total_weight = weight_estimate + weight_wrapped;
    weight_estimate = weight_estimate ./ total_weight;
    weight_wrapped = weight_wrapped ./ total_weight;
    
    % Kết hợp có trọng số
    phase_combined = weight_wrapped .* phase_unwrapped + ...
                    weight_estimate .* phase_estimate;
    
    % Bước 3: Xử lý vùng biên và interpolation
    phase_unwrapped = process_boundary_regions(phase_combined, reliability_map);
    
    fprintf('Phase unwrapping kết hợp hoàn tất\n');
end

function phase_refined = process_boundary_regions(phase_unwrapped, reliability_map)
    % Xử lý các vùng biên và interpolation
    
    % Xác định vùng cần interpolation
    reliable_mask = reliability_map > 0.3;
    
    % Sử dụng inpainting cho vùng không tin cậy
    phase_refined = phase_unwrapped;
    
    % Tìm vùng cần sửa chữa
    unreliable_regions = ~reliable_mask;
    
    if sum(unreliable_regions(:)) > 0
        % Sử dụng interpolation cho vùng không tin cậy
        [Y, X] = find(reliable_mask);
        [Yi, Xi] = find(unreliable_regions);
        
        if ~isempty(Y) && ~isempty(Yi)
            reliable_values = phase_unwrapped(reliable_mask);
            F = scatteredInterpolant(X, Y, reliable_values, 'linear', 'nearest');
            interpolated_values = F(Xi, Yi);
            phase_refined(unreliable_regions) = interpolated_values;
        end
    end
end

function phase_final = global_refinement(phase_unwrapped, reliability_map)
    % Tinh chỉnh toàn cục để đảm bảo tính liên tục
    
    % Làm mượt toàn cục với trọng số dựa trên độ tin cậy
    h = fspecial('gaussian', [3 3], 0.8);
    
    phase_smoothed = imfilter(phase_unwrapped, h, 'replicate');
    
    % Kết hợp pha gốc và pha đã làm mượt
    alpha = 0.1; % Hệ số làm mượt
    weight_smooth = alpha * (1 - reliability_map);
    weight_original = 1 - weight_smooth;
    
    phase_final = weight_original .* phase_unwrapped + ...
                  weight_smooth .* phase_smoothed;
    
    % Kiểm tra và sửa các discontinuity còn lại
    phase_final = fix_remaining_discontinuities(phase_final);
    
    fprintf('Tinh chỉnh toàn cục hoàn tất\n');
end

function phase_corrected = fix_remaining_discontinuities(phase)
    % Sửa các discontinuity còn lại
    
    [gx, gy] = gradient(phase);
    grad_magnitude = sqrt(gx.^2 + gy.^2);
    
    % Tìm các điểm có gradient quá lớn (có thể là lỗi unwrapping)
    threshold = 3 * std(grad_magnitude(:));
    discontinuity_mask = grad_magnitude > threshold;
    
    phase_corrected = phase;
    
    if sum(discontinuity_mask(:)) > 0
        % Sử dụng median filter cho các vùng có discontinuity
        phase_corrected(discontinuity_mask) = ...
            medfilt2(phase(discontinuity_mask), [3 3]);
    end
end

function display_simulation_data(true_phase, phase_wrapped, phase_estimate)
    % Hiển thị dữ liệu mô phỏng
    
    figure('Name', 'Dữ liệu mô phỏng', 'Position', [100 100 1200 400]);
    
    subplot(1,3,1);
    imagesc(true_phase);
    colorbar;
    title('Pha thực (True Phase)');
    axis equal tight;
    
    subplot(1,3,2);
    imagesc(phase_wrapped);
    colorbar;
    title('Pha Wrapped (từ FFT - đã lọc)'); % Cập nhật tiêu đề
    axis equal tight;
    
    subplot(1,3,3);
    imagesc(phase_estimate);
    colorbar;
    title('Pha ước lượng (từ vân giao thoa)');
    axis equal tight;
    
    colormap jet;
end

function evaluate_results(true_phase, phase_wrapped, phase_estimate, phase_final)
    % Đánh giá kết quả
    
    % Tính các metrics
    rmse_wrapped = sqrt(mean((wrap_phase(true_phase(:) - phase_wrapped(:))).^2));
    rmse_estimate = sqrt(mean((true_phase(:) - phase_estimate(:)).^2));
    rmse_final = sqrt(mean((true_phase(:) - phase_final(:)).^2));
    
    % Tính correlation
    corr_wrapped = corr(true_phase(:), phase_wrapped(:));
    corr_estimate = corr(true_phase(:), phase_estimate(:));
    corr_final = corr(true_phase(:), phase_final(:));
    
    % In kết quả
    fprintf('RMSE Comparison:\n');
    fprintf('  Phase Wrapped (đã lọc):  %.4f rad\n', rmse_wrapped); % Cập nhật nhãn
    fprintf('  Phase Estimate: %.4f rad\n', rmse_estimate);
    fprintf('  Phase Final:    %.4f rad\n', rmse_final);
    fprintf('\nCorrelation with True Phase:\n');
    fprintf('  Phase Wrapped (đã lọc):  %.4f\n', corr_wrapped); % Cập nhật nhãn
    fprintf('  Phase Estimate: %.4f\n', corr_estimate);
    fprintf('  Phase Final:    %.4f\n', corr_final);
    
    % Tính improvement
    improvement_vs_wrapped = (rmse_wrapped - rmse_final) / rmse_wrapped * 100;
    improvement_vs_estimate = (rmse_estimate - rmse_final) / rmse_estimate * 100;
    
    fprintf('\nImprovement:\n');
    fprintf('  vs Wrapped (đã lọc):  %.1f%%\n', improvement_vs_wrapped); % Cập nhật nhãn
    fprintf('  vs Estimate: %.1f%%\n', improvement_vs_estimate);
end

function display_final_results(true_phase, phase_wrapped, phase_estimate, ...
                              phase_final, reliability_map)
    % Hiển thị kết quả cuối cùng
    
    figure('Name', 'Kết quả Phase Unwrapping', 'Position', [100 500 1500 800]);
    
    % Hàng trên: So sánh phases
    subplot(2,3,1);
    imagesc(true_phase);
    colorbar;
    title('Pha thực');
    axis equal tight;
    
    subplot(2,3,2);
    imagesc(phase_wrapped);
    colorbar;
    title('Pha Wrapped (đã lọc)'); % Cập nhật tiêu đề
    axis equal tight;
    
    subplot(2,3,3);
    imagesc(phase_estimate);
    colorbar;
    title('Pha ước lượng');
    axis equal tight;
    
    % Hàng dưới: Kết quả và phân tích
    subplot(2,3,4);
    imagesc(phase_final);
    colorbar;
    title('Pha sau unwrapping');
    axis equal tight;
    
    subplot(2,3,5);
    imagesc(reliability_map);
    colorbar;
    title('Bản đồ độ tin cậy');
    axis equal tight;
    
    subplot(2,3,6);
    error_map = abs(true_phase - phase_final);
    imagesc(error_map);
    colorbar;
    title('Sai số tuyệt đối');
    axis equal tight;
    
    colormap jet;
    
    % Biểu đồ so sánh profile
    figure('Name', 'So sánh Profile', 'Position', [200 100 800 600]);
    
    % Lấy profile qua tâm ảnh
    center_row = size(true_phase, 1) / 2;
    profile_true = true_phase(center_row, :);
    profile_wrapped = phase_wrapped(center_row, :);
    profile_estimate = phase_estimate(center_row, :);
    profile_final = phase_final(center_row, :);
    
    subplot(2,1,1);
    plot(profile_true, 'k-', 'LineWidth', 2); hold on;
    plot(profile_wrapped, 'r--', 'LineWidth', 1.5);
    plot(profile_estimate, 'b:', 'LineWidth', 1.5);
    plot(profile_final, 'g-', 'LineWidth', 2);
    legend('True', 'Wrapped (đã lọc)', 'Estimate', 'Final', 'Location', 'best'); % Cập nhật nhãn
    title('So sánh Profile tại hàng giữa');
    xlabel('Pixel');
    ylabel('Phase (rad)');
    grid on;
    
    subplot(2,1,2);
    plot(abs(profile_true - profile_wrapped), 'r--', 'LineWidth', 1.5); hold on;
    plot(abs(profile_true - profile_estimate), 'b:', 'LineWidth', 1.5);
    plot(abs(profile_true - profile_final), 'g-', 'LineWidth', 2);
    legend('Sai số Wrapped (đã lọc)', 'Sai số Estimate', 'Sai số Final', 'Location', 'best'); % Cập nhật nhãn
    title('Sai số tuyệt đối');
    xlabel('Pixel');
    ylabel('Absolute Error (rad)');
    grid on;
end

function display_3d_views(true_phase, phase_wrapped, phase_final)
    % Hiển thị các bề mặt pha dưới dạng 3D
    
    figure('Name', 'Hiển thị 3D', 'Position', [300 300 1200 800]);
    colormap jet;
    
    % 1. Pha thực (3D)
    subplot(2, 2, 1);
    surf(true_phase);
    shading interp;
    colorbar;
    title('Pha thực (3D)');
    xlabel('X'); ylabel('Y'); zlabel('Phase (rad)');
    axis tight;
    view(-35, 45); % Đặt góc nhìn
    
    % 2. Pha Wrapped (3D)
    subplot(2, 2, 2);
    surf(phase_wrapped);
    shading interp;
    colorbar;
    title('Pha Wrapped (3D - đã lọc)'); % Cập nhật tiêu đề
    xlabel('X'); ylabel('Y'); zlabel('Phase (rad)');
    axis tight;
    view(-35, 45);
    
    % 3. Pha sau unwrapping (3D)
    subplot(2, 2, 3);
    surf(phase_final);
    shading interp;
    colorbar;
    title('Pha sau unwrapping (3D)');
    xlabel('X'); ylabel('Y'); zlabel('Phase (rad)');
    axis tight;
    view(-35, 45);
    
    % 4. Sai số tuyệt đối (3D)
    subplot(2, 2, 4);
    error_map_3d = abs(true_phase - phase_final);
    surf(error_map_3d);
    shading interp;
    colorbar;
    title('Sai số tuyệt đối (3D)');
    xlabel('X'); ylabel('Y'); zlabel('Error (rad)');
    axis tight;
    view(-35, 45);
    
    fprintf('Hiển thị 3D hoàn tất.\n');
end
