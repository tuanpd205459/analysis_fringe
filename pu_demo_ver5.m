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

%% ==================== MAIN EXECUTION (THÊM VÀO CUỐI) ====================

% Thêm vào cuối hàm main để có phân tích hoàn chỉnh
fprintf('\n=== PHÂN TÍCH TOÀN DIỆN VÀ LƯU KẾT QUẢ ===\n');
[quality_metrics, proc_time] = comprehensive_analysis(true_phase, phase_final);

fprintf('\n=== HOÀN THÀNH DEMO ===\n');
fprintf('Tổng thời gian thực hiện: %.2f giây\n', proc_time);
fprintf('Demo phase unwrapping holography đã hoàn tất!\n');



%% ==================== HÀM HỖ TRỢ THÊM ================== 

function phase_corrected = adaptive_median_filter(phase, reliability_map)
    % Adaptive median filter dựa trên reliability map
    
    phase_corrected = phase;
    [M, N] = size(phase);
    
    % Xác định kích thước kernel dựa trên reliability
    for i = 2:M-1
        for j = 2:N-1
            reliability = reliability_map(i,j);
            
            % Kernel size nghịch đảo với reliability
            if reliability < 0.3
                kernel_size = 7;
            elseif reliability < 0.6
                kernel_size = 5;
            else
                kernel_size = 3;
            end
            
            % Extract local region
            r_start = max(1, i - floor(kernel_size/2));
            r_end = min(M, i + floor(kernel_size/2));
            c_start = max(1, j - floor(kernel_size/2));
            c_end = min(N, j + floor(kernel_size/2));
            
            local_region = phase(r_start:r_end, c_start:c_end);
            
            % Apply median filter only for low reliability regions
            if reliability < 0.5
                phase_corrected(i,j) = median(local_region(:));
            end
        end
    end
end

function [quality_metrics, processing_time] = comprehensive_analysis(true_phase, phase_final)
    % Phân tích toàn diện chất lượng và hiệu suất
    
    tic;
    
    % Remove global offset
    true_adj = true_phase - mean(true_phase(:));
    final_adj = phase_final - mean(phase_final(:));
    
    % Basic metrics
    quality_metrics.rmse = sqrt(mean((final_adj - true_adj).^2));
    quality_metrics.mae = mean(abs(final_adj - true_adj));
    quality_metrics.correlation = corr(true_adj(:), final_adj(:));
    
    % Advanced metrics
    quality_metrics.ssim = ssim(mat2gray(final_adj), mat2gray(true_adj));
    
    % Gradient analysis
    [gx_true, gy_true] = gradient(true_adj);
    [gx_final, gy_final] = gradient(final_adj);
    
    grad_error = sqrt((gx_final - gx_true).^2 + (gy_final - gy_true).^2);
    quality_metrics.gradient_rmse = sqrt(mean(grad_error(:).^2));
    
    % Frequency domain analysis
    fft_true = fft2(true_adj);
    fft_final = fft2(final_adj);
    
    quality_metrics.spectral_error = mean(abs(fft_final(:) - fft_true(:)).^2) / ...
                                   mean(abs(fft_true(:)).^2);
    
    % Error distribution analysis
    error_map = abs(final_adj - true_adj);
    quality_metrics.error_std = std(error_map(:));
    quality_metrics.error_percentiles = prctile(error_map(:), [50, 75, 90, 95, 99]);
    
    processing_time = toc;
    
    % Display comprehensive results
    fprintf('\n=== PHÂN TÍCH TOÀN DIỆN ===\n');
    fprintf('Thời gian xử lý: %.3f giây\n', processing_time);
    fprintf('RMSE: %.4f rad\n', quality_metrics.rmse);
    fprintf('MAE: %.4f rad\n', quality_metrics.mae);
    fprintf('Correlation: %.4f\n', quality_metrics.correlation);
    fprintf('SSIM: %.4f\n', quality_metrics.ssim);
    fprintf('Gradient RMSE: %.4f\n', quality_metrics.gradient_rmse);
    fprintf('Spectral Error: %.4f\n', quality_metrics.spectral_error);
    fprintf('Error Percentiles (50,75,90,95,99): %.3f, %.3f, %.3f, %.3f, %.3f\n', ...
            quality_metrics.error_percentiles);
end
%%
   

%%
% 
% function true_phase = create_complex_phase_surface(X, Y, N)
%     % Tạo bề mặt pha phức tạp mô phỏng đo biến dạng thực tế
%     
%     % Thành phần chính - dạng sóng
%     component1 = 3 * sin(2*pi*X/50) .* cos(2*pi*Y/40);
%     
%     % Thành phần biến dạng cục bộ
%     component2 = 2 * exp(-((X-N/2).^2 + (Y-N/2).^2)/(N/8)^2);
%     
%     % Thành phần nhiễu có cấu trúc
%     component3 = 0.5 * sin(2*pi*X/20) + 0.3 * cos(2*pi*Y/30);
%     
%     % Gradient tuyến tính
%     component4 = 0.02 * (X + Y);
%     
%     true_phase = component1 + component2 + component3 + component4;
%     
%     fprintf('Tạo pha thực với range: %.2f đến %.2f rad\n', ...
%             min(true_phase(:)), max(true_phase(:)));
% end
function true_phase = create_complex_phase_surface(X, Y, N)
    % Tạo bề mặt pha phức tạp - GIẢM GRADIENT ĐỂ TRÁNH OVER-WRAPPING
    
    % Thành phần chính - dạng sóng (giảm frequency)
    component1 = 4 * sin(2*pi*X/80) .* cos(2*pi*Y/70);  % Tăng wavelength
    
    % Thành phần biến dạng cục bộ (giảm amplitude)
    component2 = 1.5 * exp(-((X-N/2).^2 + (Y-N/2).^2)/(N/6)^2);
    
    % Thành phần nhiễu có cấu trúc (giảm frequency)
    component3 = 0.3 * sin(2*pi*X/40) + 0.2 * cos(2*pi*Y/50);
    
    % Gradient tuyến tính (giảm độ dốc)
    component4 = 0.01 * (X + Y);
    
    % Thành phần cao tần cục bộ - NGUYÊN NHÂN CHÍNH CỦA OVER-WRAPPING
    % Tạo vùng có gradient cao để test thuật toán
    high_freq_region = (X > N/2-20 & X < N/2+20 & Y > N/4 & Y < 3*N/4);
    component5 = high_freq_region .* 1.5 .* sin(2*pi*X/8);  % Cao tần cục bộ
    
    true_phase = component1  + component3 ;
    
    % Tính thống kê gradient để đánh giá độ khó
    [gx, gy] = gradient(true_phase);
    grad_magnitude = sqrt(gx.^2 + gy.^2);
    max_gradient = max(grad_magnitude(:));
    wrapping_density = sum(grad_magnitude(:) > pi/2) / numel(grad_magnitude);
    
    fprintf('Tạo pha thực với:\n');
    fprintf('  Range: %.2f đến %.2f rad\n', min(true_phase(:)), max(true_phase(:)));
    fprintf('  Max gradient: %.2f rad/pixel\n', max_gradient);
    fprintf('  Wrapping density: %.1f%% (>π/2 grad)\n', wrapping_density*100);
    
    if max_gradient > pi
        fprintf('  ⚠️  CẢNH BÁO: Gradient quá cao -> dễ over-wrapping!\n');
    end
end
function phase_wrapped = wrap_phase(phase)
    % Wrap pha về khoảng [-π, π]
    phase_wrapped = angle(exp(1i * phase));
end

function phase_estimate = simulate_fringe_phase_estimate(true_phase, X, Y)
    % Mô phỏng pha ước lượng từ phân tích vân giao thoa
    
    % Thêm nhiễu Gaussian
    noise_level = 0;
    gaussian_noise = noise_level * randn(size(true_phase));
    
    % Thêm sai lệch hệ thống (systematic bias)
    bias = 0.1 * sin(2*pi*X/100) + 0.05 * cos(2*pi*Y/120);
    
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
    % Ước lượng số lần nhảy 2π tại mỗi pixel - cải thiện
    
    % Làm mượt phase_estimate trước khi tính jump count
    h = fspecial('gaussian', [5 5], 1.2);
    phase_estimate_smooth = imfilter(phase_estimate, h, 'replicate');
    
    phase_diff = phase_estimate_smooth - phase_wrapped;
    
    % Sử dụng phương pháp robust hơn để ước lượng jump count
    jump_count_raw = phase_diff / (2*pi);
    
    % Áp dụng median filter trước khi round
    jump_count_filtered = medfilt2(jump_count_raw, [5 5]);
    jump_count = round(jump_count_filtered);
    
    % Kiểm tra và sửa các outlier
    jump_count = remove_jump_outliers(jump_count);
    
    fprintf('Ước lượng jump count: min=%d, max=%d\n', ...
            min(jump_count(:)), max(jump_count(:)));
end

function jump_count_clean = remove_jump_outliers(jump_count)
    % Loại bỏ các outlier trong jump count
    
    % Tính local median
    local_median = medfilt2(jump_count, [7 7]);
    
    % Tìm các điểm khác biệt quá lớn so với local median
    diff_from_median = abs(jump_count - local_median);
    threshold = 2; % Ngưỡng cho phép khác biệt 2 jump
    
    outlier_mask = diff_from_median > threshold;
    
    % Thay thế outlier bằng local median
    jump_count_clean = jump_count;
    jump_count_clean(outlier_mask) = local_median(outlier_mask);
    
    fprintf('Đã sửa %d outlier trong jump count\n', sum(outlier_mask(:)));
end

function phase_unwrapped = hybrid_phase_unwrapping(phase_wrapped, phase_estimate, ...
                                                  reliability_map, jump_count)
    % Thuật toán phase unwrapping kết hợp - cải thiện
    
    % Bước 1: Unwrap cơ bản dựa trên jump count
    phase_unwrapped_basic = phase_wrapped + 2*pi * jump_count;
    
    % Bước 2: Sử dụng path-following unwrapping cho vùng tin cậy cao
    phase_unwrapped = path_following_unwrapping(phase_wrapped, reliability_map);
    
    % Bước 3: Kết hợp với phase_estimate cho vùng tin cậy thấp
    threshold = 0.6;
    low_reliability = reliability_map < threshold;
    
    if sum(low_reliability(:)) > 0
        % Tính offset trung bình giữa hai phương pháp trong vùng tin cậy cao
        high_reliability = ~low_reliability;
        if sum(high_reliability(:)) > 100
            offset_samples = phase_estimate(high_reliability) - phase_unwrapped(high_reliability);
            % Loại bỏ outlier trong offset
            offset_samples = offset_samples(abs(offset_samples) < 3*std(offset_samples));
            mean_offset = median(offset_samples); % Sử dụng median thay vì mean
            
            % Điều chỉnh phase_estimate
            phase_estimate_adjusted = phase_estimate - mean_offset;
            
            % Kết hợp cho vùng tin cậy thấp
            phase_unwrapped(low_reliability) = phase_estimate_adjusted(low_reliability);
        end
    end
    
    % Bước 4: Đảm bảo tính liên tục tại biên giữa các vùng
    phase_unwrapped = ensure_boundary_continuity(phase_unwrapped, reliability_map, threshold);
    
    fprintf('Phase unwrapping kết hợp hoàn tất\n');
end

function phase_unwrapped = path_following_unwrapping(phase_wrapped, reliability_map)
    % Path-following unwrapping bắt đầu từ điểm tin cậy nhất
    
    [M, N] = size(phase_wrapped);
    phase_unwrapped = zeros(M, N);
    visited = false(M, N);
    
    % Tìm điểm bắt đầu (tin cậy nhất và ở trung tâm)
    center_region = false(M, N);
    center_region(M/4:3*M/4, N/4:3*N/4) = true;
    [~, max_idx] = max(reliability_map(center_region));
    [center_rows, center_cols] = find(center_region);
    start_row = center_rows(max_idx);
    start_col = center_cols(max_idx);
    
    % Khởi tạo
    phase_unwrapped(start_row, start_col) = phase_wrapped(start_row, start_col);
    visited(start_row, start_col) = true;
    
    % Queue cho breadth-first search, ưu tiên theo độ tin cậy
    queue = [start_row, start_col, reliability_map(start_row, start_col)];
    
    % Các hướng di chuyển (4-connected)
    directions = [-1 0; 1 0; 0 -1; 0 1];
    
    while ~isempty(queue)
        % Sắp xếp queue theo độ tin cậy giảm dần
        [~, sort_idx] = sort(queue(:,3), 'descend');
        queue = queue(sort_idx, :);
        
        % Lấy điểm có độ tin cậy cao nhất
        current = queue(1, :);
        queue(1, :) = [];
        
        curr_row = current(1);
        curr_col = current(2);
        
        % Xử lý các điểm lân cận
        for d = 1:4
            new_row = curr_row + directions(d, 1);
            new_col = curr_col + directions(d, 2);
            
            % Kiểm tra biên
            if new_row >= 1 && new_row <= M && new_col >= 1 && new_col <= N && ...
               ~visited(new_row, new_col)
                
                % Tính pha unwrapped cho điểm mới
                phase_diff = phase_wrapped(new_row, new_col) - ...
                           phase_wrapped(curr_row, curr_col);
                phase_diff = angle(exp(1i * phase_diff)); % Wrap difference
                
                phase_unwrapped(new_row, new_col) = ...
                    phase_unwrapped(curr_row, curr_col) + phase_diff;
                
                visited(new_row, new_col) = true;
                
                % Thêm vào queue nếu đủ tin cậy
                if reliability_map(new_row, new_col) > 0.3
                    queue(end+1, :) = [new_row, new_col, reliability_map(new_row, new_col)];
                end
            end
        end
    end
    
    % Xử lý các điểm chưa được visit (tin cậy thấp)
    unvisited = ~visited;
    if sum(unvisited(:)) > 0
        % Sử dụng interpolation cho các điểm chưa được xử lý
        [Y_vis, X_vis] = find(visited);
        [Y_unvis, X_unvis] = find(unvisited);
        
        if length(Y_vis) > 3
            F = scatteredInterpolant(X_vis, Y_vis, phase_unwrapped(visited), ...
                                   'linear', 'nearest');
            phase_unwrapped(unvisited) = F(X_unvis, Y_unvis);
        end
    end
end

function phase_continuous = ensure_boundary_continuity(phase_unwrapped, reliability_map, threshold)
    % Đảm bảo tính liên tục tại biên giữa vùng tin cậy cao và thấp
    
    % Tìm biên giữa hai vùng
    high_reliability = reliability_map >= threshold;
    
    % Morphological operation để tìm biên
    se = strel('disk', 1);
    boundary = (imdilate(high_reliability, se) & ~high_reliability) | ...
               (imdilate(~high_reliability, se) & high_reliability);
    
    phase_continuous = phase_unwrapped;
    
    % Sửa discontinuity tại biên
    [boundary_rows, boundary_cols] = find(boundary);
    
    for i = 1:length(boundary_rows)
        row = boundary_rows(i);
        col = boundary_cols(i);
        
        % Lấy giá trị từ vùng lân cận 3x3
        row_range = max(1, row-1):min(size(phase_unwrapped,1), row+1);
        col_range = max(1, col-1):min(size(phase_unwrapped,2), col+1);
        
        local_values = phase_unwrapped(row_range, col_range);
        local_reliability = reliability_map(row_range, col_range);
        
        % Tính weighted average dựa trên reliability
        if sum(local_reliability(:)) > 0
            weighted_sum = sum(local_values(:) .* local_reliability(:));
            weight_sum = sum(local_reliability(:));
            phase_continuous(row, col) = weighted_sum / weight_sum;
        end
    end
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
    title('Pha Wrapped (từ FFT)');
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
    fprintf('  Phase Wrapped:  %.4f rad\n', rmse_wrapped);
    fprintf('  Phase Estimate: %.4f rad\n', rmse_estimate);
    fprintf('  Phase Final:    %.4f rad\n', rmse_final);
    fprintf('\nCorrelation with True Phase:\n');
    fprintf('  Phase Wrapped:  %.4f\n', corr_wrapped);
    fprintf('  Phase Estimate: %.4f\n', corr_estimate);
    fprintf('  Phase Final:    %.4f\n', corr_final);
    
    % Tính improvement
    improvement_vs_wrapped = (rmse_wrapped - rmse_final) / rmse_wrapped * 100;
    improvement_vs_estimate = (rmse_estimate - rmse_final) / rmse_estimate * 100;
    
    fprintf('\nImprovement:\n');
    fprintf('  vs Wrapped:  %.1f%%\n', improvement_vs_wrapped);
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
    title('Pha Wrapped');
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
    legend('True', 'Wrapped', 'Estimate', 'Final', 'Location', 'best');
    title('So sánh Profile tại hàng giữa');
    xlabel('Pixel');
    ylabel('Phase (rad)');
    grid on;
    
    subplot(2,1,2);
    plot(abs(profile_true - profile_wrapped), 'r--', 'LineWidth', 1.5); hold on;
    plot(abs(profile_true - profile_estimate), 'b:', 'LineWidth', 1.5);
    plot(abs(profile_true - profile_final), 'g-', 'LineWidth', 2);
    legend('Error Wrapped', 'Error Estimate', 'Error Final', 'Location', 'best');
    title('Sai số tuyệt đối');
    xlabel('Pixel');
    ylabel('Absolute Error (rad)');
    grid on;
end

% ----- HÀM MỚI ĐƯỢC THÊM VÀO -----
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
    title('Pha Wrapped (3D)');
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
