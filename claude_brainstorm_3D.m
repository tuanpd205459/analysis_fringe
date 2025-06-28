%% Phase Unwrapping cho bề mặt 3D
% Mô phỏng unwrapping trên dữ liệu 2D (như ảnh interferometry)
clear; clc; close all;

%% 1. Tạo dữ liệu mẫu 3D
fprintf('=== TẠO DỮ LIỆU SURFACE 3D ===\n');

% Kích thước surface
nx = 50; ny = 40;
[X, Y] = meshgrid(1:nx, 1:ny);

% Tạo true phase surface phức tạp
fprintf('Tạo true phase surface...\n');
true_phase = create_complex_phase_surface(X, Y);

% Tạo wrapped phase
wrapped_phase = angle(exp(1i * true_phase));

% Tạo estimated phase (có nhiễu và một số vùng sai lệch)
estimated_phase = create_estimated_phase(true_phase, X, Y);

% Thêm nhiễu
noise_level = 0.2;
estimated_phase = estimated_phase + noise_level * randn(size(estimated_phase));

fprintf('Kích thước surface: %dx%d\n', size(true_phase));
fprintf('Range true phase: [%.2f, %.2f]\n', min(true_phase(:)), max(true_phase(:)));

%% 2. Thực hiện 3D Phase Unwrapping
fprintf('\n=== THỰC HIỆN 3D PHASE UNWRAPPING ===\n');

% Cấu hình tham số
options.confidence_weight = 0.8;
options.gradient_threshold = pi * 0.3;
options.smooth_kernel_size = 3;
options.max_iterations = 50;

% Unwrap surface
[unwrapped_surface, quality_map, process_info] = unwrap_3d_surface(...
    wrapped_phase, estimated_phase, options);

%% 3. Visualize kết quả
fprintf('\n=== VISUALIZATION ===\n');
visualize_3d_unwrapping(X, Y, true_phase, wrapped_phase, estimated_phase, ...
                       unwrapped_surface, quality_map, process_info);

%% 4. Đánh giá chất lượng
fprintf('\n=== ĐÁNH GIÁ CHẤT LƯỢNG ===\n');
evaluate_3d_results(true_phase, estimated_phase, unwrapped_surface);

%% ========================================================================
%% FUNCTIONS
%% ========================================================================

function true_phase = create_complex_phase_surface(X, Y)
    % Tạo surface pha phức tạp với nhiều đặc trưng
    
    [ny, nx] = size(X);
    
    % Component 1: Ramped surface (xu hướng chính)
    ramp_x = 0.3 * X;
    ramp_y = 0.2 * Y;
    
    % Component 2: Gaussian peaks (các đỉnh)
    peak1 = 8 * exp(-((X-15).^2 + (Y-10).^2)/20);
    peak2 = 6 * exp(-((X-35).^2 + (Y-25).^2)/30);
    peak3 = -4 * exp(-((X-25).^2 + (Y-35).^2)/25);
    
    % Component 3: Sinusoidal waves (sóng)
    wave1 = 3 * sin(0.4 * X) .* cos(0.3 * Y);
    wave2 = 2 * sin(0.2 * (X + Y));
    
    % Component 4: Sharp discontinuity (bất liên tục)
    discontinuity = zeros(size(X));
    discontinuity(Y > 20 & X < 30) = 4;
    
    % Kết hợp tất cả
    true_phase = ramp_x + ramp_y + peak1 + peak2 + peak3 + wave1 + wave2 + discontinuity;
    
    fprintf('   - Created surface with ramps, peaks, waves, and discontinuities\n');
end

function estimated_phase = create_estimated_phase(true_phase, X, Y)
    % Tạo estimated phase có một số vùng sai lệch
    
    estimated_phase = true_phase;
    
    % Thêm sai lệch có hệ thống ở một số vùng
    error_region1 = (X > 20 & X < 35 & Y > 15 & Y < 25);
    estimated_phase(error_region1) = estimated_phase(error_region1) + 3;
    
    error_region2 = (X > 40 & Y < 15);
    estimated_phase(error_region2) = estimated_phase(error_region2) - 2;
    
    % Thêm smooth error
    [ny, nx] = size(X);
    smooth_error = 1.5 * sin(0.1 * X) .* sin(0.15 * Y);
    estimated_phase = estimated_phase + smooth_error;
    
    fprintf('   - Added systematic errors in specific regions\n');
end

function [unwrapped_surface, quality_map, process_info] = unwrap_3d_surface(...
    wrapped_phase, estimated_phase, options)
    
    fprintf('Bắt đầu 3D surface unwrapping...\n');
    
    [ny, nx] = size(wrapped_phase);
    
    % Khởi tạo
    unwrapped_surface = zeros(size(wrapped_phase));
    quality_map = zeros(size(wrapped_phase));
    confidence_map = zeros(size(wrapped_phase));
    
    % Tính gradient trong cả hai hướng
    [grad_x_wrapped, grad_y_wrapped] = gradient(wrapped_phase);
    [grad_x_estimated, grad_y_estimated] = gradient(estimated_phase);
    
    % Phát hiện và sửa chữa phase jumps
    fprintf('Phát hiện phase jumps trong 2D...\n');
    [grad_x_corrected, grad_y_corrected, jump_info] = fix_2d_phase_jumps(...
        grad_x_wrapped, grad_y_wrapped, grad_x_estimated, grad_y_estimated, options);
    
    % Tính confidence map
    fprintf('Tính confidence map...\n');
    confidence_map = calculate_2d_confidence(grad_x_corrected, grad_y_corrected, ...
                                           grad_x_estimated, grad_y_estimated);
    
    % Iterative unwrapping với path-following
    fprintf('Thực hiện iterative unwrapping...\n');
    unwrapped_surface = perform_2d_unwrapping(wrapped_phase, ...
        grad_x_corrected, grad_y_corrected, estimated_phase, confidence_map, options);
    
    % Tính quality map
    quality_map = calculate_quality_map(unwrapped_surface, estimated_phase, confidence_map);
    
    % Thông tin quá trình
    process_info.num_jumps_x = jump_info.num_jumps_x;
    process_info.num_jumps_y = jump_info.num_jumps_y;
    process_info.avg_confidence = mean(confidence_map(:));
    process_info.convergence = 'completed';
    
    fprintf('   - Phát hiện %d jumps theo X, %d jumps theo Y\n', ...
            jump_info.num_jumps_x, jump_info.num_jumps_y);
    fprintf('   - Confidence trung bình: %.3f\n', process_info.avg_confidence);
end

function [grad_x_corrected, grad_y_corrected, jump_info] = fix_2d_phase_jumps(...
    grad_x_wrapped, grad_y_wrapped, grad_x_estimated, grad_y_estimated, options)
    
    threshold = options.gradient_threshold;
    
    % Sửa chữa gradient X
    grad_x_corrected = grad_x_wrapped;
    jump_count_x = 0;
    
    for i = 1:size(grad_x_wrapped, 1)
        for j = 1:size(grad_x_wrapped, 2)
            if abs(grad_x_wrapped(i,j)) > threshold
                jump_count_x = jump_count_x + 1;
                
                if abs(grad_x_estimated(i,j)) < threshold/2
                    % Tin estimated gradient
                    grad_x_corrected(i,j) = grad_x_estimated(i,j);
                else
                    % Sửa chữa truyền thống
                    if grad_x_wrapped(i,j) > threshold
                        grad_x_corrected(i,j) = grad_x_wrapped(i,j) - 2*pi;
                    else
                        grad_x_corrected(i,j) = grad_x_wrapped(i,j) + 2*pi;
                    end
                end
            end
        end
    end
    
    % Sửa chữa gradient Y
    grad_y_corrected = grad_y_wrapped;
    jump_count_y = 0;
    
    for i = 1:size(grad_y_wrapped, 1)
        for j = 1:size(grad_y_wrapped, 2)
            if abs(grad_y_wrapped(i,j)) > threshold
                jump_count_y = jump_count_y + 1;
                
                if abs(grad_y_estimated(i,j)) < threshold/2
                    grad_y_corrected(i,j) = grad_y_estimated(i,j);
                else
                    if grad_y_wrapped(i,j) > threshold
                        grad_y_corrected(i,j) = grad_y_wrapped(i,j) - 2*pi;
                    else
                        grad_y_corrected(i,j) = grad_y_wrapped(i,j) + 2*pi;
                    end
                end
            end
        end
    end
    
    jump_info.num_jumps_x = jump_count_x;
    jump_info.num_jumps_y = jump_count_y;
end

function confidence_map = calculate_2d_confidence(grad_x_corrected, grad_y_corrected, ...
                                                grad_x_estimated, grad_y_estimated)
    
    % Tính sai lệch gradient
    diff_x = abs(grad_x_corrected - grad_x_estimated);
    diff_y = abs(grad_y_corrected - grad_y_estimated);
    
    % Kết hợp sai lệch
    total_diff = sqrt(diff_x.^2 + diff_y.^2);
    
    % Chuyển thành confidence (0-1)
    confidence_map = exp(-total_diff / 0.7);
    confidence_map = max(confidence_map, 0.1);  % Minimum confidence
end

function unwrapped_surface = perform_2d_unwrapping(wrapped_phase, ...
    grad_x_corrected, grad_y_corrected, estimated_phase, confidence_map, options)
    
    [ny, nx] = size(wrapped_phase);
    unwrapped_surface = zeros(ny, nx);
    
    % Bắt đầu từ góc (1,1)
    unwrapped_surface(1,1) = wrapped_phase(1,1);
    
    % Unwrap hàng đầu tiên
    for j = 2:nx
        weight = options.confidence_weight * confidence_map(1, j-1);
        
        combined_grad = (1 - weight) * grad_x_corrected(1, j-1) + ...
                       weight * (estimated_phase(1,j) - estimated_phase(1,j-1));
        
        unwrapped_surface(1, j) = unwrapped_surface(1, j-1) + combined_grad;
    end
    
    % Unwrap cột đầu tiên
    for i = 2:ny
        weight = options.confidence_weight * confidence_map(i-1, 1);
        
        combined_grad = (1 - weight) * grad_y_corrected(i-1, 1) + ...
                       weight * (estimated_phase(i,1) - estimated_phase(i-1,1));
        
        unwrapped_surface(i, 1) = unwrapped_surface(i-1, 1) + combined_grad;
    end
    
    % Unwrap phần còn lại bằng path-following
    for i = 2:ny
        for j = 2:nx
            % Tính từ hướng X
            weight_x = options.confidence_weight * confidence_map(i, j-1);
            combined_grad_x = (1 - weight_x) * grad_x_corrected(i, j-1) + ...
                             weight_x * (estimated_phase(i,j) - estimated_phase(i,j-1));
            value_from_x = unwrapped_surface(i, j-1) + combined_grad_x;
            
            % Tính từ hướng Y
            weight_y = options.confidence_weight * confidence_map(i-1, j);
            combined_grad_y = (1 - weight_y) * grad_y_corrected(i-1, j) + ...
                             weight_y * (estimated_phase(i,j) - estimated_phase(i-1,j));
            value_from_y = unwrapped_surface(i-1, j) + combined_grad_y;
            
            % Kết hợp hai giá trị với trọng số confidence
            total_confidence = confidence_map(i, j-1) + confidence_map(i-1, j);
            if total_confidence > 0
                w1 = confidence_map(i, j-1) / total_confidence;
                w2 = confidence_map(i-1, j) / total_confidence;
                unwrapped_surface(i, j) = w1 * value_from_x + w2 * value_from_y;
            else
                unwrapped_surface(i, j) = 0.5 * (value_from_x + value_from_y);
            end
        end
    end
end

function quality_map = calculate_quality_map(unwrapped_surface, estimated_phase, confidence_map)
    
    % Tính local quality dựa trên nhiều yếu tố
    local_error = abs(unwrapped_surface - estimated_phase);
    local_smoothness = calculate_local_smoothness(unwrapped_surface);
    
    % Kết hợp các yếu tố
    error_score = 1 ./ (1 + local_error);
    smoothness_score = 1 ./ (1 + local_smoothness);
    
    quality_map = 0.4 * error_score + 0.3 * smoothness_score + 0.3 * confidence_map;
end

function smoothness = calculate_local_smoothness(surface)
    
    [grad_x, grad_y] = gradient(surface);
    [grad_xx, ~] = gradient(grad_x);
    [~, grad_yy] = gradient(grad_y);
    
    % Laplacian như measure của smoothness
    laplacian = abs(grad_xx + grad_yy);
    smoothness = laplacian;
end

function visualize_3d_unwrapping(X, Y, true_phase, wrapped_phase, estimated_phase, ...
                               unwrapped_surface, quality_map, process_info)
    
    fprintf('Tạo visualization...\n');
    
    figure('Position', [50, 50, 1600, 1200], 'Name', '3D Phase Unwrapping Results');
    
    % Tính toán để hiển thị đồng nhất
    all_phases = [true_phase(:); unwrapped_surface(:); estimated_phase(:)];
    phase_limits = [min(all_phases), max(all_phases)];
    
    % 1. True Phase Surface
    subplot(3,3,1);
    surf(X, Y, true_phase, 'EdgeColor', 'none');
    title('True Phase Surface');
    colorbar; clim(phase_limits);
    view(45, 30);
    
    % 2. Wrapped Phase
    subplot(3,3,2);
    surf(X, Y, wrapped_phase, 'EdgeColor', 'none');
    title('Wrapped Phase');
    colorbar; clim([-pi, pi]);
    view(45, 30);
    
    % 3. Estimated Phase
    subplot(3,3,3);
    surf(X, Y, estimated_phase, 'EdgeColor', 'none');
    title('Estimated Phase');
    colorbar; clim(phase_limits);
    view(45, 30);
    
    % 4. Unwrapped Result
    subplot(3,3,4);
    surf(X, Y, unwrapped_surface, 'EdgeColor', 'none');
    title('Unwrapped Result');
    colorbar; clim(phase_limits);
    view(45, 30);
    
    % 5. Quality Map
    subplot(3,3,5);
    surf(X, Y, quality_map, 'EdgeColor', 'none');
    title('Quality Map');
    colorbar; clim([0, 1]);
    view(45, 30);
    
    % 6. Error: Unwrapped vs True
    subplot(3,3,6);
    error_unwrapped = abs(unwrapped_surface - true_phase);
    surf(X, Y, error_unwrapped, 'EdgeColor', 'none');
    title('Error: Unwrapped vs True');
    colorbar;
    view(45, 30);
    
    % 7. Error: Estimated vs True
    subplot(3,3,7);
    error_estimated = abs(estimated_phase - true_phase);
    surf(X, Y, error_estimated, 'EdgeColor', 'none');
    title('Error: Estimated vs True');
    colorbar;
    view(45, 30);
    
    % 8. Cross-section comparison
    subplot(3,3,8);
    mid_row = round(size(true_phase, 1)/2);
    plot(true_phase(mid_row, :), 'g-', 'LineWidth', 2); hold on;
    plot(wrapped_phase(mid_row, :), 'r-', 'LineWidth', 1);
    plot(estimated_phase(mid_row, :), 'b--', 'LineWidth', 1.5);
    plot(unwrapped_surface(mid_row, :), 'm:', 'LineWidth', 2);
    legend('True', 'Wrapped', 'Estimated', 'Unwrapped');
    title('Cross-section (Middle Row)');
    grid on;
    
    % 9. Statistics
    subplot(3,3,9);
    stats = [process_info.avg_confidence, ...
             corr(unwrapped_surface(:), true_phase(:)), ...
             corr(estimated_phase(:), true_phase(:))];
    bar(stats);
    set(gca, 'XTickLabel', {'Avg Confidence', 'Unwrapped Corr', 'Estimated Corr'});
    title('Performance Statistics');
    ylim([0, 1]);
    grid on;
    
    % Tổng kết thông tin
    sgtitle(sprintf('3D Phase Unwrapping - Jumps: %d(X) + %d(Y), Confidence: %.3f', ...
                   process_info.num_jumps_x, process_info.num_jumps_y, process_info.avg_confidence));
end

function evaluate_3d_results(true_phase, estimated_phase, unwrapped_surface)
    
    % Tính các metric
    mse_estimated = mean((estimated_phase(:) - true_phase(:)).^2);
    mse_unwrapped = mean((unwrapped_surface(:) - true_phase(:)).^2);
    
    corr_estimated = corr(estimated_phase(:), true_phase(:));
    corr_unwrapped = corr(unwrapped_surface(:), true_phase(:));
    
    improvement_mse = (mse_estimated - mse_unwrapped) / mse_estimated * 100;
    improvement_corr = (corr_unwrapped - corr_estimated) / corr_estimated * 100;
    
    % In kết quả
    fprintf('MSE Estimated vs True: %.4f\n', mse_estimated);
    fprintf('MSE Unwrapped vs True: %.4f\n', mse_unwrapped);
    fprintf('Improvement in MSE: %.1f%%\n', improvement_mse);
    fprintf('\n');
    fprintf('Correlation Estimated vs True: %.4f\n', corr_estimated);
    fprintf('Correlation Unwrapped vs True: %.4f\n', corr_unwrapped);
    fprintf('Improvement in Correlation: %.1f%%\n', improvement_corr);
    fprintf('\n');
    
    % Đánh giá vùng có vấn đề
    error_threshold = 2;
    problem_pixels_est = sum(abs(estimated_phase(:) - true_phase(:)) > error_threshold);
    problem_pixels_unw = sum(abs(unwrapped_surface(:) - true_phase(:)) > error_threshold);
    
    fprintf('Pixels with error > %.1f:\n', error_threshold);
    fprintf('   Estimated: %d/%d (%.1f%%)\n', problem_pixels_est, numel(true_phase), ...
           problem_pixels_est/numel(true_phase)*100);
    fprintf('   Unwrapped: %d/%d (%.1f%%)\n', problem_pixels_unw, numel(true_phase), ...
           problem_pixels_unw/numel(true_phase)*100);
    
    if improvement_mse > 10
        fprintf('\n✅ UNWRAPPING THÀNH CÔNG - Cải thiện đáng kể!\n');
    elseif improvement_mse > 0
        fprintf('\n✅ UNWRAPPING THÀNH CÔNG - Cải thiện nhẹ\n');
    else
        fprintf('\n⚠️  UNWRAPPING KHÔNG HIỆU QUẢ - Estimated phase có thể không phù hợp\n');
    end
end