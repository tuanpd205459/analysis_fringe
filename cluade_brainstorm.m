%% Phân tích vấn đề khi Estimated Phase sai lệch lớn
clear; clc; close all;

% Tạo dữ liệu mẫu
t = 0:0.2:10;
true_phase = 2*t;  % Pha thực tăng tuyến tính

% Wrapped phase
wrapped_phase = angle(exp(1i*true_phase));

% Hai trường hợp estimated phase
estimated_good = true_phase + 0.3*randn(size(true_phase));  % Sai lệch nhỏ
estimated_bad = true_phase + 5*sin(0.5*t) + 0.5*randn(size(true_phase));  % Sai lệch lớn
% Chạy phân tích điều kiện
analyze_success_conditions();
%% Test với estimated phase TỐT
fprintf('=== TEST VỚI ESTIMATED PHASE TỐT ===\n');
[result_good, quality_good] = test_unwrapping(wrapped_phase, estimated_good, true_phase);

%% Test với estimated phase XẤU  
fprintf('\n=== TEST VỚI ESTIMATED PHASE XẤU ===\n');
[result_bad, quality_bad] = test_unwrapping(wrapped_phase, estimated_bad, true_phase);

%% So sánh kết quả
fprintf('\n=== SO SÁNH KẾT QUẢ ===\n');
fprintf('Estimated tốt  - MSE: %.4f, Correlation: %.4f\n', ...
        quality_good.mse_vs_true, quality_good.correlation_vs_true);
fprintf('Estimated xấu  - MSE: %.4f, Correlation: %.4f\n', ...
        quality_bad.mse_vs_true, quality_bad.correlation_vs_true);

%% Visualization
figure('Position', [100, 100, 1400, 900]);

% Plot 1: Trường hợp estimated tốt
subplot(2,3,1);
plot(t, true_phase, 'g-', 'LineWidth', 2); hold on;
plot(t, wrapped_phase, 'r-', 'LineWidth', 1);
plot(t, estimated_good, 'b--', 'LineWidth', 1.5);
plot(t, result_good, 'm:', 'LineWidth', 2);
legend('True', 'Wrapped', 'Estimated', 'Result');
title('Trường hợp Estimated TỐT');
ylabel('Phase (rad)');

% Plot 2: Trường hợp estimated xấu
subplot(2,3,2);
plot(t, true_phase, 'g-', 'LineWidth', 2); hold on;
plot(t, wrapped_phase, 'r-', 'LineWidth', 1);
plot(t, estimated_bad, 'b--', 'LineWidth', 1.5);
plot(t, result_bad, 'm:', 'LineWidth', 2);
legend('True', 'Wrapped', 'Estimated', 'Result');
title('Trường hợp Estimated XẤU');
ylabel('Phase (rad)');

% Plot 3: Sai lệch của estimated
subplot(2,3,3);
error_good = abs(estimated_good - true_phase);
error_bad = abs(estimated_bad - true_phase);
plot(t, error_good, 'b-', 'LineWidth', 1.5); hold on;
plot(t, error_bad, 'r-', 'LineWidth', 1.5);
yline(2*pi, 'k--', '2\pi threshold');
legend('Error good', 'Error bad', '2\pi line');
title('Sai lệch của Estimated vs True');
ylabel('Error (rad)');

% Plot 4: Sai lệch của kết quả unwrap
subplot(2,3,4);
result_error_good = abs(result_good - true_phase);
result_error_bad = abs(result_bad - true_phase);
plot(t, result_error_good, 'b-', 'LineWidth', 1.5); hold on;
plot(t, result_error_bad, 'r-', 'LineWidth', 1.5);
legend('Result error (good est.)', 'Result error (bad est.)');
title('Sai lệch của kết quả Unwrap');
ylabel('Error (rad)');

% Plot 5: Gradient comparison
subplot(2,3,5);
grad_true = diff(true_phase);
grad_est_good = diff(estimated_good);
grad_est_bad = diff(estimated_bad);
plot(grad_true, 'g-', 'LineWidth', 2); hold on;
plot(grad_est_good, 'b-', 'LineWidth', 1.5);
plot(grad_est_bad, 'r-', 'LineWidth', 1.5);
legend('True grad', 'Good est. grad', 'Bad est. grad');
title('So sánh Gradient');
ylabel('Gradient');

% Plot 6: Thống kê
subplot(2,3,6);
metrics = [quality_good.correlation_vs_true, quality_bad.correlation_vs_true; ...
           1/(1+quality_good.mse_vs_true), 1/(1+quality_bad.mse_vs_true)];
bar(metrics);
set(gca, 'XTickLabel', {'Correlation', 'Inv MSE'});
legend('Good estimated', 'Bad estimated');
title('So sánh chất lượng');

%% Hàm test unwrapping
function [result, quality] = test_unwrapping(wrapped_phase, estimated_phase, true_phase)
    
    % Unwrap đơn giản bằng cách kết hợp thông tin
    result = zeros(size(wrapped_phase));
    result(1) = wrapped_phase(1);
    
    grad_wrapped = diff(wrapped_phase);
    grad_estimated = diff(estimated_phase);
    
    % Sửa chữa gradient wrapped
    grad_corrected = grad_wrapped;
    for i = 1:length(grad_wrapped)
        if abs(grad_wrapped(i)) > pi*0.8
            % Có nhảy pha, quyết định dựa trên estimated
            if abs(grad_estimated(i)) < pi/2
                % Tin estimated phase
                grad_corrected(i) = grad_estimated(i);
            else
                % Sửa chữa truyền thống
                if grad_wrapped(i) > pi*0.8
                    grad_corrected(i) = grad_wrapped(i) - 2*pi;
                else
                    grad_corrected(i) = grad_wrapped(i) + 2*pi;
                end
            end
        end
    end
    
    % Tích phân để có kết quả
    for i = 2:length(wrapped_phase)
        result(i) = result(i-1) + grad_corrected(i-1);
    end
    
    % Đánh giá chất lượng
    quality.mse_vs_true = mean((result - true_phase).^2);
    quality.correlation_vs_true = corr(result', true_phase');
    
    % Kiểm tra các vùng estimated sai lệch > 2*pi
    est_error = abs(estimated_phase - true_phase);
    bad_regions = sum(est_error > 2*pi);
    quality.bad_regions = bad_regions;
    quality.max_est_error = max(est_error);
    
    fprintf('   Max estimated error: %.2f rad (%.1f*pi)\n', ...
            quality.max_est_error, quality.max_est_error/pi);
    fprintf('   Points with error > 2pi: %d/%d\n', bad_regions, length(estimated_phase));
    fprintf('   Final MSE vs true: %.4f\n', quality.mse_vs_true);
end

%% Phân tích điều kiện thành công
function analyze_success_conditions()
    fprintf('\n=== ĐIỀU KIỆN THÀNH CÔNG ===\n');
    fprintf('1. Estimated phase phải có sai lệch < 2*pi tại MỖI điểm\n');
    fprintf('2. Gradient của estimated phải "gần đúng" với gradient true\n');
    fprintf('3. Không được có nhiều vùng liên tiếp sai lệch lớn\n');
    fprintf('4. SNR của estimated phase phải đủ cao\n');
    
    fprintf('\n=== DẤU HIỆU CẢNH BÁO ===\n');
    fprintf('- Estimated error > pi tại nhiều điểm\n');
    fprintf('- Gradient estimated và wrapped ngược chiều\n');
    fprintf('- Correlation giữa estimated và wrapped < 0.5\n');
    fprintf('- Kết quả unwrap có độ dốc đột ngột bất thường\n');
end

