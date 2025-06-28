% thuật toán mới k = round(wrapped_estimate - estimate)/2pi
% Script chính để chạy ví dụ mở pha

clear; clc; close all;

% Tạo dữ liệu thử nghiệm
t = linspace(0, 4*pi, 100);

% Pha liên tục ban đầu (đây sẽ là pha ước lượng của bạn)
phase_estimate = 3 * t + 0.5 * sin(2*t);

% Pha đã được bao bọc (đây sẽ là pha đo được đã bị bao bọc)
% Thêm một chút nhiễu Gauss
wrapped_phase = wrap_phase(phase_estimate + 0.1 * randn(1, length(t)));

% Áp dụng thuật toán mở pha
[unwrapped_result, k_vals, wrapped_est] = unwrap_phase_with_estimate(...
    phase_estimate, wrapped_phase ...
);

% Vẽ kết quả
figure('Position', [100, 100, 1000, 800]);

% Subplot 1: Pha ước lượng vs Pha ước lượng đã bao bọc
subplot(2, 2, 1);
plot(t, phase_estimate, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Original Estimate');
hold on;
plot(t, wrapped_est, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Wrapped Estimate');
title('Phase Estimate vs Wrapped Estimate');
ylabel('Phase (rad)');
legend;
grid on;
hold off;

% Subplot 2: Các giá trị K (số lần nhảy 2pi)
subplot(2, 2, 2);
plot(t, k_vals, 'g-o', 'MarkerSize', 3, 'DisplayName', 'K values');
title('K values (2\pi jumps)');
ylabel('K');
grid on;

% Subplot 3: Pha đã được bao bọc
subplot(2, 2, 3);
plot(t, wrapped_phase, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Wrapped Phase');
title('Wrapped Phase');
ylabel('Phase (rad)');
xlabel('Time/Index');
legend;
grid on;

% Subplot 4: Kết quả mở pha
subplot(2, 2, 4);
plot(t, phase_estimate, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Original Estimate', 'Color', [0 0 1 0.7]);
hold on;
plot(t, unwrapped_result, 'g-', 'LineWidth', 2, 'DisplayName', 'Unwrapped Result');
title('Unwrapping Result');
ylabel('Phase (rad)');
xlabel('Time/Index');
legend;
grid on;
hold off;

sgtitle('Phase Unwrapping with Estimate in MATLAB');

% In một số thống kê
rms_error = sqrt(mean((phase_estimate - unwrapped_result).^2));
max_k_abs = max(abs(k_vals));
total_jumps = sum(abs(diff(k_vals)));

fprintf('RMS error between estimate and unwrapped: %.4f\n', rms_error);
fprintf('Max |k| value: %.0f\n', max_k_abs);
fprintf('Total 2\pi jumps detected: %d\n', total_jumps);

function wrapped_phase = wrap_phase(phase, method)
% wrap_phase: Bao bọc pha về khoảng [-pi, pi] hoặc [0, 2pi]
%
% Args:
%   phase: Mảng pha đầu vào
%   method: 'symmetric' cho [-pi, pi], 'positive' cho [0, 2pi]
%           Mặc định là 'symmetric'

    if nargin < 2
        method = 'symmetric';
    end

    if strcmp(method, 'symmetric')
        % Bao bọc về [-pi, pi]
        wrapped_phase = angle(exp(1i * phase));
    else
        % Bao bọc về [0, 2pi]
        wrapped_phase = mod(phase, 2 * pi);
    end
end

function [unwrapped_2d, k_values_2d, wrapped_estimate_2d] = unwrap_phase_2d_with_estimate(estimate_2d, wrapped_phase_2d)
% unwrap_phase_2d_with_estimate: Mở pha 2D sử dụng một pha ước lượng
%
% Args:
%   estimate_2d: Mảng pha ước lượng 2D
%   wrapped_phase_2d: Mảng pha đã được bao bọc 2D
%
% Returns:
%   unwrapped_2d: Pha 2D đã được mở
%   k_values_2d: Mảng 2D chứa số lần nhảy 2*pi
%   wrapped_estimate_2d: Pha ước lượng 2D sau khi được bao bọc

    % Bao bọc pha ước lượng 2D
    wrapped_estimate_2d = wrap_phase(estimate_2d);
    
    % Tính toán giá trị k cho mỗi pixel
    k_values_2d = round((estimate_2d - wrapped_estimate_2d) / (2 * pi));
    
    % Mở pha
    unwrapped_2d = wrapped_phase_2d + 2 * pi * k_values_2d;
    
end

function [unwrapped_phase, reliability_mask, k_values] = robust_unwrap_with_estimate(estimate, wrapped_phase, threshold)
% robust_unwrap_with_estimate: Mở pha robust với phát hiện ngoại lệ
%
% Args:
%   estimate: Pha ước lượng
%   wrapped_phase: Pha đo được đã bị bao bọc
%   threshold: Ngưỡng để phát hiện các ước lượng không đáng tin cậy
%              Mặc định là pi/2
%
% Returns:
%   unwrapped_phase: Pha đã được mở
%   reliability_mask: Mặt nạ logic chỉ ra các điểm đáng tin cậy
%   k_values: Mảng chứa số lần nhảy 2*pi

    if nargin < 3
        threshold = pi / 2;
    end
    
    wrapped_estimate = wrap_phase(estimate);
    
    % Kiểm tra độ tin cậy dựa trên sự khác biệt giữa ước lượng đã bao bọc và giá trị đo
    phase_diff = abs(wrap_phase(wrapped_estimate - wrapped_phase));
    reliability_mask = phase_diff < threshold;
    
    % Chỉ sử dụng các ước lượng đáng tin cậy để tính toán k
    k_values = zeros(size(estimate));
    k_values(reliability_mask) = round(...
        (estimate(reliability_mask) - wrapped_estimate(reliability_mask)) / (2 * pi) ...
    );
    
    % Đối với các điểm không đáng tin cậy, sử dụng nội suy hoặc các giá trị lân cận
    unreliable_indices = find(~reliability_mask);
    if ~isempty(unreliable_indices)
        reliable_indices = find(reliability_mask);
        
        % Nội suy tuyến tính đơn giản cho các giá trị k không đáng tin cậy
        if length(reliable_indices) > 1
            k_interp = interp1(...
                reliable_indices, k_values(reliable_indices), unreliable_indices, 'linear', 'extrap'...
            );
            k_values(unreliable_indices) = round(k_interp);
        end
    end
    
    unwrapped_phase = wrapped_phase + 2 * pi * k_values;

end
function [unwrapped_phase, k_values, wrapped_estimate] = unwrap_phase_with_estimate(estimate, wrapped_phase)
% unwrap_phase_with_estimate: Mở pha sử dụng một pha ước lượng và pha đã được bao bọc
%
% Args:
%   estimate: Pha ước lượng ban đầu
%   wrapped_phase: Pha đã được bao bọc
%
% Returns:
%   unwrapped_phase: Pha đã được mở
%   k_values: Số lần nhảy 2*pi
%   wrapped_estimate: Pha ước lượng sau khi được bao bọc

    % Bước 1: Bao bọc pha ước lượng
    wrapped_estimate = wrap_phase(estimate, 'symmetric');
    
    % Bước 2: Tính toán k = round((estimate - wrapped_estimate) / (2π))
    k_values = round((estimate - wrapped_estimate) / (2 * pi));
    
    % Bước 3: Mở pha sử dụng k và pha đã được bao bọc
    unwrapped_phase = wrapped_phase + 2 * pi * k_values;
    
end

