clc;
clear;
close all;

%% ==== THIẾT LẬP THÔNG SỐ ====
amp = 10;
N = 1080;
object_type = 'gaussian_on_tilt';  % Thử với 'gaussian', 'gaussian_on_tilt', hoặc 'peaks'

%% ==== TẠO LƯỚI TỌA ĐỘ ====
[xGrid, yGrid] = meshgrid(1:N, 1:N);
x_norm = (xGrid - N/2) / (N/2);
y_norm = (yGrid - N/2) / (N/2);

%% ==== TẠO MẶT PHA ====
switch object_type
    case 'gaussian'
        phi_vat = amp * exp(-10 * (x_norm.^2 + y_norm.^2));

    case 'gaussian_on_tilt'
        gaussian_part = amp * exp(-(x_norm.^2 + y_norm.^2) / (2 * 0.2^2));
        tilt_part = (x_norm + y_norm) * amp / 2;
        phi_vat = gaussian_part + tilt_part;

    case 'peaks'
        peaks_part = 2 * peaks(N);
        peaks_part = imresize(peaks_part, [N, N]);
        tilt_part = 2 * x_norm + y_norm;
        phi_vat = peaks_part + tilt_part;

    otherwise
        error("Loại vật thể không hỗ trợ.");
end

%% ==== WRAPPING PHA BẰNG 2 CÁCH ====
phi_wrapped_builtin = wrapToPi(phi_vat);                     % Cách 1: dùng MATLAB
phi_wrapped_custom  = mod(phi_vat + pi, 2*pi) - pi;          % Cách 2: thủ công

%% ==== TÍNH SAI KHÁC ====
diff_map = abs(phi_wrapped_builtin - phi_wrapped_custom);
max_diff = max(diff_map(:));
fprintf('Sai lệch cực đại giữa hai phương pháp wrap: %.2e rad\n', max_diff);

%% ==== HIỂN THỊ KẾT QUẢ ====
figure('Name','So sánh 2 cách wrap phase');

subplot(2,2,1);
surf(x_norm, y_norm, phi_vat, 'EdgeColor', 'none');
title('Mặt pha gốc');
xlabel('x'); ylabel('y'); zlabel('\phi'); colorbar; view(3); colormap jet;

subplot(2,2,2);
surf(x_norm, y_norm, phi_wrapped_builtin, 'EdgeColor', 'none');
title('Wrapped phase - wrapToPi');
xlabel('x'); ylabel('y'); zlabel('\phi_{wrapped}'); colorbar; view(3); colormap jet;

subplot(2,2,3);
surf(x_norm, y_norm, phi_wrapped_custom, 'EdgeColor', 'none');
title('Wrapped phase - custom mod');
xlabel('x'); ylabel('y'); zlabel('\phi_{wrapped}'); colorbar; view(3); colormap jet;

subplot(2,2,4);
imagesc(diff_map);
title('Sai khác giữa hai phương pháp');
colorbar;
axis image;
colormap hot;

    phi_est_aligned = phi_est_aligned -min(phi_est_aligned(:));


%% 9. GIẢI BỌC PHA SỬ DỤNG PHA ƯỚC LƯỢNG
fprintf('8. Đang giải bọc pha bằng phương pháp ước lượng...\n');
% Sử dụng các biến _aligned đã được thống nhất
[finalUnwrappedPhase, kMap] = unwrapUsingEstimate(phi_est_aligned, wrappedPhase_aligned);

%% 10. TÍNH TOÁN SAI SỐ TOÀN DIỆN
fprintf('9. Đang tính toán sai số toàn diện...\n');
% Sử dụng các biến _aligned đã được thống nhất
error_metrics = calculate_comprehensive_errors(finalUnwrappedPhase, phi_est_aligned, phi_ground_truth_aligned);

%% 11. HIỂN THỊ KẾT QUẢ SAI SỐ CHI TIẾT
fprintf('10. Đang hiển thị kết quả phân tích sai số...\n');
display_error_summary(error_metrics);

%% 12. TẠO CÁC HÌNH ẢNH PHÂN TÍCH CHI TIẾT
fprintf('11. Đang tạo các hình ảnh phân tích chi tiết...\n');
% Visualization tổng quan - Sử dụng các biến _aligned
create_overview_visualization(phi_ground_truth_aligned, phi_est_aligned, ...
                            wrappedPhase_aligned, finalUnwrappedPhase, kMap);
% Phân tích sai số nâng cao - Sử dụng các biến _aligned
create_advanced_error_analysis(finalUnwrappedPhase, phi_est_aligned, ...
                              phi_ground_truth_aligned, error_metrics);
% % Phân tích thống kê sai số - Sử dụng các biến _aligned
% create_statistical_error_analysis(finalUnwrappedPhase, phi_est_aligned, ...
%                                  phi_ground_truth_aligned);
% % So sánh cross-section - Sử dụng các biến _aligned
% create_cross_section_analysis(finalUnwrappedPhase, phi_est_aligned, ...
%                              phi_ground_truth_aligned);
fprintf('Hoàn thành!\n');

