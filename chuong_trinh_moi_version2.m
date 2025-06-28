% =========================================================================
% SCRIPT CHÍNH: MÔ PHỎNG VÀ TÁI TẠO PHA TỪ OFF-AXIS HOLOGRAM
% =========================================================================
% Author:
% Date: 2025-06-21
% Description:
%   1. Mô phỏng quá trình tạo ảnh giao thoa (hologram).
%   2. Tái tạo lại mặt pha từ hologram bằng phương pháp lọc Fourier,
%      cho phép người dùng tương tác để chọn vùng phổ.
%   3. Phân tích sai số toàn diện với visualization nâng cao.
% =========================================================================

%% Dọn dẹp môi trường làm việc
clc;
clear;
close all;
%%
%% 1. THIẾT LẬP CÁC THÔNG SỐ MÔ PHỎNG VÀ VẬT LÝ
fprintf('1. Đang thiết lập các thông số...\n');
params = define_simulation_parameters();

%% 2. TẠO LƯỚI TỌA ĐỘ
[X, Y] = meshgrid(1:params.imageSize.X, 1:params.imageSize.Y);

%% 3. TẠO SÓNG VẬT THỂ VÀ SÓNG THAM CHIẾU
fprintf('2. Đang tạo sóng vật thể và sóng tham chiếu...\n');
[Es, phi_ground_truth] = create_object_wave(params, X, Y);
[E0, phi_ref] = create_reference_wave(params, X, Y);

%% 4. MÔ PHỎNG ẢNH GIAO THOA (HOLOGRAM)
fprintf('3. Đang mô phỏng ảnh giao thoa...\n');
hologram = simulate_hologram(Es, E0);

% gans lai 
% hologram = I_refine;
hologram_abs = mat2gray(hologram);
imwrite(hologram_abs, 'hologram.bmp');
% Hiển thị các kết quả mô phỏng ban đầu
plot_simulation_inputs(angle(Es), phi_ground_truth, angle(E0), hologram);

%% 5. TÁI TẠO PHA TỪ HOLOGRAM (CÓ TƯƠNG TÁC)
fprintf('4. Đang tái tạo pha từ hologram...\n');
fprintf('   Vui lòng vẽ một hình chữ nhật quanh phổ bậc +1 và DOUBLE-CLICK để xác nhận.\n');
[wrappedPhase, params] = reconstruct_phase_interactively(hologram, params);
%%
% figure;
% imshow(wrappedPhase, []), title('Select Region to Estimate Tilt');
% region_mask = drawrectangle();  % hoặc dùng drawrectangle, drawpolygon

[wrappedPhase, plane_est] = remove_tilt_from_wrapped(wrappedPhase);
% 
% figure;
% subplot(1,2,1); surf(phi_wrapped); axis image; title('Original Wrapped Phase');
% subplot(1,2,2); surf(phi_corrected); axis image; title('After Tilt Removal');





%% 6. HIỂN THỊ KẾT QUẢ TÁI TẠO BAN ĐẦU
fprintf('5. Đang hiển thị kết quả tái tạo ban đầu...\n');
figure('Name', 'Kết quả Tái tạo Pha Ban đầu');
subplot(1, 2, 1);
surf(phi_ground_truth, 'EdgeColor', 'none');
title('Pha Gốc (Gốc)');
xlabel('x'); ylabel('y'); zlabel('Pha (rad)');
colormap(gca, jet); colorbar; view([45, 30]);
subplot(1, 2, 2);
mesh(wrappedPhase);
title('Pha Wrapped (Sau khi loại bỏ nghiêng)');
xlabel('x'); ylabel('y'); zlabel('Pha (rad)');
colormap(gca, jet); colorbar; view([45, 30]);

%% 7. LẤY DỮ LIỆU TỪ APP (PHA ƯỚC LƯỢNG)
fprintf('6. Đang lấy dữ liệu pha ước lượng từ GUI...\n');
% Chạy ứng dụng GUI để người dùng xử lý và trả về pha ước lượng.
% Giả sử app trả về một bề mặt pha đã được xử lý.
% Trong trường hợp không có app, bạn có thể tạo dữ liệu giả ở đây.
% Ví dụ: phi_est = imgaussfilt(phi_ground_truth, 10);
app = app1_fringe_detection_backup4_6(); 
uiwait(app.UIFigure); 
phi_est = double(app.recons_surface);
phi_est = imgaussfilt(phi_est, 3);
phi_est = phi_est(1:end-1, 1:end-1);
delete(app);

%% 8. CĂN CHỈNH KÍCH THƯỚC VÀ CHUẨN HÓA TÊN BIẾN
% off_set = 10;
% phi_est = phi_est(off_set:end - off_set, off_set:end-off_set);
% fprintf('7. Đang căn chỉnh kích thước các ma trận pha...\n');
% [M1, N1] = size(wrappedPhase);
% [M2, N2] = size(phi_est);
% 
% if M2 <= M1 && N2 <= N1
%     diff_M = M1 - M2;
%     diff_N = N1 - N2;
%     x_start = floor(diff_M / 2) + 1;
%     x_end   = x_start + M2 - 1;
%     y_start = floor(diff_N / 2) + 1;
%     y_end   = y_start + N2 - 1;
%     
%     % Cắt các ma trận lớn hơn để có cùng kích thước với phi_est
%     wrappedPhase_aligned = wrappedPhase(x_start:x_end, y_start:y_end);
%     phi_ground_truth_aligned = phi_ground_truth(x_start:x_end, y_start:y_end);
%     
%     % Giữ nguyên phi_est và đặt tên mới cho nhất quán
%     phi_est_aligned = phi_est;
%     phi_est_aligned = phi_est_aligned -min(phi_est_aligned(:));
% else
%     error('Kích thước của phi_est lớn hơn wrappedPhase. Vui lòng kiểm tra lại dữ liệu đầu vào từ GUI.');
% end

% Giá trị offset để loại bỏ rìa
offset_val = 10; 

 [wrappedPhase_aligned, phi_est_aligned, phi_ground_truth_aligned] = alignAndCropPhaseMaps...
                                            (wrappedPhase, phi_est, phi_ground_truth, offset_val);

%% 9. GIẢI BỌC PHA SỬ DỤNG PHA ƯỚC LƯỢNG
fprintf('8. Đang giải bọc pha bằng phương pháp ước lượng...\n');
% Sử dụng các biến _aligned đã được thống nhất
[finalUnwrappedPhase, kMap] = unwrapUsingEstimate(phi_est_aligned, wrappedPhase_aligned);


% THÊM BƯỚC XỬ LÝ ĐIỂM NHIỄU SPARSE SAU UNWRAPPING
fprintf('\n=== XỬ LÝ ĐIỂM NHIỄU SPARSE TRONG PHA UNWRAPPED ===\n');
finalUnwrappedPhase = correct_sparse_artifacts(finalUnwrappedPhase);
finalUnwrappedPhase = correct_sparse_artifacts(finalUnwrappedPhase);
finalUnwrappedPhase = correct_sparse_artifacts(finalUnwrappedPhase);
finalUnwrappedPhase = correct_sparse_artifacts(finalUnwrappedPhase);
finalUnwrappedPhase = correct_sparse_artifacts(finalUnwrappedPhase);

fprintf('Xử lý điểm nhiễu sparse hoàn tất.\n');

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


%% ========================================================================
% CÁC HÀM PHỤ (LOCAL FUNCTIONS)
% ========================================================================
function params = define_simulation_parameters()
    % Định nghĩa tất cả các tham số cho mô phỏng.
    params.imageSize.X = 1024;
    params.imageSize.Y = 1024;
    
    params.object.amplitude = 15;
    params.object.type = 'gaussian';
    
    % === CẢI TIẾN: TÙY CHỈNH VÂN GIAO THOA TRỰC QUAN ===
    % Thay đổi hai tham số này để kiểm soát giao thoa.
    
    % 1. SỐ LƯỢNG VÂN: Số vân giao thoa bạn muốn thấy trên chiều rộng ảnh.
    %    Tăng giá trị này để vân dày hơn, giảm để vân thưa hơn.
    params.reference.fringe_count = 60; 
    
    % 2. GÓC NGHIÊNG VÂN: Góc nghiêng của các vân giao thoa (đơn vị: độ).
    %    0 độ = vân dọc, 45 độ = vân nghiêng chéo, 90 độ = vân ngang.
    params.reference.fringe_angle_deg = -40;

    % Cấu hình cho vật thể Zernike
    params.object.zernike.indices = [4, 5, 7, 11];
    params.object.zernike.coefficients = [1.5, -0.8, 0.7, -1.2];
end

% -------------------------------------------------------------------------
% function [wrappedPhase, params] = reconstruct_phase_interactively(hologram, params)
% % Tái tạo pha từ hologram bằng cách lọc trong miền tần số.
%     hologramGray = myConvGrayScale(hologram);
%     [numRows, numCols] = size(hologramGray);
%     fourierTransform = fftshift(fft2(hologramGray));
%     
%     figure('Name','Phổ Fourier của Hologram');
%     imshow(log(1 + abs(fourierTransform)), []);
%     title('Chọn phổ bậc +1 (Vẽ HCN và Double-click)');
%     
%     [~, xRec, yRec, widthRec, heightRec] = myDrawRec();
%     
%     roiContent = fourierTransform(yRec:yRec + heightRec - 1, xRec:xRec + widthRec - 1);
%     
%     umax = xRec + widthRec/2 - 1;
%     vmax = yRec + heightRec/2 - 1;
%     u0 = numCols/2; 
%     v0 = numRows/2; 
%     
%     tilt_angle_X = asin(abs(umax - u0) * params.physics.lambda / (numCols * params.physics.delta_xy));
%     tilt_angle_Y = asin(abs(vmax - v0) * params.physics.lambda / (numRows * params.physics.delta_xy));
%     
%     params.reconstruction.tilt_angle_X_deg = rad2deg(tilt_angle_X);
%     params.reconstruction.tilt_angle_Y_deg = rad2deg(tilt_angle_Y);
%     
%     filteredSpectrum = zeros(size(fourierTransform));
%     
%     startRow = round(v0 - heightRec/2) + 1;
%     startCol = round(u0 - widthRec/2) + 1;
%     
%     filteredSpectrum(startRow : startRow + heightRec - 1, startCol : startCol + widthRec - 1) = roiContent;
%     
%     figure('Name','Phổ sau khi xử lý');
%     imshow(log(1 + abs(filteredSpectrum)), []);
%     title('Phổ sau khi lọc và dịch về tâm');
%     
%     finalPhaseComplex = ifft2(ifftshift(filteredSpectrum));
%     wrappedPhase = angle(finalPhaseComplex);
%     wrappedPhase = wrappedPhase';
% end

% -------------------------------------------------------------------------
% -------------------------------------------------------------------------
function [wrappedPhase, params] = reconstruct_phase_interactively(hologram, params)
% Tái tạo pha từ hologram bằng cách lọc trong miền tần số.
% *** CẢI TIẾN: Hiển thị góc nghiêng tính toán được ngay sau khi chọn ROI.

    hologramGray = myConvGrayScale(hologram);
    [numRows, numCols] = size(hologramGray);
    fourierTransform = fftshift(fft2(hologramGray));
    
    % Hiển thị phổ Fourier để người dùng chọn
    figure('Name','Phổ Fourier của Hologram');
    imshow(log(1 + abs(fourierTransform)), []);
    title('Chọn phổ bậc +1 (Vẽ HCN và Double-click)');
    
    % Gọi hàm tương tác để vẽ hình chữ nhật (ROI)
    % Hàm myDrawRec sẽ tự đóng cửa sổ này sau khi chọn xong
    [~, xRec, yRec, widthRec, heightRec] = myDrawRec();
    
    % Trích xuất nội dung phổ trong vùng ROI đã chọn
    roiContent = fourierTransform(yRec:yRec + heightRec - 1, xRec:xRec + widthRec - 1);
    
    % Tính toán tọa độ tâm của vùng ROI (so với tâm ảnh)
    % Tọa độ này trong miền tần số không gian (spatial frequency domain)
    umax = xRec + widthRec/2 - 1;
    vmax = yRec + heightRec/2 - 1;
    u0 = numCols/2; 
    v0 = numRows/2; 
    
    % Tính toán góc nghiêng từ vị trí của phổ bậc +1
    % công thức: sin(theta) = (delta_u * lambda) / (N * delta_x)
    % trong đó delta_u là khoảng cách từ tâm đến tâm ROI theo trục 
    
    % Tạo một ma trận zero để chứa phổ đã được lọc
    filteredSpectrum = zeros(size(fourierTransform));
    
    % Dịch chuyển vùng phổ đã chọn về lại tâm của ma trận
    startRow = round(v0 - heightRec/2) + 1;
    startCol = round(u0 - widthRec/2) + 1;
    filteredSpectrum(startRow : startRow + heightRec - 1, startCol : startCol + widthRec - 1) = roiContent;
    
    % === HIỂN THỊ KẾT QUẢ VỚI THÔNG TIN GÓC ===
    figure('Name','Phổ sau khi xử lý');
    imshow(log(1 + abs(filteredSpectrum)), []);
   

    
    % Thực hiện biến đổi Fourier ngược để tái tạo lại trường sóng phức
    finalPhaseComplex = ifft2(ifftshift(filteredSpectrum));
    
    % Lấy pha từ trường phức (kết quả là pha bị bọc trong khoảng [-pi, pi])
    wrappedPhase = angle(finalPhaseComplex);
%     wrappedPhase = wrappedPhase'; % Chuyển vị để khớp với định dạng tọa độ
end
function [pos, xRec, yRec, widthRec, heightRec] = myDrawRec()
% Cho phép người dùng vẽ một hình chữ nhật (ROI) trên ảnh hiện tại.
    hFig = gcf;
    hROI = drawrectangle();
    centerRec = [hROI.Position(1) + hROI.Position(3)/2, hROI.Position(2) + hROI.Position(4)/2];
    hold on;
    hMarker = plot(centerRec(1), centerRec(2), 'r+', 'MarkerSize', 10, 'LineWidth', 2);
    hold off;
    addlistener(hROI, 'MovingROI', @(src, evt) updateCenterRectangle(src, hMarker));
    
    % Đợi người dùng double-click để xác nhận
    wait(hROI);
    
    pos = round(hROI.Position);
    xRec = pos(1); yRec = pos(2);
    widthRec = pos(3); heightRec = pos(4);
    
    % Đóng cửa sổ sau khi đã chọn xong
    if ishandle(hFig)
        close(hFig);
    end
end

% -------------------------------------------------------------------------
function updateCenterRectangle(roi, centerMarker)
% Cập nhật vị trí dấu cộng ở tâm ROI khi đang di chuyển.
    centerMarker.XData = roi.Position(1) + roi.Position(3)/2;
    centerMarker.YData = roi.Position(2) + roi.Position(4)/2;
    drawnow;
end

% -------------------------------------------------------------------------
function output = myConvGrayScale(inputImage)
% Chuyển ảnh đầu vào sang ảnh grayscale kiểu double.
    if size(inputImage, 3) > 1
        inputImage = rgb2gray(inputImage);
    end
    output = double(inputImage);
end

% -------------------------------------------------------------------------


function [Es, phi_vat] = create_object_wave(params, X, Y)
    % CẢI TIẾN: Tạo trường sóng vật thể phức một cách nhất quán và chính xác.
    % Tất cả các loại vật thể đều được tạo trên lưới tọa độ chuẩn hóa từ X, Y.

    amp = params.object.amplitude;
    
    % --- Chuẩn hóa tọa độ một cách nhất quán ---
    % Chuẩn hóa lưới tọa độ pixel X, Y về một đĩa tròn bán kính 1, tâm tại (0,0).
    % Hệ tọa độ này là tiêu chuẩn cho nhiều hàm, bao gồm cả Zernike.
    [rows, cols] = size(X);
    radius = min(rows, cols) / 2;
    x_norm = (X - cols/2) / radius;
    y_norm = (Y - rows/2) / radius;

    fprintf('   - Đang tạo vật thể loại: ''%s''...\n', params.object.type);
    switch params.object.type
        case 'gaussian'
            % Tạo đỉnh Gaussian tại tâm
            phi_vat = amp * exp(-5 * (x_norm.^2 + y_norm.^2));
            
        case 'gaussian_on_tilt'
            % Tạo đỉnh Gaussian trên một mặt phẳng nghiêng
            gaussian_part = amp * exp(-5 * (x_norm.^2 + y_norm.^2));
            tilt_part = (x_norm + y_norm) * amp / 2;
            phi_vat = gaussian_part + tilt_part;
            
        case 'peaks'
            % SỬA LỖI: Gọi hàm 'peaks' trực tiếp trên tọa độ đã chuẩn hóa.
            % Hàm peaks hoạt động tốt trong khoảng [-3, 3], ta co giãn x_norm.
            phi_vat = amp * peaks(x_norm * 3, y_norm * 3);
            
        case 'zernike'
            % SỬA LỖI: Truyền tọa độ đã chuẩn hóa vào hàm tạo Zernike.
            indices = params.object.zernike.indices;
            coeffs = params.object.zernike.coefficients;
            if numel(indices) ~= numel(coeffs)
                error('Zernike: Số lượng chỉ số và hệ số không khớp.');
            end
            
            [Z_modes, n, m] = tao_da_thuc_zernike(rows, indices);
            
            phi_vat = zeros(size(X)); % Khởi tạo với kích thước của X
            for k = 1:numel(indices)
                %fprintf('     - j=%d (n=%d, m=%d) hệ số %.2f\n', indices(k), n(k), m(k), coeffs(k));
                phi_vat = phi_vat + coeffs(k) * Z_modes(:,:,k);
            end
            phi_vat = amp * phi_vat; % Áp dụng biên độ chung
            
        otherwise
            error("Loại vật thể '%s' không được hỗ trợ.", params.object.type);
    end
    
    % Tạo trường sóng phức từ mặt pha. Kích thước của Es luôn bằng size(X).
    Es = exp(1i * phi_vat);
end

% -------------------------------------------------------------------------
function [E0, phi_ref] = create_reference_wave(params, X, Y)
    % === CẢI TIẾN: Tạo sóng tham chiếu từ SỐ LƯỢNG VÂN và GÓC NGHIÊNG ===
    
    [~, cols] = size(X);
    
    % Lấy các tham số trực quan từ người dùng
    fringe_count = params.reference.fringe_count;
    fringe_angle_deg = params.reference.fringe_angle_deg;
    
    % 1. Tính toán tổng tần số sóng mang (độ lớn của vector tần số)
    %    Tần số này quyết định mật độ (số lượng) vân.
    %    Định nghĩa: fx = fringe_count / image_width
    f_total = fringe_count / cols;
    
    % 2. Tính toán góc của vector tần số
    %    Vector tần số [fx, fy] sẽ vuông góc với hướng của vân giao thoa.
    %    Do đó, góc của nó sẽ là (góc vân + 90 độ).
    angle_vec_rad = deg2rad(fringe_angle_deg + 90);
    
    % 3. Phân rã tổng tần số thành các thành phần fx và fy
    %    Đây là phép chiếu vector trong hình học.
    fx = f_total * cos(angle_vec_rad);
    fy = f_total * sin(angle_vec_rad);
    
    % 4. Tạo mặt phẳng pha nghiêng từ các thành phần tần số
    phi_ref = 2 * pi * (fx * X + fy * Y);
    
    % 5. Tạo sóng tham chiếu phức
    E0 = exp(1i * phi_ref);
end

% -------------------------------------------------------------------------
function I = simulate_hologram(Es, E0)
% Mô phỏng hologram từ sóng vật thể và sóng tham chiếu.
    I = abs(E0 + Es).^2;
end

% -------------------------------------------------------------------------
function plot_simulation_inputs(phase_obj, surf_obj, phase_ref, hologram)
% Hiển thị các kết quả của quá trình mô phỏng.
    figure('Name', 'Kết quả Mô phỏng ban đầu');
    
    subplot(2, 2, 1);
    imagesc(phase_obj); title('Pha sóng vật thể (bọc)');
    axis image; colormap(gca, hsv); colorbar; axis off;
    
    subplot(2, 2, 2);
    surf(surf_obj, 'EdgeColor', 'none'); title('Bề mặt pha vật thể (Gốc)');
    axis image; colormap(gca, jet); colorbar; view([45, 30]);
    
    subplot(2, 2, 3);
    imagesc(phase_ref); title('Pha sóng tham chiếu (bọc)');
    axis image; colormap(gca, hsv); colorbar; axis off;
    
    subplot(2, 2, 4);
    imagesc(hologram); title('Ảnh Hologram mô phỏng');
    axis image; colormap(gca, gray); colorbar; axis off;
end

% -------------------------------------------------------------------------
function [unwrappedPhase, kMap] = unwrapUsingEstimate(estimatedPhase, wrappedPhase)
% Giải bọc pha `wrappedPhase` dựa trên pha ước lượng `estimatedPhase`.
    wrappedEstimate = wrapToPi(estimatedPhase);
    kMap = round((estimatedPhase - wrappedEstimate) / (2*pi));
    unwrappedPhase = wrappedPhase + 2*pi * kMap;
end

% -------------------------------------------------------------------------
function error_metrics = calculate_comprehensive_errors(phi_result, phi_estimate, phi_ground_truth)
% Tính toán các chỉ số sai số.
    error_metrics = struct();
    
    error_final_vs_truth = phi_result - phi_ground_truth;
    error_estimate_vs_truth = phi_estimate - phi_ground_truth;
    error_final_vs_estimate = phi_result - phi_estimate;
    
    error_metrics.rms_final_vs_truth = sqrt(mean(error_final_vs_truth(:).^2));
    error_metrics.rms_estimate_vs_truth = sqrt(mean(error_estimate_vs_truth(:).^2));
    error_metrics.rms_final_vs_estimate = sqrt(mean(error_final_vs_estimate(:).^2));
    
    error_metrics.mae_final_vs_truth = mean(abs(error_final_vs_truth(:)));
    
    truth_range = range(phi_ground_truth(:));
    if truth_range > 0
        error_metrics.psnr = 20 * log10(truth_range / error_metrics.rms_final_vs_truth);
    else
        error_metrics.psnr = Inf;
    end
    
    corr_matrix = corrcoef(phi_result(:), phi_ground_truth(:));
    error_metrics.correlation_final_truth = corr_matrix(1,2);
    
    error_metrics.error_map_final_vs_truth = error_final_vs_truth;
    error_metrics.error_map_estimate_vs_truth = error_estimate_vs_truth;
    error_metrics.error_map_final_vs_estimate = error_final_vs_estimate;
end

% -------------------------------------------------------------------------
function display_error_summary(metrics)
% Hiển thị bảng tóm tắt sai số.
    fprintf('\n--- TÓM TẮT KẾT QUẢ PHÂN TÍCH SAI SỐ ---\n');
    fprintf('So sánh KẾT QUẢ CUỐI CÙNG với Gốc:\n');
    fprintf('  - Sai số RMS (RMS Error) : %.6f rad\n', metrics.rms_final_vs_truth);
    fprintf('  - Sai số Tuyệt đối TB   : %.6f rad\n', metrics.mae_final_vs_truth);
    fprintf('  - Tỷ lệ Tín hiệu/Nhiễu (PSNR) : %.2f dB\n', metrics.psnr);
    fprintf('  - Hệ số tương quan (Corr)   : %.6f\n', metrics.correlation_final_truth);
    fprintf('-------------------------------------------------\n');
    fprintf('So sánh PHA ƯỚC LƯỢNG với Gốc:\n');
    fprintf('  - Sai số RMS (RMS Error) : %.6f rad\n', metrics.rms_estimate_vs_truth);
    fprintf('-------------------------------------------------\n');
    fprintf('So sánh KẾT QUẢ CUỐI CÙNG với PHA ƯỚC LƯỢNG:\n');
    fprintf('  - Sai số RMS (RMS Error) : %.6f rad\n', metrics.rms_final_vs_estimate);
    fprintf('-------------------------------------------------\n');
end

% -------------------------------------------------------------------------
function create_overview_visualization(phi_gt, phi_est, phi_wrapped, phi_final, kMap)
% Tạo visualization tổng quan các bề mặt pha.
    figure('Name', 'Tổng quan các bề mặt Pha', 'Position', [50, 50, 1400, 800]);
    
    sgtitle('So sánh các Bề mặt Pha', 'FontSize', 16, 'FontWeight', 'bold');
    
    subplot(2, 5, 1); surf(phi_gt, 'EdgeColor', 'none'); title('Gốc'); axis tight; view(45, 30); colorbar;
    subplot(2, 5, 2); surf(phi_est, 'EdgeColor', 'none'); title('Pha Ước lượng'); axis tight; view(45, 30); colorbar;
    subplot(2, 5, 3); surf(phi_wrapped, 'EdgeColor', 'none'); title('Pha Wrapped'); axis tight; view(45, 30); colorbar;
    subplot(2, 5, 4); surf(phi_final, 'EdgeColor', 'none'); title('Kết quả Cuối cùng'); axis tight; view(45, 30); colorbar;
    subplot(2, 5, 5); surf(kMap, 'EdgeColor', 'none'); title('Bản đồ K (Fringe Order)'); axis tight; view(45, 30); colormap(gca, parula); colorbar;
    
    subplot(2, 5, 6); imagesc(phi_gt); title('Gốc (2D)'); axis image; colorbar;
    subplot(2, 5, 7); imagesc(phi_est); title('Pha Ước lượng (2D)'); axis image; colorbar;
    subplot(2, 5, 8); imagesc(phi_wrapped); title('Pha Wrapped (2D)'); axis image; colorbar;
    subplot(2, 5, 9); imagesc(phi_final); title('Kết quả Cuối cùng (2D)'); axis image; colorbar;
    subplot(2, 5, 10); imagesc(kMap); title('Bản đồ K (2D)'); axis image; colormap(gca, parula); colorbar;
%
    
       
    figure(); surf(phi_gt, 'EdgeColor', 'none'); title('Gốc'); axis tight; view(45, 30); colorbar;
    figure(); surf(phi_est, 'EdgeColor', 'none'); title('Pha Ước lượng'); axis tight; view(45, 30); colorbar;
     figure(); surf(phi_wrapped, 'EdgeColor', 'none'); title('Pha Wrapped'); axis tight; view(45, 30); colorbar;
     figure(); surf(phi_final, 'EdgeColor', 'none'); title('Kết quả Cuối cùng'); axis tight; view(45, 30); colorbar;
    figure(); surf(kMap, 'EdgeColor', 'none'); title('Bản đồ K (Fringe Order)'); axis tight; view(45, 30); colormap(gca, parula); colorbar;
    
    figure(); imagesc(phi_gt); title('Gốc (2D)'); axis image; colorbar;
     figure(); imagesc(phi_est); title('Pha Ước lượng (2D)'); axis image; colorbar;
    figure(); imagesc(phi_wrapped); title('Pha Wrapped (2D)'); axis image; colorbar;
    figure(); imagesc(phi_final); title('Kết quả Cuối cùng (2D)'); axis image; colorbar;
     figure(); imagesc(kMap); title('Bản đồ K (2D)'); axis image; colormap(gca, parula); colorbar;
end
function create_advanced_error_analysis(phi_final, phi_est, phi_gt, error_metrics)
% Tạo các đồ thị phân tích sai số nâng cao.
    figure('Name', 'Phân tích Sai số Nâng cao', 'Position', [100, 100, 1200, 600]);
    sgtitle('Phân tích Chi tiết Bề mặt và Bản đồ Sai số', 'FontSize', 16, 'FontWeight', 'bold');
    
    % 3D Error Surfaces
    subplot(2, 3, 1);
    surf(error_metrics.error_map_final_vs_truth, 'EdgeColor', 'none');
    title(sprintf('Sai số: Final vs Truth (RMS=%.4f)', error_metrics.rms_final_vs_truth));
    axis tight; colormap(gca, jet); colorbar; view(45, 30);
    
    subplot(2, 3, 2);
    surf(error_metrics.error_map_estimate_vs_truth, 'EdgeColor', 'none');
    title(sprintf('Sai số: Est vs Truth (RMS=%.4f)', error_metrics.rms_estimate_vs_truth));
    axis tight; colormap(gca, jet); colorbar; view(45, 30);
    
    subplot(2, 3, 3);
    surf(error_metrics.error_map_final_vs_estimate, 'EdgeColor', 'none');
    title(sprintf('Sai số: Final vs Est (RMS=%.4f)', error_metrics.rms_final_vs_estimate));
    axis tight; colormap(gca, jet); colorbar; view(45, 30);
    
    % 2D Error Maps
    subplot(2, 3, 4);
    imagesc(error_metrics.error_map_final_vs_truth);
    title('Bản đồ Sai số: Final vs Truth');
    axis image; colormap(gca, jet); colorbar;
    max_err = max(abs(error_metrics.error_map_final_vs_truth(:)));
    if max_err > 0, clim([-max_err max_err]); end
    
    subplot(2, 3, 5);
    imagesc(error_metrics.error_map_estimate_vs_truth);
    title('Bản đồ Sai số: Est vs Truth');
    axis image; colormap(gca, jet); colorbar;
    max_err = max(abs(error_metrics.error_map_estimate_vs_truth(:)));
    if max_err > 0, clim([-max_err max_err]); end
    
    subplot(2, 3, 6);
    imagesc(error_metrics.error_map_final_vs_estimate);
    title('Bản đồ Sai số: Final vs Est');
    axis image; colormap(gca, jet); colorbar;
    max_err = max(abs(error_metrics.error_map_final_vs_estimate(:)));
    if max_err > 0, clim([-max_err max_err]); end
end

function [Z, n_modes, m_modes] = tao_da_thuc_zernike(N, indices)
%TAO_DA_THUC_ZERNIKE_HCN - Tạo đa thức Zernike trên hình chữ nhật (không giới hạn bởi đĩa tròn)
%
% Cú pháp:
%   [Z, n, m] = tao_da_thuc_zernike_HCN(N, indices)
%
% ĐẦU VÀO:
%   N       - Kích thước lưới vuông N x N
%   indices - Vector các chỉ số Noll của các mode Zernike
%
% ĐẦU RA:
%   Z       - 3D matrix N x N x num_modes, mỗi lớp là một mode Zernike
%   n_modes - Bậc xuyên tâm tương ứng
%   m_modes - Bậc phương vị tương ứng

% 1. Tạo lưới tọa độ
[x, y] = meshgrid(linspace(-1, 1, N));
theta = atan2(y, x);             % góc cực
rho = sqrt(x.^2 + y.^2);         % bán kính

% 2. Khởi tạo
num_modes = numel(indices);
Z = zeros(N, N, num_modes);
n_modes = zeros(1, num_modes);
m_modes = zeros(1, num_modes);

% 3. Lặp qua từng chỉ số Noll
for k = 1:num_modes
    j = indices(k);

    % Chuyển đổi từ Noll -> (n, m)
    n = 0;
    while (n+1)*(n+2)/2 < j
        n = n + 1;
    end
    m = j - n*(n+1)/2 - 1;
    if mod(n-m,2) ~= 0
        if mod(n,2)==m
            m = -m;
        else
            m = -m + 1;
        end
    end
    if mod(j,2)==0 && m~=0
        m = -m;
    end

    n_modes(k) = n;
    m_modes(k) = m;

    % Tính R_n^|m|(rho)
    R = zeros(size(rho));
    if mod(n - abs(m), 2) == 0
        for s = 0:((n - abs(m)) / 2)
            num = (-1)^s * factorial(n - s);
            den = factorial(s) * factorial((n + abs(m))/2 - s) * factorial((n - abs(m))/2 - s);
            R = R + (num / den) * rho.^(n - 2*s);
        end
    end

    % Kết hợp với theta
    if m > 0
        Z_temp = R .* cos(m * theta);
    elseif m < 0
        Z_temp = R .* sin(abs(m) * theta);
    else
        Z_temp = R;
    end

    % Chuẩn hóa
    if m == 0
        norm_factor = sqrt(n + 1);
    else
        norm_factor = sqrt(2 * (n + 1));
    end
    Z_temp = norm_factor * Z_temp;

    % KHÔNG mặt nạ đĩa tròn → giữ toàn bộ hình chữ nhật
    Z(:, :, k) = Z_temp;
end
end

function [phi_corrected, plane_est] = remove_tilt_from_wrapped(phi_wrapped)
    % REMOVE_TILT_FROM_WRAPPED_GUI - Vẽ vùng chữ nhật để fit mặt phẳng nghiêng trên pha wrapped
    %
    % Inputs:
    %   phi_wrapped - ảnh pha đã wrap [-pi, pi]
    %
    % Outputs:
    %   phi_corrected - ảnh pha đã loại bỏ mặt phẳng nghiêng (wrapped lại)
    %   plane_est     - mặt phẳng nghiêng ước lượng (a*x + b*y + c)

    [rows, cols] = size(phi_wrapped);
    [X, Y] = meshgrid(1:cols, 1:rows);

    % --- Hiển thị ảnh và chọn vùng bằng hình chữ nhật ---
    figure; imagesc(phi_wrapped); axis image; colormap jet; colorbar;
    title('Draw rectangle to estimate phase tilt');
    h = drawrectangle('Color','r');  % tương tác GUI
    wait(h);
    % Lấy vùng được chọn
    rect_pos = round(h.Position);  % [x, y, w, h]
    x1 = max(1, rect_pos(1));
    y1 = max(1, rect_pos(2));
    x2 = min(cols, x1 + rect_pos(3) - 1);
    y2 = min(rows, y1 + rect_pos(4) - 1);

    % Trích xuất vùng
    phi_region = phi_wrapped(y1:y2, x1:x2);
    [Xr, Yr] = meshgrid(x1:x2, y1:y2);

    % Unwrap vùng nhỏ để khôi phục mặt phẳng
    phi_region_unwrapped = unwrap(unwrap(phi_region, [], 2), [], 1);

    % Fit mặt phẳng tuyến tính: phi = ax + by + c
    A = [Xr(:), Yr(:), ones(numel(Xr),1)];
    coeffs = A \ phi_region_unwrapped(:);
    a = coeffs(1); b = coeffs(2); c = coeffs(3);

    % Tính mặt phẳng toàn ảnh
    plane_est = a*X + b*Y + c;

    % Xoay pha để loại bỏ nghiêng
    phi_corrected = wrapToPi(phi_wrapped - plane_est);

    % Hiển thị kết quả
    figure;
    subplot(1,2,1); imagesc(phi_wrapped); axis image; title('Original Wrapped Phase'); colormap jet; colorbar;
    subplot(1,2,2); imagesc(phi_corrected); axis image; title('After Tilt Removal'); colormap jet; colorbar;
    figure;
    subplot(1, 2, 1);
    mesh(phi_wrapped);
    title('Pha Wrapped (Ban đầu)');
    xlabel('x'); ylabel('y'); zlabel('Pha (rad)');
    colormap(gca, jet); colorbar; view([45, 30]);
    subplot(1, 2, 2);
    mesh(phi_corrected);
    title('Pha Wrapped (Sau khi loại bỏ nghiêng)');
    xlabel('x'); ylabel('y'); zlabel('Pha (rad)');
    colormap(gca, jet); colorbar; view([45, 30]);


end


function corrected_unwrapped_phase = correct_sparse_artifacts(unwrapped_phase_input)
    % Hàm mới: Xử lý các điểm nhiễu sparse (Sparse artifact points)
    % Dựa trên phương pháp lọc trung vị để xác định và hiệu chỉnh các điểm lỗi.
    % Phương pháp này không làm mịn toàn bộ pha mà chỉ hiệu chỉnh các điểm bất thường.

    % Bước 1: Áp dụng bộ lọc trung vị cho pha unwrapped thô
    % Kích thước bộ lọc nhỏ (ví dụ: 3x3) để chủ yếu nhắm vào các điểm 1- hoặc 2-pixel
    filter_size = [15 15]; 
    filtered_unwrapped_phase = medfilt2(unwrapped_phase_input, filter_size);
    
    % Bước 2: Tính toán sự khác biệt về "thứ tự vân" (fringe order differences)
    % Ak(x, y) = Round[ (Phi_m(x, y) - Phi(x, y)) / 2pi ]
    % Phi_m là pha đã lọc, Phi là pha thô
    delta_k = round((filtered_unwrapped_phase - unwrapped_phase_input) / (2*pi));
    
    % Bước 3: Hiệu chỉnh các điểm có thứ tự vân không chính xác
    % ke(x, y) = k(x, y) + Ak(x, y)
    % Điều này tương đương với unwrapped_phase_input + delta_k * 2pi
    corrected_unwrapped_phase = unwrapped_phase_input + delta_k * (2*pi);

    % Ghi chú: Phương pháp này chỉ sửa các điểm nhiễu mà không làm mịn pha
    % như bộ lọc trung vị tiêu chuẩn.
end

function [wrappedPhase_aligned, phi_est_aligned, phi_ground_truth_aligned] = alignAndCropPhaseMaps(wrappedPhase, phi_est, phi_ground_truth, off_set)
% alignAndCropPhaseMaps - Căn chỉnh và cắt các ma trận pha để có cùng kích thước.
%
% Hàm này thực hiện các công việc sau:
% 1. Loại bỏ phần rìa của ma trận pha ước tính (phi_est) dựa vào off_set.
% 2. Kiểm tra kích thước và đảm bảo ma trận ước tính không lớn hơn ma trận gốc.
% 3. Cắt ma trận gốc (wrappedPhase) và pha thực (phi_ground_truth) để có 
%    kích thước bằng với ma trận ước tính đã cắt rìa, lấy vùng trung tâm.
% 4. Chuẩn hóa ma trận ước tính để giá trị nhỏ nhất của nó bằng 0.
%
% Syntax:
%   [wrappedPhase_aligned, phi_est_aligned, phi_ground_truth_aligned] = ...
%       alignAndCropPhaseMaps(wrappedPhase, phi_est, phi_ground_truth, off_set)
%
% Input:
%   wrappedPhase     - Ma trận pha bị wrap (ma trận gốc).
%   phi_est          - Ma trận pha ước tính từ thuật toán.
%   phi_ground_truth - Ma trận pha thực (ground truth) để so sánh.
%   off_set          - (Tùy chọn) Số pixel cần loại bỏ ở mỗi rìa của phi_est. 
%                      Mặc định là 10.
%
% Output:
%   wrappedPhase_aligned     - Ma trận wrappedPhase đã được căn chỉnh.
%   phi_est_aligned          - Ma trận phi_est đã được cắt rìa và chuẩn hóa.
%   phi_ground_truth_aligned - Ma trận phi_ground_truth đã được căn chỉnh.
%

%% 1. KIỂM TRA ĐẦU VÀO VÀ THIẾT LẬP GIÁ TRỊ MẶC ĐỊNH
fprintf('Bắt đầu quá trình căn chỉnh và chuẩn hóa kích thước ma trận pha...\n');

if nargin < 3
    error('Hàm yêu cầu ít nhất 3 đối số đầu vào: wrappedPhase, phi_est, phi_ground_truth.');
end

if nargin < 4 || isempty(off_set)
    off_set = 10; % Gán giá trị mặc định nếu off_set không được cung cấp
    fprintf('  - Không có off_set, sử dụng giá trị mặc định là %d.\n', off_set);
end

% Kiểm tra quan trọng: wrappedPhase và phi_ground_truth phải có cùng kích thước
if ~isequal(size(wrappedPhase), size(phi_ground_truth))
    error('Kích thước của `wrappedPhase` và `phi_ground_truth` phải giống nhau.');
end

%% 2. LOẠI BỎ RÌA CỦA MA TRẬN PHA ƯỚC TÍNH (PHI_EST)
% Việc này thường cần thiết để loại bỏ các sai số ở biên do thuật toán gây ra.
fprintf('  - Cắt bỏ %d pixels ở mỗi rìa của ma trận pha ước tính.\n', off_set);

% Kiểm tra xem offset có quá lớn không
if 2 * off_set >= size(phi_est, 1) || 2 * off_set >= size(phi_est, 2)
    error('Giá trị `off_set` quá lớn so với kích thước của `phi_est`.');
end
phi_est_cropped = phi_est(off_set + 1 : end - off_set, off_set + 1 : end - off_set);

%% 3. CĂN CHỈNH KÍCH THƯỚC CÁC MA TRẬN
fprintf('  - Đang căn chỉnh kích thước các ma trận...\n');

[M1, N1] = size(wrappedPhase);
[M2, N2] = size(phi_est_cropped);

% Đảm bảo ma trận ước tính sau khi cắt không lớn hơn ma trận gốc
if M2 > M1 || N2 > N1
    error('Kích thước của phi_est sau khi cắt (%d x %d) lớn hơn wrappedPhase (%d x %d). Vui lòng kiểm tra lại dữ liệu.', M2, N2, M1, N1);
end

% Nếu kích thước đã bằng nhau, không cần cắt
if M1 == M2 && N1 == N2
    wrappedPhase_aligned = wrappedPhase;
    phi_ground_truth_aligned = phi_ground_truth;
    phi_est_aligned = phi_est_cropped;
else
    % Tính toán vùng trung tâm của ma trận lớn để cắt
    diff_M = M1 - M2;
    diff_N = N1 - N2;
    
    % Tọa độ bắt đầu và kết thúc để cắt ma trận lớn hơn
    % floor() đảm bảo vùng cắt được định vị đúng tâm ngay cả khi chênh lệch là số lẻ
    x_start = floor(diff_M / 2) + 1;
    x_end   = x_start + M2 - 1;
    y_start = floor(diff_N / 2) + 1;
    y_end   = y_start + N2 - 1;
    
    % Thực hiện cắt các ma trận lớn hơn để có cùng kích thước
    wrappedPhase_aligned = wrappedPhase(x_start:x_end, y_start:y_end);
    phi_ground_truth_aligned = phi_ground_truth(x_start:x_end, y_start:y_end);
    
    % Gán ma trận đã cắt rìa vào biến output
    phi_est_aligned = phi_est_cropped;
end

%% 4. CHUẨN HÓA MA TRẬN PHA ƯỚC TÍNH
% Dịch chuyển các giá trị của phi_est_aligned sao cho giá trị nhỏ nhất là 0.
% Điều này hữu ích cho việc hiển thị và tính toán sai số.
fprintf('  - Chuẩn hóa ma trận pha ước tính (min value = 0).\n');
phi_est_aligned = phi_est_aligned - min(phi_est_aligned(:));

fprintf('Hoàn tất căn chỉnh và chuẩn hóa!\n');

end