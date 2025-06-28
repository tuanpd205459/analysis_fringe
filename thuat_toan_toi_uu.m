clc; clear; close all;

%% === THÔNG SỐ ===
filePath = 'C:\Users\admin\Máy tính\Lab thầy Tùng\Code Matlab\analysis fringe\analysis_fringe\data\anh_nham_chuan.bmp';

%% === ĐỌC ẢNH VÀ XỬ LÝ PHA BỌC ===
hologram = imread(filePath);
wrappedPhase = processing.processFourier(hologram); % Hàm do bạn định nghĩa
phi_wrapped = wrappedPhase(1+1:end-1, 1+1:end-1); % Cắt viền nếu cần (offSet = 1)

%% === LẤY DỮ LIỆU PHA ESTIMATE TỪ GUI ===
app = app1_fringe_detection_backup4_6();
uiwait(app.UIFigure);
phi_est = double(app.recons_surface');
delete(app);

% Làm mượt pha estimate
phi_est = imgaussfilt(phi_est, 3);

%% === CĂN CHỈNH KÍCH THƯỚC ===
[M1, N1] = size(phi_wrapped);
[M2, N2] = size(phi_est);

if M2 <= M1 && N2 <= N1
    x_start = floor((M1 - M2)/2) + 1;
    y_start = floor((N1 - N2)/2) + 1;
    phi_wrapped = phi_wrapped(x_start:x_start+M2-1, y_start:y_start+N2-1);
else
    error('Kích thước phi_est lớn hơn phi_wrapped.');
end

%% === UNWRAP PHA BẰNG PHƯƠNG PHÁP PROPAGATION ===
% phi_unwrapped = unwrap_phase_with_estimate(phi_wrapped, phi_est);



[phi_unwrapped, kMap, wrappedEstimate] = unwrapUsingEstimate(phi_est, phi_wrapped);

%% === HIỂN THỊ KẾT QUẢ ===
figure('Name','Pha Estimate');
surf(phi_est, 'EdgeColor', 'none');
xlabel('x'); ylabel('y'); zlabel('\phi_{est}');
title('Pha Estimate'); colormap jet; view([45 30]); colorbar;

figure('Name','Pha Unwrapped');
surf(phi_unwrapped, 'EdgeColor', 'none');
xlabel('x'); ylabel('y'); zlabel('\phi'); 
title('Pha sau giải bọc'); colormap jet; view([45 30]); colorbar;

figure('Name','Sai lệch so với Estimate');
surf(phi_unwrapped - phi_est, 'EdgeColor', 'none');
xlabel('x'); ylabel('y'); zlabel('\Delta\phi'); 
title('Sai lệch pha (modulo 2π)'); colormap jet; view([45 30]); colorbar;

figure('Name','K-map');
surf(kMap, 'EdgeColor', 'none');
xlabel('x'); ylabel('y'); zlabel('\Delta\phi'); 
title('k - map'); colormap jet; view([45 30]); colorbar;


%% %%%


%%



function [Es, phi_vat] = create_object_wave(params, X, Y)
% CREATE_OBJECT_WAVE - Tạo trường sóng vật thể phức dựa trên loại được chọn.
%
% INPUTS:
%   params - Struct chứa các thông số, bao gồm `params.object.type`.
%   X, Y   - Lưới tọa độ.
%
% OUTPUTS:
%   Es      - Trường sóng vật thể phức.
%   phi_vat - Mặt pha gốc (ground truth).

    % Lấy các tham số cần thiết
    amp = params.object.amplitude;
    N = params.imageSize.X;

    % Tạo tọa độ chuẩn hóa (từ -1 đến 1) cho các hàm toán học
    x_norm = (X - N/2) / (N/2);
    y_norm = (Y - N/2) / (N/2);

    % Chọn và tạo mặt pha dựa trên loại vật thể đã định nghĩa trong params
    switch params.object.type
        case 'gaussian'
            % Trường hợp 1: Một đỉnh Gaussian ở tâm
            fprintf('   Đang tạo vật thể: Đỉnh Gaussian...\n');
            phi_vat = amp * exp(-10 * (x_norm.^2 + y_norm.^2));

        case 'gaussian_on_tilt'
            % Trường hợp 2: Đỉnh Gaussian trên nền nghiêng
            fprintf('   Đang tạo vật thể: Đỉnh Gaussian trên nền nghiêng...\n');
            gaussian_part = amp * exp(-(x_norm.^2 + y_norm.^2) / (2 * 0.2^2));
            tilt_part = (x_norm + y_norm) * amp / 2;
            phi_vat = gaussian_part + tilt_part;

        case 'peaks'
            % Trường hợp 3: Hàm "peaks" của MATLAB trên nền nghiêng nhẹ
            fprintf('   Đang tạo vật thể: Hàm "peaks" phức tạp...\n');
            % Lưu ý: hàm peaks() dùng kích thước N, nhưng nền nghiêng
            % nên dùng tọa độ chuẩn hóa để không phụ thuộc kích thước ảnh.
            peaks_part = 2 * peaks(N);
            tilt_part = 2 * x_norm + y_norm; % Nền nghiêng
            phi_vat = peaks_part + tilt_part;

        otherwise
            error("Loại vật thể '%s' không được hỗ trợ. Vui lòng kiểm tra lại trong 'define_simulation_parameters'.", params.object.type);
    end

    % Tạo trường sóng phức từ mặt pha
    Es = exp(1i * phi_vat);
end

function [E0, phi_ref] = create_reference_wave(params, X, Y)
    theta_x_rad = deg2rad(params.reference.theta_x_deg);
    theta_y_rad = deg2rad(params.reference.theta_y_deg);
    k = 2 * pi / params.physics.lambda; % Sử dụng lambda từ physics
    phi_ref = k * (sin(theta_x_rad) * X * params.physics.delta_xy + ...
                   sin(theta_y_rad) * Y * params.physics.delta_xy);
    E0 = exp(1i * phi_ref);
end

function plot_simulation_inputs(phase_obj, surf_obj, phase_ref, hologram)
% Hiển thị các kết quả của quá trình mô phỏng ban đầu.
    figure('Name', 'Kết quả Mô phỏng ban đầu');
    
    % Sóng vật thể
    subplot(2, 2, 1);
    imagesc(phase_obj); title('Pha sóng vật thể (bọc)');
    axis square; colormap(gca, hsv); colorbar; axis off;
    
    subplot(2, 2, 2);
    surf(surf_obj, 'EdgeColor', 'none'); title('Bề mặt pha vật thể (Ground Truth)');
    axis square; colormap(gca, jet); colorbar; view([45, 30]);

    % Sóng tham chiếu
    subplot(2, 2, 3);
    imagesc(phase_ref); title('Pha sóng tham chiếu (bọc)');
    axis square; colormap(gca, hsv); colorbar; axis off;
    
    % Hologram
    subplot(2, 2, 4);
    imagesc(hologram); title('Ảnh Hologram mô phỏng');
    axis square; colormap(gca, gray); colorbar; axis off;
end


function [unwrappedPhase, kMap, wrappedEstimate] = unwrapUsingEstimate(estimatedPhase, wrappedPhase)
    wrappedEstimate = wrapToPi(estimatedPhase);
    kMap = round((estimatedPhase - wrappedEstimate) / (2*pi));
    unwrappedPhase = wrappedPhase + 2*pi * kMap;
end
