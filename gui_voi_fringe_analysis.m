clc; clear; close all;

%% === THÔNG SỐ ===
filePath = 'C:\Users\admin\Máy tính\Lab thầy Tùng\Code Matlab\analysis fringe\analysis_fringe\data\anh_nham_chuan.bmp';
DPD = 25; he_so = 1; wavelength = 633; offSet = 0;

%% === ĐỌC ẢNH VÀ XỬ LÝ PHA BỌC ===
hologram = imread(filePath);
wrappedPhase = processing.processFourier(hologram);  % Hàm bạn định nghĩa
% wrappedPhase = double(wrappedPhase);
phi_wrapped = wrappedPhase(offSet+1:end-offSet, offSet+1:end-offSet);  % Cắt viền nếu cần
% unwrapped_phase_manual  = unwrap(phi_wrapped, [], 1);     % unwrap theo hàng
% unwrapped_phase_manual = unwrap(unwrapped_phase_manual, [], 2);     % rồi theo cột
% unwrapped_phase_manual= unwrap2D_simple(phi_wrapped);
methodGroup = 'poisson';
methodType ='';
% unwrapped_phase_manual = unwrap2D_simple(phi_wrapped);
unwrapped_phase_manual = unwrapping.unwrapPhase(phi_wrapped, methodGroup);
% unwrapped_phase_manual = unwrap(phi_wrapped, [], 2);     % unwrap theo hàng
% unwrapped_phase_manual = unwrap(unwrapped_phase_manual, [], 1);     % rồi theo cột
%
figure('Name','Bề mặt pha thu được từ unwrapped thu cong');
surf(unwrapped_phase_manual, 'EdgeColor', 'none');
xlabel('x (px)'); ylabel('y (px)'); zlabel('Pha (rad)');
title('Bề mặt pha thu được tu unwrapped thu cong');
colormap jet; colorbar; view([45 30]);

%% === LẤY DỮ LIỆU TỪ APP ===
app = app1_fringe_detection_backup4_6();  % Gọi GUI
uiwait(app.UIFigure);                     % Đợi người dùng thao tác
phi_est = double(app.recons_surface');    % Pha ước lượng từ GUI
phi_est = imgaussfilt(phi_est, 3);  % Gaussian filter để tránh nhiễu cục bộ

delete(app);                              % Xoá app sau khi lấy dữ liệu

%% === CĂN CHỈNH KÍCH THƯỚC PHA ===
[M1, N1] = size(phi_wrapped);
[M2, N2] = size(phi_est);

if M2 <= M1 && N2 <= N1
    diff_M = M1 - M2;
    diff_N = N1 - N2;
    x_start = floor(diff_M / 2) + 1;
    x_end   = x_start + M2 - 1;
    y_start = floor(diff_N / 2) + 1;
    y_end   = y_start + N2 - 1;
    phi_wrapped = phi_wrapped(x_start:x_end, y_start:y_end);
else
    error('phi_est lớn hơn phi_wrapped — kiểm tra lại dữ liệu đầu vào.');
end

%% === GIẢI PHA BỌC VỚI PHA ƯỚC LƯỢNG ===
phi_unwrapped = unwrap_phase_est_wrap(phi_wrapped, phi_est, ...
    0.5, ...        % Tin cậy lệch < 0.5 rad
    5,   ...        % Median filter 5x5
    pi);            % Gradient threshold = pi

%% === HIỂN THỊ KẾT QUẢ 3D ===
figure('Name','Bề mặt pha thu được từ GUI');
surf(phi_est, 'EdgeColor', 'none');
xlabel('x (px)'); ylabel('y (px)'); zlabel('Pha (rad)');
title('Bề mặt pha thu được từ GUI');
colormap jet; colorbar; view([45 30]);

figure('Name','Pha sau giải bọc & tối ưu');
surf(phi_unwrapped, 'EdgeColor', 'none');
xlabel('x (px)'); ylabel('y (px)'); zlabel('Pha (rad)');
title('Bề mặt pha sau unwrap');
colormap jet; colorbar; view([45 30]);

figure('Name','Sai lệch giữa pha estimate và kết quả');
surf(phi_unwrapped - phi_est, 'EdgeColor', 'none');
xlabel('x'); ylabel('y'); zlabel('\Delta \phi');
title('Sai lệch so với estimate');
colormap jet; colorbar; view([45 30]);


function phi_unwrapped_final = unwrap_phase_est_wrap(phi_wrapped, phi_est, threshold, medfilt_size, gradient_thresh)
% Giải pha có hỗ trợ từ pha estimate
% Input:
%   phi_wrapped     - ảnh pha bọc (-pi, pi)
%   phi_est         - pha ước lượng (continuous)
%   threshold       - ngưỡng độ lệch tin cậy để giữ K
%   medfilt_size    - kích thước lọc median
%   gradient_thresh - ngưỡng gradient để phát hiện vùng gián đoạn

% --- 1. Tính bội số K từ phi_est
delta = phi_est - wrapToPi(phi_est);
K_est = round(delta / (2 * pi));

% --- 2. Mặt nạ tin cậy
%     reliable_mask = abs(delta) < threshold;
%     K = zeros(size(phi_wrapped));
%     K(reliable_mask) = K_est(reliable_mask);
K = K_est;
% --- 3. Pha unwrap sơ bộ
phi_unwrapped = phi_wrapped + 2 * pi * K;

% --- 4. So với unwrap không gian (2 chiều)
phi_spatial = unwrap(phi_wrapped, [], 2);     % unwrap theo hàng
phi_spatial = unwrap(phi_spatial, [], 1);     % rồi theo cột
methodGroup = 'poisson';
methodType ='';
% phi_spatial = unwrapping.unwrapPhase(phi_wrapped, methodGroup);
% phi_spatial = unwrap(phi_wrapped);

residual = abs(phi_unwrapped - phi_spatial);

% --- 5. Vá các điểm sai lớn
bad_mask = residual > 2 * pi;
phi_unwrapped(bad_mask) = phi_spatial(bad_mask);

% --- 6. Làm mượt
phi_unwrapped_filtered = medfilt2(phi_unwrapped, [medfilt_size medfilt_size]);

% --- 7. Đánh giá liên tục
[Gx, Gy] = gradient(phi_unwrapped_filtered);
discontinuities = abs(Gx) > gradient_thresh | abs(Gy) > gradient_thresh;

% --- 8. Hiển thị trung gian
figure('Name','Trung gian unwrap (3D)');

subplot(2,3,1);
surf(phi_wrapped, 'EdgeColor', 'none'); title('Pha wrapped');
xlabel('x'); ylabel('y'); zlabel('\phi'); colormap jet; view([45 30]); colorbar;

subplot(2,3,2);
surf(phi_est, 'EdgeColor', 'none'); title('Pha estimate');
xlabel('x'); ylabel('y'); zlabel('\phi_{est}'); colormap jet; view([45 30]); colorbar;

subplot(2,3,3);
surf(K, 'EdgeColor', 'none'); title('Fringe Order K');
xlabel('x'); ylabel('y'); zlabel('K'); colormap parula; view([45 30]); colorbar;

subplot(2,3,4);
surf(phi_unwrapped, 'EdgeColor', 'none'); title('Unwrap sơ bộ');
xlabel('x'); ylabel('y'); zlabel('\phi_{unwrap}'); colormap jet; view([45 30]); colorbar;

subplot(2,3,5);
surf(phi_unwrapped_filtered, 'EdgeColor', 'none'); title('Sau lọc median');
xlabel('x'); ylabel('y'); zlabel('\phi_{filtered}'); colormap jet; view([45 30]); colorbar;

subplot(2,3,6);
surf(double(discontinuities), 'EdgeColor', 'none'); title('Gradient mạnh');
xlabel('x'); ylabel('y'); zlabel('mask'); colormap hot; view([45 30]); colorbar;

% --- 9. Trả kết quả
phi_unwrapped_final = phi_unwrapped_filtered;
end

