clc; clear; close all;

%% === THÔNG SỐ ===
filePath = 'C:\Users\admin\Máy tính\Lab thầy Tùng\Code Matlab\analysis fringe\analysis_fringe\data\anh_nham_chuan.bmp';
DPD = 25; he_so = 1; wavelength = 633; offSet = 0;

%% === ĐỌC ẢNH VÀ XỬ LÝ PHA BỌC ===
hologram = imread(filePath);
wrappedPhase = processing.processFourier(hologram); % Hàm bạn định nghĩa
phi_wrapped = wrappedPhase(offSet+1:end-offSet, offSet+1:end-offSet);

%% === LẤY DỮ LIỆU TỪ APP ===
app = app1_fringe_detection_backup4_6();
uiwait(app.UIFigure);
phi_est = (app.recons_surface');
% phi_est = imgaussfilt(phi_est, 3);
delete(app);

%% === CĂN CHỈNH KÍCH THƯỚC ===
[M1, N1] = size(phi_wrapped);
[M2, N2] = size(phi_est);
if M2 <= M1 && N2 <= N1
    diff_M = M1 - M2;
    diff_N = N1 - N2;
    x_start = floor(diff_M / 2) + 1;
    x_end = x_start + M2 - 1;
    y_start = floor(diff_N / 2) + 1;
    y_end = y_start + N2 - 1;
    phi_wrapped = phi_wrapped(x_start:x_end, y_start:y_end);
else
    error('phi_est lớn hơn phi_wrapped — kiểm tra lại dữ liệu đầu vào.');
end

%% === GIẢI PHA BỌC BẰNG TỐI ƯU TOÀN CỤC ===
lambda1 = 0.05; % Trọng số giữa estimate và wrapped
n_iter = 300; % Số lần lặp
lr = 0.05; % Learning rate
lambda2 = 0.01;
step_size = 0.01;
% [Delta_x, Delta_y] = gradient(phi_wrapped); % Tạo Delta_x, Delta_y từ pha đã gói
tol = 1e-6; % Ngưỡng hội tụ
max_iter = 200;

% Gọi hàm giải
% phi_unwrapped = unwrap_using_estimate(phi_wrapped, phi_est);



% Tính gradient đã gói từ phi_wrapped
[dx, dy] = gradient(phi_wrapped);
Delta_x = mod(dx + pi, 2*pi) - pi;
Delta_y = mod(dy + pi, 2*pi) - pi;

% Giải bằng FFT
phi_opt = solve_with_fft_updated(phi_est, Delta_x, Delta_y);
phi_unwrapped =phi_opt;
% Hiển thị kết quả
figure;
subplot(2, 2, 1); surf(phi_est); title('Phi Estimated'); colorbar;
subplot(2, 2, 2); surf(phi_wrapped); title('Phi Wrapped'); colorbar;
subplot(2, 2, 3); surf(Delta_x); title('Delta_x'); colorbar;
subplot(2, 2, 4); surf(phi_opt); title('Phi Optimized'); colorbar;
figure;
surf(phi_opt  - phi_est     ); title("sai lech");

%% === HIỂN THỊ KẾT QUẢ ===
figure('Name','Bề mặt pha estimate');
surf(phi_est, 'EdgeColor', 'none'); 
xlabel('x'); ylabel('y'); zlabel('\phi_{est}'); 
title('Pha Estimate'); 
colormap jet; view([45 30]); colorbar;

figure('Name','Bề mặt pha sau tối ưu');
surf(phi_unwrapped, 'EdgeColor', 'none'); 
xlabel('x'); ylabel('y'); zlabel('\phi'); 
title('Pha sau tối ưu'); 
colormap jet; view([45 30]); colorbar;

figure('Name','Sai lệch so với estimate');
surf(phi_unwrapped - phi_est, 'EdgeColor', 'none'); 
xlabel('x'); ylabel('y'); zlabel('\Delta\phi'); 
title('Sai lệch so với Estimate'); 
colormap jet; view([45 30]); colorbar;



function phi_opt = solve_with_fft_updated(phi_est, Delta_x, Delta_y)
    % Kích thước ma trận
    [M, N] = size(phi_est);
    
    % Tính vectơ hằng số
    rho = zeros(M, N);
    for i = 2:M-1
        for j = 2:N-1
            rho(i,j) = phi_est(i,j) + Delta_x(i,j) - Delta_x(i-1,j) + Delta_y(i,j-1) - Delta_y(i,j);
        end
    end
    % Xử lý biên (giả định biên = 0 để đơn giản)
    rho(1,:) = phi_est(1,:) + Delta_x(1,:);
    rho(M,:) = phi_est(M,:) - Delta_x(M-1,:);
    rho(:,1) = phi_est(:,1) + Delta_y(:,1);
    rho(:,N) = phi_est(:,N) - Delta_y(:,N-1);
    b = rho;
    
    % Áp dụng FFT
    b_hat = fft2(b);
    
    % Tạo ma trận eigenvalue của Laplace trong miền tần số
    [k, l] = meshgrid(0:N-1, 0:M-1);
    lambda = 1 + 4 * (sin(pi * k / N).^2 + sin(pi * l / M).^2); % Eigenvalues of I + L
    
    % Giải trong miền tần số
    phi_hat = b_hat ./ lambda;
    
    % Biến đổi ngược về miền không gian
    phi_opt = ifft2(phi_hat, 'symmetric');
    
    % Đảm bảo giá trị thực
    phi_opt = real(phi_opt);
end