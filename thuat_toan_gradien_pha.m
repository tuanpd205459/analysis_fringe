clc; clear; close all;

%% === INPUT & PREPROCESSING ===
filePath = 'C:\Users\admin\Máy tính\Lab thầy Tùng\Code Matlab\analysis fringe\analysis_fringe\data\anh_nham_chuan.bmp';
hologram = imread(filePath);
wrappedPhase = processing.processFourier(hologram);  % Tự định nghĩa
phi_wrapped = wrappedPhase(2:end-1, 2:end-1);  % Cắt viền nếu offSet = 1

% Lấy pha estimate từ app
app = app1_fringe_detection_backup4_6();
uiwait(app.UIFigure);
phi_est = double(app.recons_surface');
delete(app);

% Làm mượt phi_est
phi_est = imgaussfilt(phi_est, 3);

% Đồng bộ kích thước
[M1, N1] = size(phi_wrapped);
[M2, N2] = size(phi_est);
if M2 <= M1 && N2 <= N1
    x_start = floor((M1 - M2)/2) + 1;
    y_start = floor((N1 - N2)/2) + 1;
    phi_wrapped = phi_wrapped(x_start:x_start+M2-1, y_start:y_start+N2-1);
else
    error('phi_est lớn hơn phi_wrapped.');
end

%% === UNWRAP BẰNG PHƯƠNG PHÁP PROPAGATION MỞ RỘNG ===
phi_unwrapped = unwrap_phase_propagation_enhanced(phi_wrapped, phi_est);

%% === HIỂN THỊ KẾT QUẢ ===
figure('Name','Pha Estimate');
surf(phi_est, 'EdgeColor','none'); colormap jet;
xlabel('x'); ylabel('y'); zlabel('\phi_{est}'); title('Pha Estimate'); view([45 30]); colorbar;

figure('Name','Pha Unwrapped');
surf(phi_unwrapped, 'EdgeColor','none'); colormap jet;
xlabel('x'); ylabel('y'); zlabel('\phi_{unwrap}'); title('Pha Unwrapped'); view([45 30]); colorbar;

figure('Name','Sai lệch so với phi_{est}');
surf(mod(phi_unwrapped - phi_est + pi, 2*pi) - pi, 'EdgeColor','none'); colormap jet;
xlabel('x'); ylabel('y'); zlabel('\Delta\phi'); title('Sai lệch modulo 2π'); view([45 30]); colorbar;


%%
function phi_unwrap = unwrap_phase_propagation_enhanced(phi_wrapped, phi_est)
    [M, N] = size(phi_wrapped);
    phi_unwrap = NaN(M, N);

    %% 1. Phát hiện vùng nhảy pha (jump mask)
    [gx, gy] = gradient(phi_wrapped);
    grad_mag = sqrt(gx.^2 + gy.^2);
    jump_mask = grad_mag > pi;

    %% 2. Ước lượng k từ phi_est
    k_map = round((phi_est - phi_wrapped) / (2*pi));
    phi_init = phi_wrapped + 2*pi * k_map;

    %% 3. Tạo vùng seed unwrap tin cậy
    diff = abs(mod(phi_est - phi_wrapped + pi, 2*pi) - pi);
    seed_mask = diff < pi/3 & ~jump_mask;
    phi_unwrap(seed_mask) = phi_init(seed_mask);
    queue = find(seed_mask);

    %% 4. Lan truyền unwrap
    [dx, dy] = deal([0 1 0 -1], [1 0 -1 0]);  % 4 hướng
    while ~isempty(queue)
        idx = queue(1); queue(1) = [];
        [i, j] = ind2sub([M, N], idx);
        for d = 1:4
            ni = i + dx(d); nj = j + dy(d);
            if ni < 1 || nj < 1 || ni > M || nj > N
                continue;
            end
            if jump_mask(ni, nj) || ~isnan(phi_unwrap(ni, nj))
                continue;
            end
            dw = phi_wrapped(ni, nj) - phi_wrapped(i, j);
            if dw > pi
                dw = dw - 2*pi;
            elseif dw < -pi
                dw = dw + 2*pi;
            end
            phi_unwrap(ni, nj) = phi_unwrap(i, j) + dw;
            queue(end+1) = sub2ind([M, N], ni, nj);
        end
    end

    %% 5. Làm mượt nhẹ và điền điểm thiếu
    phi_unwrap = inpaint_nans(phi_unwrap);
    phi_unwrap = medfilt2(phi_unwrap, [3 3]);
end
function A = inpaint_nans(A)
    nan_mask = isnan(A);
    [X, Y] = meshgrid(1:size(A,2), 1:size(A,1));
    A(nan_mask) = griddata(X(~nan_mask), Y(~nan_mask), A(~nan_mask), ...
                           X(nan_mask), Y(nan_mask), 'nearest');
end
