% =========================================================================
% SCRIPT CHÍNH: MÔ PHỎNG VÀ TÁI TẠO PHA TỪ OFF-AXIS HOLOGRAM
% =========================================================================
% thêm lặp antifact 
% =========================================================================
%% Dọn dẹp môi trường làm việc
clc;
clear;
close all;

%% 1. THIẾT LẬP CÁC THÔNG SỐ MÔ PHỎNG VÀ VẬT LÝ
fprintf('1. Đang thiết lập các thông số...\n');
params = define_simulation_parameters();

%% 2. TẠO LƯỚI TỌA ĐỘ
x_vec = linspace(-1, 1, params.imageSize.X); 
y_vec = linspace(-1, 1, params.imageSize.Y);
[X, Y] = meshgrid(x_vec, y_vec);
% [X, Y] = meshgrid(1:params.imageSize.X, 1:params.imageSize.Y);

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
plot_simulation_inputs(phi_ground_truth, phi_ref, hologram);

%% 5. TÁI TẠO PHA TỪ HOLOGRAM (CÓ TƯƠNG TÁC)
fprintf('4. Đang tái tạo pha từ hologram...\n');
fprintf('   Vui lòng vẽ một hình chữ nhật quanh phổ bậc +1 và DOUBLE-CLICK để xác nhận.\n');
[wrappedPhase, params] = reconstruct_phase_interactively(hologram, params);
% [wrappedPhase, params] = reconstruct_phase_auto(hologram, params);


%% Loại bỏ nghiêng ở mặt wrapped phase
% Hỏi người dùng có muốn xoá góc nghiêng không - Wrapped Phase
choice = questdlg('Bạn có muốn loại bỏ góc nghiêng không:?', ...
    'Xác nhận', ...
    'Có', 'Không', 'Có');
if strcmp(choice, 'Có')
    [wrappedPhase, ~] = remove_tilt_from_wrapped2(wrappedPhase);
    
%         [wrappedPhase, ~] = remove_plane_manual2(wrappedPhase);

end


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

% Loại bỏ bề mặt nghiêng - thủ công
% Hỏi người dùng có muốn xoá góc nghiêng không - Wrapped Phase
choice = questdlg('Bạn có muốn loại bỏ góc nghiêng không:?', ...
    'Xác nhận', ...
    'Có', 'Không', 'Có');
if strcmp(choice, 'Có')
        [phi_est, ~] = remove_plane_manual(phi_est);
%     [phi_est,~] = remove_tilt_from_wrapped2(phi_est);
end


%% 8. CĂN CHỈNH KÍCH THƯỚC VÀ CHUẨN HÓA TÊN BIẾN


% Giá trị offset để loại bỏ rìa
offset_val = 10;
[wrappedPhase_aligned, phi_est_aligned, phi_ground_truth_aligned] = alignAndCropPhaseMaps...
    (wrappedPhase, phi_est, phi_ground_truth, offset_val);

%% 9. GIẢI BỌC PHA SỬ DỤNG PHA ƯỚC LƯỢNG
fprintf('8. Đang giải Wrapped pha bằng phương pháp ước lượng...\n');
% Sử dụng các biến _aligned đã được thống nhất
[finalUnwrappedPhase, kMap] = unwrapUsingEstimate(phi_est_aligned, wrappedPhase_aligned);

figure;
subplot(1,2,1); 
surf(finalUnwrappedPhase,"EdgeColor","none"); colorbar;
title("Final truoc khi refine");
subplot(1,2,2);
surf(kMap,"EdgeColor","none"); 
title("Final truoc khi refine");
% THÊM BƯỚC XỬ LÝ ĐIỂM NHIỄU SPARSE SAU UNWRAPPING
fprintf('\n=== XỬ LÝ ĐIỂM NHIỄU SPARSE TRONG PHA UNWRAPPED ===\n');

[finalUnwrappedPhase, iter, hist] = correct_sparse_artifacts_iterative(finalUnwrappedPhase, ...
    'BoundaryCondition', 'symmetric', 'BoundaryWidth', 5, 'MaxIterations', 150);
plot_convergence_analysis(hist);


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



%% Tuỳ chỉnh lưu ảnh
figs = findall(0, 'Type', 'figure');
% Hỏi người dùng có muốn lưu không
choice = questdlg('Bạn có muốn lưu figure ?', ...
    'Xác nhận lưu', ...
    'Có', 'Không', 'Có');

if strcmp(choice, 'Có')
    for i = 1:length(figs)
        fig = figs(i);
        figure(fig);  % Hiển thị figure lên trước khi hỏi

        % Lấy tên figure
        name = get(fig, 'Name');
        if isempty(name)
            name = sprintf('Figure_%d', fig.Number);
        end

        % Xử lý tên cho hợp lệ làm tên file
        name_valid = matlab.lang.makeValidName(name);
        filename = [name_valid '.png'];
        saveas(fig, filename);
        disp(['Đã lưu: ' filename]);

    end
end

fprintf('Hoàn thành!\n');


%% ========================================================================
% CÁC HÀM PHỤ (LOCAL FUNCTIONS)
% ========================================================================
% function params = define_simulation_parameters()
%     % Định nghĩa tất cả các tham số mô phỏng và vật lý.
%     params.imageSize.X = 1080;
%     params.imageSize.Y = 1440;
%     params.object.amplitude = 10;
%     params.reference.theta_x_deg = 10; % Góc nghiêng ban đầu (độ)
%     params.reference.theta_y_deg = 10; % Góc nghiêng ban đầu (độ)
% 
%     % Các loại vật thể có thể chọn: 'gaussian', 'gaussian_on_tilt',
%     % 'peaks', 'zernike', "sinusoidal',"concentric_rings',"spiral"
%     params.object.type = 'sinusoidal';
% 
%     % --- THÊM MỚI: Cấu hình cho vật thể Zernike ---
%     % Chỉ số Noll: j=1(piston), j=2(tip), j=3(tilt), j=4(defocus),
%     % j=5,6(astigmatism), j=7,8(coma), j=9,10(trefoil), j=11(spherical)...
%     params.object.zernike.indices = [4, 5, 6, 7, 8, 11];
%     % Hệ số (coefficients) tương ứng cho mỗi đa thức Zernike
%     % Đây là "trọng số" của mỗi loại quang sai.
%     params.object.zernike.coefficients = [1.5, -0.8, 0.6, 0.7, -0.5, -1.2];
% 
% 
%     params.physics.lambda = 5;  
%     params.physics.delta_xy = 0.5;   
% end
% 
% % -------------------------------------------------------------------------
% function [Es, phi_vat] = create_object_wave(params, X, Y)
%     % Tạo trường sóng vật thể phức từ mặt pha giả lập.
%     amp = params.object.amplitude;
% 
%     % Chuẩn hóa X và Y từ -1 đến 1
%     X_norm = (X - mean([min(X(:)), max(X(:))])) / (max(X(:)) - min(X(:)));
%     Y_norm = (Y - mean([min(Y(:)), max(Y(:))])) / (max(Y(:)) - min(Y(:)));
% 
%     switch params.object.type
%         case 'gaussian'
%             fprintf('   Đang tạo vật thể: Đỉnh Gaussian...\n');
%             phi_vat = amp * exp(-10 * (X_norm.^2 + Y_norm.^2));
%         case 'gaussian_on_tilt'
%             fprintf('   Đang tạo vật thể: Gaussian trên nền nghiêng...\n');
%             gaussian_part = amp * exp(-(X_norm.^2 + Y_norm.^2) / (2 * 0.2^2));
%             tilt_part = (X_norm + Y_norm) * amp / 2;
%             phi_vat = gaussian_part + tilt_part;
%         case 'peaks'
%             fprintf('   Đang tạo vật thể: Hàm "peaks"...\n');
%             sz = size(X);
%             phi_vat = imresize(peaks(max(sz)), sz);  % resize peaks cho khớp ảnh
%         case 'zernike'
%             fprintf('   Đang tạo vật thể: Zernike modes...\n');
%             indices = params.object.zernike.indices;
%             coeffs = params.object.zernike.coefficients;
% 
%             if numel(indices) ~= numel(coeffs)
%                 error('Zernike: số chỉ số và hệ số không khớp.');
%             end
% 
%             N = size(X, 1);  % assume vuông
%             [Z_modes, n, m] = tao_da_thuc_zernike(N, indices);
%             phi_vat = zeros(N, N);
%             for k = 1:numel(indices)
%                 fprintf('     - j=%d (n=%d, m=%d) hệ số %.2f\n', indices(k), n(k), m(k), coeffs(k));
%                 phi_vat = phi_vat + coeffs(k) * Z_modes(:,:,k);
%             end
%             phi_vat = amp * phi_vat;
%             case 'sinusoidal'
%             fprintf('   Đang tạo vật thể: Bề mặt hình sin tuần hoàn...\n');
%             % Tham số sóng sin (có thể điều chỉnh nếu muốn)
%             freq_x = 5;   % số chu kỳ theo trục X
%             freq_y = 0;   % số chu kỳ theo trục Y
% 
%             % Tạo pha hình sin 2D
%             phi_vat = amp * sin(2 * pi * freq_x * X_norm) .* cos(2 * pi * freq_y * Y_norm);
%         case 'concentric_rings'
%             fprintf('   Đang tạo vật thể: Vòng tròn đồng tâm...\n');
%             R = sqrt(X_norm.^2 + Y_norm.^2);
%             rings = floor(R * 10);  % số vòng đồng tâm
%             phi_vat = amp * mod(2 * pi * rings, 2*pi);
%         case 'spiral'
%             fprintf('   Đang tạo vật thể: Bề mặt pha xoắn ốc...\n');
%             R = sqrt(X_norm.^2 + Y_norm.^2);
%             Theta = atan2(Y_norm, X_norm);
%             % Pha xoắn: số chu kỳ phụ thuộc vào tham số scale
%             num_turns = 5;  % số vòng xoắn
%             phi_vat = amp * mod(num_turns * Theta + R * pi, 2*pi);
%             phi_vat = phi_vat - mean(phi_vat(:));  % trung bình về 0
% 
%         otherwise
%             error("Loại vật thể '%s' không hỗ trợ.", params.object.type);
%     end
% 
%     Es = exp(1i * phi_vat);
% end
% 
% 
% function [E0, phi_ref] = create_reference_wave(params, Xa, Ya)
%     % Tạo sóng tham chiếu nghiêng theo cả trục X và Y (off-axis)
% 
%     % Tham số vật lý
%     lambda = params.physics.lambda;  % bước sóng
%     k = 2 * pi / lambda;             % số sóng
% 
%     % Góc nghiêng theo trục x và y (đơn vị độ -> rad)
%     theta_x = deg2rad(params.reference.theta_x_deg);
%     theta_y = deg2rad(params.reference.theta_y_deg);
% 
%     % Thành phần vector sóng
%     kSinThetaX = k * sin(theta_x);
%     kSinThetaY = k * sin(theta_y);
% 
%     % Pha tổng theo cả hai trục
%     phi_ref = kSinThetaX * Xa + kSinThetaY * Ya;
% 
%     % Sóng tham chiếu
%     E0 = exp(1i * phi_ref);
% end
% 
% % -------------------------------------------------------------------------
% function I = simulate_hologram(Es, E0)
% % Mô phỏng hologram từ sóng vật thể và sóng tham chiếu.
%     I = abs(E0 + Es).^2;
% end
function params = define_simulation_parameters()
    % Định nghĩa tất cả các tham số mô phỏng.
    params.imageSize.X = 1080;
    params.imageSize.Y = 1080;
    
    params.lambda = 0.1;
    % Biên độ (độ "sâu"/cao của pha vật thể)
    params.object.amplitude = 10;
    
    params.object.theta_x = 20;                      % Góc nghiêng theo trục x (độ)
    params.object.theta_y = 20;                      % Góc nghiêng theo trục y (độ)

    % Tham số sóng tham chiếu (thay cho góc vật lý)
    % Các giá trị này kiểm soát trực tiếp số lượng vân giao thoa
    params.reference.freq_x = 20; % Số vân giao thoa theo chiều ngang
    params.reference.freq_y = 10; % Số vân giao thoa theo chiều dọc


    % Tùy chọn: Thay đổi bán kính nếu cần
% params.filter_radius = 60; 
% params.dc_suppression_radius = 30;

    % Các loại vật thể: 'gaussian', 'peaks', 'zernike', "sinusoidal",
    %                                   "concentric_rings",
    %                                   "spiral",'custom'
    params.object.type = 'sinusoidal';


    % Chỉ định các đa thức bạn muốn qua số thứ tự của chúng
    params.object.zernike.indices = [5;   % j=5 là Defocus
        8;   % j=8 là Vertical Coma
        13]; % j=13 là Primary Spherical

    % Các hệ số tương ứng
    params.object.zernike.coefficients = [0.8; -0.4; 0.35];

end

% -------------------------------------------------------------------------

function [Es, phi_vat] = create_object_wave(params, X_norm, Y_norm)
    % Tạo trường sóng vật thể từ pha, sử dụng tọa độ đã chuẩn hóa [-1, 1].
    amp = params.object.amplitude;

    switch params.object.type
        case 'gaussian'
            phi_vat = amp * exp(-5 * (X_norm.^2 + Y_norm.^2));
        case 'peaks'
            % Hàm peaks hoạt động tốt trên lưới tọa độ [-3, 3]
            % Chúng ta co giãn X_norm, Y_norm cho phù hợp
            phi_vat = amp/8 * peaks(3*X_norm, 3*Y_norm); 
        case 'zernike'
            fprintf('   Đang tạo vật thể: Zernike modes...\n');
            indices = params.object.zernike.indices;
            coeffs = params.object.zernike.coefficients;

            if numel(indices) ~= numel(coeffs)
                error('Zernike: số chỉ số và hệ số không khớp.');
            end
            N = size(X_norm, 1);
            % Tái tạo bề mặt
            phi_vat = reconstructZernikeAdvanced(indices, coeffs, N);
            phi_vat = amp * phi_vat;
        case 'sinusoidal'
            freq_x = 2;
            freq_y = 0;
            phi_vat = amp * sin(2 * pi * freq_x * X_norm) .* cos(2 * pi * freq_y * Y_norm);
        case 'concentric_rings'
            R = sqrt(X_norm.^2 + Y_norm.^2);
            rings = floor(R * 15); % 15 là số lượng vòng
            phi_vat = amp * sin(rings);
        case 'spiral'
            R = sqrt(X_norm.^2 + Y_norm.^2);
            Theta = atan2(Y_norm, X_norm);
            num_turns = 5;
            phi_vat = amp * (num_turns * Theta + 2*pi*R);
        case 'custom'
            P1 = 20 * exp(-((X_norm-0.4).^2 + (Y_norm-0.4).^2) / 0.1); % Đỉnh lồi
            P2 = -18 * exp(-((X_norm+0.3).^2 + (Y_norm+0.2).^2) / 0.08); % Đỉnh lõm
            phi_vat = P1+ P2;
        otherwise
            error("Loại vật thể '%s' không hỗ trợ.", params.object.type);
    end
    
    % Tạo trường sóng phức từ pha
    Es = exp(1i * phi_vat);
end

% -------------------------------------------------------------------------

function [E0, phi_ref] = create_reference_wave(params, Xa, Ya)
theta_x_deg =   params.object.theta_x;
theta_y_deg = params.object.theta_y;

A_ref = 1.0;          % Biên độ sóng tham chiếu (thường là 1)

% --- TỰ ĐỘNG TÍNH TOÁN TẦN SỐ KHÔNG GIAN ---
% Chuyển đổi góc từ độ sang radian để tính toán
theta_x_rad = theta_x_deg * pi / 180;
theta_y_rad = theta_y_deg * pi / 180;

% Tính tần số không gian (fx, fy) từ góc nghiêng và bước sóng
% Công thức vật lý: fx = sin(theta_x) / lambda
fx = sin(theta_x_rad) / params.lambda;
fy = sin(theta_y_rad) / params.lambda;

k = 2 * pi / params.lambda;                         % Số sóng
% kSinThetaX = k * sin(theta_x);
% kSinThetaY = k * sin(theta_y_deg);
% % kSinTheta = sqrt(kSinThetaY*kSinThetaY + kSinThetaX*kSinThetaX);

% phi_ref = kSinThetaX * Xa + kSinThetaY * Ya;  % Pha tổng
phi_ref = k * (fx * Xa + fy * Ya);  % Pha tổng

E0 = A_ref * exp(1i * phi_ref);                       % Sóng tham chiếu nghiêng theo cả hai trục

end
% -------------------------------------------------------------------------

function I = simulate_hologram(Es, E0)
    % Mô phỏng hologram từ sóng vật thể và sóng tham chiếu.
    % Công thức này không thay đổi.
    % =================== THÊM NHIỄU VÀO ĐÂY ===================
    % --- TÙY CHỌN 1: NHIỄU CỘNG GAUSS ---
    I = abs(E0 + Es).^2;
    hologram  = I;
    noise_level = 0; % Điều chỉnh độ lớn của nhiễu
    noise = noise_level *  randn(size(hologram));
    hologram_noisy = hologram + noise;
    fprintf('   -> Đã thêm nhiễu cộng Gauss vào hologram.\n');
    I = hologram_noisy;
end
%--------------------------------------------------------------------------
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

function [wrappedPhase, params] = reconstruct_phase_auto(hologram, params)
% Tái tạo pha từ hologram bằng cách lọc trong miền tần số với lựa chọn tự động.
%
% Chức năng sẽ tự động tìm phổ bậc +1 ở nửa trên của miền tần số,
% tạo một bộ lọc tròn và tiến hành tái tạo pha.
%
% Tham số (params) có thể chứa:
% params.filter_radius: Bán kính của bộ lọc tròn trong miền tần số (mặc định: 50).
% params.dc_suppression_radius: Bán kính để loại bỏ thành phần DC (mặc định: 25).

    % --- Kiểm tra và đặt giá trị mặc định cho params ---
    if ~exist('params', 'var')
        params = struct();
    end
    if ~isfield(params, 'filter_radius')
        params.filter_radius = 50; % Bán kính của bộ lọc tròn
    end
    if ~isfield(params, 'dc_suppression_radius')
        params.dc_suppression_radius = 25; % Bán kính vùng trung tâm để loại bỏ
    end

    % --- Xử lý ban đầu ---
    hologramGray = myConvGrayScale(hologram);
    [numRows, numCols] = size(hologramGray);
    fourierTransform = fftshift(fft2(hologramGray));
    spectrumMagnitude = abs(fourierTransform);
    
    % --- Tự động tìm kiếm phổ bậc +1 ---
    
    % Tọa độ tâm của phổ
    u0 = floor(numCols / 2) + 1;
    v0 = floor(numRows / 2) + 1;
    
    % Tạo một bản sao của phổ cường độ để tìm kiếm
    searchSpectrum = spectrumMagnitude;
    
    % Loại bỏ thành phần DC (bậc 0) để tránh chọn nhầm.
    % Tạo một mask tròn ở tâm và đặt giá trị trong vùng đó bằng 0.
    [U, V] = meshgrid(1:numCols, 1:numRows);
    dist_from_center = sqrt((U - u0).^2 + (V - v0).^2);
    searchSpectrum(dist_from_center <= params.dc_suppression_radius) = 0;
    
    % Chỉ tìm kiếm ở nửa trên của phổ (nơi thường chứa phổ bậc +1)
    upperHalfSpectrum = searchSpectrum(1:v0-1, :);
    
    % Tìm tọa độ của điểm có cường độ lớn nhất
    [~, maxIdx] = max(upperHalfSpectrum(:));
    [v_max, u_max] = ind2sub(size(upperHalfSpectrum), maxIdx);
    % (v_max, u_max) là tọa độ của tâm vùng ROI được chọn tự động
    
    % --- Hiển thị phổ Fourier và vùng được chọn tự động ---
    figure('Name','Phổ Fourier và Vùng chọn tự động');
    imshow(log(1 + spectrumMagnitude), []);
    hold on;
    % Vẽ vòng tròn tại vị trí đã tìm thấy để kiểm tra
    theta = 0:0.01:2*pi;
    x_circle = params.filter_radius * cos(theta) + u_max;
    y_circle = params.filter_radius * sin(theta) + v_max;
    plot(x_circle, y_circle, 'g', 'LineWidth', 2); % Vẽ vòng tròn màu xanh lá
    title(['Phổ bậc +1 được tự động chọn tại (', num2str(u_max), ', ', num2str(v_max), ')']);
    hold off;
    
    % --- Tạo bộ lọc và trích xuất phổ ---
    
    % Tạo một bộ lọc (mask) hình tròn tại vị trí (u_max, v_max)
    % Meshgrid (U, V) đã được tạo ở trên
    roi_mask = sqrt((U - u_max).^2 + (V - v_max).^2) <= params.filter_radius;
    
    % Áp dụng mask để chỉ giữ lại phổ bậc +1
    filteredContent = fourierTransform .* roi_mask;
    
    % Dịch chuyển vùng phổ đã chọn về lại tâm của ma trận
    % Tính toán độ dịch chuyển cần thiết
    v_shift = v0 - v_max;
    u_shift = u0 - u_max;
    
    % Dùng circshift để dịch chuyển. Vì phần còn lại của filteredContent là 0,
    % điều này tương đương với việc di chuyển vùng tròn về tâm.
    filteredSpectrum = circshift(filteredContent, [v_shift, u_shift]);
    
    % --- Hiển thị kết quả phổ sau khi lọc và dịch chuyển ---
    figure('Name','Phổ sau khi xử lý');
    imshow(log(1 + abs(filteredSpectrum)), []);
    title('Phổ bậc +1 sau khi lọc và dịch về tâm');
   
    % --- Tái tạo trường sóng phức và lấy pha ---
    finalPhaseComplex = ifft2(ifftshift(filteredSpectrum));
    
    % Lấy pha từ trường phức (kết quả là pha bị Wrapped trong khoảng [-pi, pi])
    wrappedPhase = angle(finalPhaseComplex);
end


function [wrappedPhase, params] = reconstruct_phase_interactively(hologram, params)
% Tái tạo pha từ hologram bằng cách lọc trong miền tần số.

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
    % trong đó delta_u là khoảng cách từ tâm đến tâm ROI theo trục u

    
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
    
    % Lấy pha từ trường phức (kết quả là pha bị Wrapped trong khoảng [-pi, pi])
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
function plot_simulation_inputs(surf_obj, phase_ref, hologram)
% Hiển thị các kết quả của quá trình mô phỏng.
    figure('Name', 'Kết quả Mô phỏng ban đầu');

    
    subplot(2, 2, 2);
    surf(surf_obj, 'EdgeColor', 'none'); title('Bề mặt pha vật thể (Gốc)');
    colormap(gca, jet); colorbar; view([45, 30]);
    
    subplot(2, 2, 3);
    surf(phase_ref,'EdgeColor', 'none'); title('Pha sóng tham chiếu');
    colormap(gca, jet); colorbar; view([45, 30]);
    
    subplot(2, 2, 4);
    imagesc(hologram); title('Ảnh Hologram mô phỏng');
    axis image; colormap(gca, gray); colorbar; axis off;
end

% -------------------------------------------------------------------------
function [unwrappedPhase, kMap] = unwrapUsingEstimate(estimatedPhase, wrappedPhase)
% Giải Wrapped pha `wrappedPhase` dựa trên pha ước lượng `estimatedPhase`.
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
    
    error_metrics.error_map_final_vs_truth = error_final_vs_truth -min(error_final_vs_truth(:));

    error_metrics.error_map_estimate_vs_truth = error_estimate_vs_truth - min(error_estimate_vs_truth(:));
    error_metrics.error_map_final_vs_estimate = error_final_vs_estimate - min(error_final_vs_estimate(:));
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
function create_advanced_error_analysis(~, phi_est, phi_gt, error_metrics)
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

% function [phi_corrected, phi_plane] = remove_plane_manual(phi)
% %REMOVE_PLANE_MANUAL Cho phép người dùng chọn điểm để nội suy và loại mặt phẳng nghiêng
% %   [phi_corrected, phi_plane] = remove_plane_manual(phi)
% %   - phi: bản đồ pha đầu vào
% %   - phi_corrected: bản đồ sau khi loại nghiêng
% %   - phi_plane: mặt phẳng đã nội suy
% 
% [N, M] = size(phi);
% [X, Y] = meshgrid(1:M, 1:N);
% 
% % --- Hiển thị ảnh ban đầu để người dùng chọn điểm ---
%     figure;
%     surf(phi,"EdgeColor","none"); colorbar;
%     title('Be mat phase  wrapped');
% 
%     figure;
%     imagesc(phi); axis image; colormap jet; colorbar;
%     title('Chọn các điểm trên mặt phẳng cần nội suy (ấn Enter khi xong)');
% % mesh(phi);
% % title('Pha Wrapped (Sau khi loại bỏ nghiêng)');
% % xlabel('x'); ylabel('y'); zlabel('Pha (rad)');
% % colormap(gca, jet); colorbar; view([45, 30]);
% 
% % --- Ginput: chọn điểm ---
% [x_pts, y_pts] = ginput();
% z_pts = interp2(phi, x_pts, y_pts);
% 
%     % --- Hiển thị lại điểm đã chọn ---
%     figure;
%     imagesc(phi); axis image; colormap turbo; hold on;
%     plot(x_pts, y_pts, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
%     for i = 1:length(x_pts)
%         text(x_pts(i)+5, y_pts(i), sprintf('%d', i), ...
%             'Color', 'w', 'FontSize', 10, 'FontWeight', 'bold');
%     end
%     title('Pha gốc với điểm đã chọn');
%     hold off;
% 
%     % --- Fit mặt phẳng ---
%     tbl = table(x_pts, y_pts, z_pts, 'VariableNames', {'x', 'y', 'z'});
%     f = fit([tbl.x, tbl.y], tbl.z, 'poly11');  % Fit mặt phẳng tuyến tính
%     phi_plane = f(X, Y);
% 
%     % --- Trừ nghiêng ---
%     phi_corrected = phi - phi_plane;
% 
%     % --- Hiển thị kết quả ---
%     figure;
%     subplot(1,3,1);
%     imagesc(phi); axis image; colormap turbo;
%     title('Pha gốc');
% 
%     subplot(1,3,2);
%     imagesc(phi_plane); axis image; colormap turbo;
%     title('Mặt phẳng đã fit');
% 
%     subplot(1,3,3);
%     imagesc(phi_corrected); axis image; colormap turbo;
%     title('Pha đã loại nghiêng');
% end
function [phi_corrected, phi_plane] = remove_plane_manual(phi)
%REMOVE_PLANE_MANUAL Cho phép người dùng chọn điểm hoặc vẽ HCN để nội suy và loại mặt phẳng nghiêng
%   [phi_corrected, phi_plane] = remove_plane_manual(phi)
%   - phi: bản đồ pha đầu vào
%   - phi_corrected: bản đồ sau khi loại nghiêng
%   - phi_plane: mặt phẳng đã nội suy

[N, M] = size(phi);
[X, Y] = meshgrid(1:M, 1:N);

% --- Hiển thị ảnh ban đầu để người dùng chọn phương thức ---
figure;
surf(phi,"EdgeColor","none"); colormap jet; colorbar;
title('Bản đồ pha gốc');

figure;
imagesc(phi); axis image; colormap jet; colorbar;
title('Bản đồ pha gốc');

% --- Hộp thoại lựa chọn phương thức ---
choice = questdlg('Chọn phương thức để xác định mặt phẳng:', ...
    'Lựa chọn nội suy', ...
    'Chọn điểm','Vẽ HCN','Chọn điểm');

% --- Lấy điểm dựa trên lựa chọn của người dùng ---
switch choice
    case 'Chọn điểm'
        % --- Chức năng GINPUT nguyên bản: chọn điểm thủ công ---
        title('Chọn các điểm trên mặt phẳng cần nội suy (ấn Enter khi xong)');
        [x_pts, y_pts] = ginput();
        if isempty(x_pts)
            disp('Không có điểm nào được chọn. Đang hủy bỏ...');
            phi_corrected = phi;
            phi_plane = zeros(N, M);
            return;
        end
    case 'Vẽ HCN'
        % --- Chức năng GETRECT mới: vẽ hình chữ nhật ---
        title('Vẽ một hình chữ nhật trên vùng cần nội suy');
        rect = getrect; % [xmin ymin width height]
        
        % Lấy tọa độ 4 góc từ hình chữ nhật
        xmin = rect(1);
        ymin = rect(2);
        width = rect(3);
        height = rect(4);
        
        x_pts = [xmin; xmin + width; xmin + width; xmin];
        y_pts = [ymin; ymin; ymin + height; ymin + height];
        
        if width == 0 || height == 0
            disp('Hình chữ nhật không hợp lệ. Đang hủy bỏ...');
            phi_corrected = phi;
            phi_plane = zeros(N, M);
            return;
        end
    case ''
        % Người dùng đã đóng hộp thoại
        disp('Không có lựa chọn nào được thực hiện. Đang hủy bỏ...');
        phi_corrected = phi;
        phi_plane = zeros(N, M);
        return;
end

% --- Lấy giá trị Z tại các điểm đã chọn ---
z_pts = interp2(phi, x_pts, y_pts);

% --- Hiển thị lại ảnh với các điểm đã chọn ---
figure;
imagesc(phi); axis image; colormap jet; hold on;
plot(x_pts, y_pts, 'rx', 'MarkerSize', 12, 'LineWidth', 2);
if strcmp(choice, 'Vẽ HCN')
    % Vẽ lại hình chữ nhật để xác nhận
    rect_x = [x_pts' x_pts(1)];
    rect_y = [y_pts' y_pts(1)];
    plot(rect_x, rect_y, 'r-', 'LineWidth', 2);
end
for i = 1:length(x_pts)
    text(x_pts(i) + 5, y_pts(i), sprintf('%d', i), ...
        'Color', 'w', 'FontSize', 10, 'FontWeight', 'bold');
end
title('Pha gốc với các điểm nội suy đã chọn');
hold off;

% --- Fit mặt phẳng ---
% Sử dụng các điểm (x, y, z) để khớp với một mặt phẳng tuyến tính
tbl = table(x_pts, y_pts, z_pts, 'VariableNames', {'x', 'y', 'z'});
fit_model = fit([tbl.x, tbl.y], tbl.z, 'poly11');  % poly11: f(x,y) = p00 + p10*x + p01*y

% Tạo mặt phẳng đã khớp trên toàn bộ lưới tọa độ
phi_plane = fit_model(X, Y);

% --- Trừ mặt phẳng (nghiêng) khỏi pha gốc ---
phi_corrected = phi - phi_plane;

% --- Hiển thị kết quả ---
figure;
sgtitle('Kết quả loại bỏ mặt phẳng nghiêng');

subplot(1,3,1);
imagesc(phi); axis image; colormap turbo; colorbar;
title('Pha gốc');

subplot(1,3,2);
imagesc(phi_plane); axis image; colormap turbo; colorbar;
title('Mặt phẳng đã fit');

subplot(1,3,3);
imagesc(phi_corrected); axis image; colormap turbo; colorbar;
title('Pha đã loại nghiêng');

end

function [phi_corrected, phi_plane] = remove_plane_manual2(phi)
%REMOVE_PLANE_MANUAL Cho phép người dùng chọn điểm hoặc vẽ HCN để nội suy và loại mặt phẳng nghiêng
%   [phi_corrected, phi_plane] = remove_plane_manual(phi)
%   - phi: bản đồ pha đầu vào
%   - phi_corrected: bản đồ sau khi loại nghiêng
%   - phi_plane: mặt phẳng đã nội suy

[N, M] = size(phi);
[X, Y] = meshgrid(1:M, 1:N);

% --- Hiển thị ảnh ban đầu để người dùng chọn phương thức ---
figure;
surf(phi,"EdgeColor","none"); colormap jet; colorbar;
title('Bản đồ pha gốc');

figure;
imagesc(phi); axis image; colormap jet; colorbar;
title('Bản đồ pha gốc');

% --- Hộp thoại lựa chọn phương thức ---
choice = questdlg('Chọn phương thức để xác định mặt phẳng:', ...
    'Lựa chọn nội suy', ...
    'Chọn điểm','Vẽ HCN','Chọn điểm');

% --- Lấy điểm dựa trên lựa chọn của người dùng ---
switch choice
    case 'Chọn điểm'
        % --- Chức năng GINPUT nguyên bản: chọn điểm thủ công ---
        title('Chọn các điểm trên mặt phẳng cần nội suy (ấn Enter khi xong)');
        [x_pts, y_pts] = ginput();
        if isempty(x_pts)
            disp('Không có điểm nào được chọn. Đang hủy bỏ...');
            phi_corrected = phi;
            phi_plane = zeros(N, M);
            return;
        end
    case 'Vẽ HCN'
        % --- Chức năng GETRECT mới: vẽ hình chữ nhật ---
        title('Vẽ một hình chữ nhật trên vùng cần nội suy');
        rect = getrect; % [xmin ymin width height]
        
        % Lấy tọa độ 4 góc từ hình chữ nhật
        xmin = rect(1);
        ymin = rect(2);
        width = rect(3);
        height = rect(4);
        
        x_pts = [xmin; xmin + width; xmin + width; xmin];
        y_pts = [ymin; ymin; ymin + height; ymin + height];
        
        if width == 0 || height == 0
            disp('Hình chữ nhật không hợp lệ. Đang hủy bỏ...');
            phi_corrected = phi;
            phi_plane = zeros(N, M);
            return;
        end
    case ''
        % Người dùng đã đóng hộp thoại
        disp('Không có lựa chọn nào được thực hiện. Đang hủy bỏ...');
        phi_corrected = phi;
        phi_plane = zeros(N, M);
        return;
end

% --- Lấy giá trị Z tại các điểm đã chọn ---
z_pts = interp2(phi, x_pts, y_pts);

% --- Hiển thị lại ảnh với các điểm đã chọn ---
figure;
imagesc(phi); axis image; colormap jet; hold on;
plot(x_pts, y_pts, 'rx', 'MarkerSize', 12, 'LineWidth', 2);
if strcmp(choice, 'Vẽ HCN')
    % Vẽ lại hình chữ nhật để xác nhận
    rect_x = [x_pts' x_pts(1)];
    rect_y = [y_pts' y_pts(1)];
    plot(rect_x, rect_y, 'r-', 'LineWidth', 2);
end
for i = 1:length(x_pts)
    text(x_pts(i) + 5, y_pts(i), sprintf('%d', i), ...
        'Color', 'w', 'FontSize', 10, 'FontWeight', 'bold');
end
title('Pha gốc với các điểm nội suy đã chọn');
hold off;

% --- Fit mặt phẳng ---
% Sử dụng các điểm (x, y, z) để khớp với một mặt phẳng tuyến tính
tbl = table(x_pts, y_pts, z_pts, 'VariableNames', {'x', 'y', 'z'});
fit_model = fit([tbl.x, tbl.y], tbl.z, 'poly11');  % poly11: f(x,y) = p00 + p10*x + p01*y

% Tạo mặt phẳng đã khớp trên toàn bộ lưới tọa độ
phi_plane = fit_model(X, Y);

% --- Trừ mặt phẳng (nghiêng) khỏi pha gốc ---
phi_corrected = phi - wrapToPi(phi_plane);

% --- Hiển thị kết quả ---
figure;
sgtitle('Kết quả loại bỏ mặt phẳng nghiêng');

subplot(1,3,1);
imagesc(phi); axis image; colormap turbo; colorbar;
title('Pha gốc');

subplot(1,3,2);
imagesc(phi_plane); axis image; colormap turbo; colorbar;
title('Mặt phẳng đã fit');

subplot(1,3,3);
imagesc(phi_corrected); axis image; colormap turbo; colorbar;
title('Pha đã loại nghiêng');

end

function [phi_corrected, plane_est] = remove_tilt_from_wrapped2(phi_wrapped)
% REMOVE_TILT_FROM_WRAPPED - Loại bỏ mặt phẳng nghiêng từ ảnh pha wrapped
% Cho phép chọn vùng bằng hình chữ nhật hoặc chọn nhiều điểm tự do
%
% Inputs:
%   phi_wrapped - ảnh pha đã wrap [-pi, pi]
%
% Outputs:
%   phi_corrected - ảnh pha đã loại bỏ mặt phẳng nghiêng (wrapped lại)
%   plane_est - mặt phẳng nghiêng ước lượng (a*x + b*y + c)

% Kiểm tra input
if nargin < 1
    error('Cần ít nhất 1 input argument');
end

if ~ismatrix(phi_wrapped) || ~isnumeric(phi_wrapped)
    error('phi_wrapped phải là ma trận số');
end

[rows, cols] = size(phi_wrapped);
[X, Y] = meshgrid(1:cols, 1:rows);

% --- Chọn phương thức ---
mode = questdlg('Chọn phương pháp chọn vùng?', ...
                'Chọn vùng nghiêng', ...
                'Hình chữ nhật', 'Chọn điểm', 'Hủy', 'Hình chữ nhật');

% Xử lý trường hợp user hủy
if isempty(mode) || strcmp(mode, 'Hủy')
    phi_corrected = phi_wrapped;
    plane_est = zeros(size(phi_wrapped));
    return;
end

try
    switch mode
        case 'Hình chữ nhật'
            % --- Vẽ HCN chọn vùng ---
            figure;
            surf(phi_wrapped,"EdgeColor","none");
            colormap jet; colorbar; 
            title("Ảnh ban đầu wrapped phase");
            fig1 = figure;
            imagesc(phi_wrapped); 
            axis image; 
            colormap gray; 
            colorbar;
            title('Vẽ hình chữ nhật để chọn vùng nghiêng (double-click khi xong)');
            
            % Sử dụng drawrectangle với error handling
            h = drawrectangle('Color','g', 'LineWidth', 1);
            wait(h);
            
            rect_pos = round(h.Position); % [x, y, w, h]
            
            % Kiểm tra tính hợp lệ của rectangle
            if rect_pos(3) < 3 || rect_pos(4) < 3
                error('Vùng chọn quá nhỏ. Vui lòng chọn vùng lớn hơn.');
            end
            
            x1 = max(1, rect_pos(1));
            y1 = max(1, rect_pos(2));
            x2 = min(cols, x1 + rect_pos(3) - 1);
            y2 = min(rows, y1 + rect_pos(4) - 1);
            
            % Lấy 4 điểm góc của hình chữ nhật
            corner_x = [x1, x2, x1, x2];  % góc trái trên, phải trên, trái dưới, phải dưới
            corner_y = [y1, y1, y2, y2];
            
            % Lấy giá trị phase tại 4 điểm góc
            corner_phases = zeros(4, 1);
            for i = 1:4
                corner_phases(i) = phi_wrapped(corner_y(i), corner_x(i));
            end
            
            % Hiển thị các điểm góc trên ảnh
            hold on;
            plot(corner_x, corner_y, 'ro', 'MarkerSize', 8, 'LineWidth', 2, 'MarkerFaceColor', 'y');
            for i = 1:4
                text(corner_x(i)+5, corner_y(i), sprintf('%.3f', corner_phases(i)), ...
                     'Color', 'w', 'FontSize', 8, 'FontWeight', 'bold', 'BackgroundColor', 'k');
            end
            hold off;
            
            % Fit mặt phẳng từ 4 điểm góc
            A = [corner_x(:), corner_y(:), ones(4,1)];
            
            % Kiểm tra điều kiện của ma trận A
            if rank(A) < 3
                warning('Các điểm góc không đủ để xác định mặt phẳng duy nhất. Sử dụng least squares.');
                coeffs = A \ corner_phases(:);
            else
                coeffs = A \ corner_phases(:);
            end
           
            
            pause(3); % Cho user xem các điểm góc
            close(fig1);
            
        case 'Chọn điểm'
            % --- Chọn nhiều điểm trên ảnh ---
            fig2 = figure;
            imagesc(phi_wrapped); 
            axis image; 
            colormap jet; 
            colorbar;
            title('Click các điểm phẳng (ít nhất 3 điểm, ấn Enter khi xong)');
            %
%             % Sử dụng drawrectangle với error handling
%             h = drawrectangle('Color','g', 'LineWidth', 1);
%             wait(h);
%             
%             rect_pos = round(h.Position); % [x, y, w, h]
%             
%             % Kiểm tra tính hợp lệ của rectangle
%             if rect_pos(3) < 3 || rect_pos(4) < 3
%                 error('Vùng chọn quá nhỏ. Vui lòng chọn vùng lớn hơn.');
%             end
%             
%             x1 = max(1, rect_pos(1));
%             y1 = max(1, rect_pos(2));
%             x2 = min(cols, x1 + rect_pos(3) - 1);
%             y2 = min(rows, y1 + rect_pos(4) - 1);
%             
%             % Lấy 4 điểm góc của hình chữ nhật
%             x_pts = [x1, x2, x1, x2];  % góc trái trên, phải trên, trái dưới, phải dưới
%             y_pts = [y1, y1, y2, y2];
            %
            [x_pts, y_pts] = ginput(); % Chọn nhiều điểm
            
            % Kiểm tra số điểm
            if length(x_pts) < 3
                error('Cần ít nhất 3 điểm để fit mặt phẳng');
            end
            
            % Đảm bảo điểm nằm trong ảnh
            x_pts = max(1, min(cols, x_pts));
            y_pts = max(1, min(rows, y_pts));
            
            z_pts = interp2(X, Y, phi_wrapped, x_pts, y_pts, 'linear');
            
            % Loại bỏ điểm NaN
            valid_idx = ~isnan(z_pts);
            x_pts = x_pts(valid_idx);
            y_pts = y_pts(valid_idx);
            z_pts = z_pts(valid_idx);
            
            if length(x_pts) < 3
                error('Không đủ điểm hợp lệ để fit mặt phẳng');
            end
            
            % Unwrap phase values để tránh discontinuity
            z_pts_unwrapped = unwrap(z_pts);
            
            % Fit mặt phẳng từ điểm
            A = [x_pts(:), y_pts(:), ones(numel(x_pts),1)];
            coeffs = A \ z_pts_unwrapped(:);
            
            % Hiển thị lại các điểm
            hold on;
            plot(x_pts, y_pts, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
            for i = 1:length(x_pts)
                text(x_pts(i)+5, y_pts(i), sprintf('%d', i), ...
                     'Color', 'w', 'FontSize', 10, 'FontWeight', 'bold');
            end
            hold off;
            
            pause(2); % Cho user xem kết quả
            close(fig2);
            
        otherwise
            error('Phương thức không hợp lệ');
    end
    
    % --- Tính mặt phẳng trên toàn ảnh ---
    a = coeffs(1); 
    b = coeffs(2); 
    c = coeffs(3);
    plane_est = a*X + b*Y + c;
    
    % --- Trừ mặt phẳng nghiêng ---
    phi_corrected = wrapToPi(phi_wrapped - plane_est);
    
    % --- Hiển thị kết quả ---
    figure('Name', 'Kết quả loại bỏ nghiêng', 'NumberTitle', 'off');
    
    subplot(2,2,1); 
    imagesc(phi_wrapped); 
    axis image; 
    title('Pha Wrapped gốc'); 
    colormap jet; 
    colorbar;
    
    subplot(2,2,2); 
    imagesc(phi_corrected); 
    axis image; 
    title('Sau khi loại nghiêng'); 
    colormap jet; 
    colorbar;
    
    subplot(2,2,3); 
    imagesc(plane_est); 
    axis image; 
    title('Mặt phẳng nghiêng ước lượng'); 
    colormap jet; 
    colorbar;
    
    subplot(2,2,4); 
    imagesc(phi_wrapped - phi_corrected); 
    axis image; 
    title('Độ lệch đã loại bỏ'); 
    colormap jet; 
    colorbar;
    
    % Figure 3D
    figure('Name', 'Hiển thị 3D', 'NumberTitle', 'off');
    
    subplot(1,2,1); 
    mesh(X, Y, phi_wrapped); 
    title('Wrapped phase (gốc)');
    xlabel('x'); ylabel('y'); zlabel('Pha'); 
    colormap jet; colorbar; 
    view(45,30);
    
    subplot(1,2,2); 
    mesh(X, Y, phi_corrected); 
    title('Sau khi loại nghiêng');
    xlabel('x'); ylabel('y'); zlabel('Pha'); 
    colormap jet; colorbar; 
    view(45,30);
    
    % In thông tin
    fprintf('Tham số mặt phẳng: a=%.6f, b=%.6f, c=%.6f\n', a, b, c);
    fprintf('Độ nghiêng X: %.6f rad/pixel\n', a);
    fprintf('Độ nghiêng Y: %.6f rad/pixel\n', b);
    
catch ME
    % Xử lý lỗi
    fprintf('Lỗi: %s\n', ME.message);
    phi_corrected = phi_wrapped;
    plane_est = zeros(size(phi_wrapped));
    
    % Đóng figure nếu có
    if exist('fig1', 'var') && isvalid(fig1)
        close(fig1);
    end
    if exist('fig2', 'var') && isvalid(fig2)
        close(fig2);
    end
end

end

function [corrected_unwrapped_phase, num_iterations, convergence_history] = correct_sparse_artifacts_iterative(unwrapped_phase_input, varargin)
% Hàm cải tiến: Xử lý các điểm nhiễu sparse với thuật toán lặp và ràng buộc biên
% Dựa trên phương pháp lọc trung vị để xác định và hiệu chỉnh các điểm lỗi.
% Lặp đến khi hội tụ (không còn thay đổi k hoặc thay đổi < epsilon)
%
% Inputs:
%   unwrapped_phase_input - Ma trận pha unwrapped đầu vào
%   varargin - Các tham số tùy chọn:
%       'FilterSize' - Kích thước bộ lọc [default: [15 15]]
%       'Epsilon' - Ngưỡng hội tụ [default: 1e-6]
%       'MaxIterations' - Số lần lặp tối đa [default: 50]
%       'Verbose' - Hiển thị thông tin debug [default: false]
%       'BoundaryCondition' - Điều kiện biên ['zero'|'symmetric'|'replicate'|'circular'] [default: 'symmetric']
%       'BoundaryWidth' - Độ rộng vùng biên không được hiệu chỉnh [default: 0]
%       'PreserveBoundary' - Giữ nguyên giá trị biên [default: true]
%       'MaxDeltaK' - Giới hạn tối đa cho |delta_k| [default: 10]
%       'MaskInvalid' - Mask cho các pixel không hợp lệ [default: []]
%
% Outputs:
%   corrected_unwrapped_phase - Pha đã được hiệu chỉnh
%   num_iterations - Số lần lặp thực tế
%   convergence_history - Lịch sử hội tụ (RMS của delta_k)

    % Xử lý tham số đầu vào
    p = inputParser;
    addParameter(p, 'FilterSize', [15 15], @(x) isnumeric(x) && length(x) == 2);
    addParameter(p, 'Epsilon', 1e-6, @(x) isnumeric(x) && x > 0);
    addParameter(p, 'MaxIterations', 100, @(x) isnumeric(x) && x > 0);
    addParameter(p, 'Verbose', false, @islogical);
    addParameter(p, 'BoundaryCondition', 'symmetric', @(x) ischar(x) && ismember(x, {'zero', 'symmetric', 'replicate', 'circular'}));
    addParameter(p, 'BoundaryWidth', 0, @(x) isnumeric(x) && x >= 0);
    addParameter(p, 'PreserveBoundary', true, @islogical);
    addParameter(p, 'MaxDeltaK', 10, @(x) isnumeric(x) && x > 0);
    addParameter(p, 'MaskInvalid', [], @(x) isempty(x) || islogical(x));
    parse(p, varargin{:});
    
    filter_size = p.Results.FilterSize;
    epsilon = p.Results.Epsilon;
    max_iterations = p.Results.MaxIterations;
    verbose = p.Results.Verbose;
    boundary_condition = p.Results.BoundaryCondition;
    boundary_width = p.Results.BoundaryWidth;
    preserve_boundary = p.Results.PreserveBoundary;
    max_delta_k = p.Results.MaxDeltaK;
    mask_invalid = p.Results.MaskInvalid;
    
    % Khởi tạo
    [rows, cols] = size(unwrapped_phase_input);
    current_phase = unwrapped_phase_input;
    original_phase = unwrapped_phase_input; % Lưu pha gốc để tham chiếu biên
    convergence_history = [];
    num_iterations = 0;
    previous_delta_k = [];
    
    % Tạo mask cho vùng biên nếu cần
    if preserve_boundary && boundary_width > 0
        boundary_mask = create_boundary_mask(rows, cols, boundary_width);
    else
        boundary_mask = false(rows, cols);
    end

% Hàm hỗ trợ: Tạo mask cho vùng biên
function boundary_mask = create_boundary_mask(rows, cols, width)
    boundary_mask = false(rows, cols);
    if width > 0
        boundary_mask(1:width, :) = true;           % Biên trên
        boundary_mask(end-width+1:end, :) = true;   % Biên dưới
        boundary_mask(:, 1:width) = true;           % Biên trái
        boundary_mask(:, end-width+1:end) = true;   % Biên phải
    end
end

% Hàm hỗ trợ: Áp dụng điều kiện biên
function phase_with_boundary = apply_boundary_condition(phase, condition, filter_size)
    [rows, cols] = size(phase);
    pad_rows = floor(filter_size(1)/2);
    pad_cols = floor(filter_size(2)/2);
    
    switch lower(condition)
        case 'zero'
            phase_with_boundary = padarray(phase, [pad_rows, pad_cols], 0, 'both');
        case 'symmetric'
            phase_with_boundary = padarray(phase, [pad_rows, pad_cols], 'symmetric', 'both');
        case 'replicate'
            phase_with_boundary = padarray(phase, [pad_rows, pad_cols], 'replicate', 'both');
        case 'circular'
            phase_with_boundary = padarray(phase, [pad_rows, pad_cols], 'circular', 'both');
        otherwise
            phase_with_boundary = padarray(phase, [pad_rows, pad_cols], 'symmetric', 'both');
    end
end

% Hàm hỗ trợ: Ràng buộc tính liên tục không gian
function delta_k_constrained = apply_spatial_continuity_constraint(delta_k, current_phase)
    % Kiểm tra gradient địa phương để tránh các thay đổi đột ngột
    [rows, cols] = size(delta_k);
    delta_k_constrained = delta_k;
    
    % Tính gradient của pha hiện tại
    [grad_x, grad_y] = gradient(current_phase);
    grad_magnitude = sqrt(grad_x.^2 + grad_y.^2);
    
    % Định nghĩa ngưỡng gradient (vùng có gradient cao được phép thay đổi nhiều hơn)
    grad_threshold = prctile(grad_magnitude(:), 75); % 75th percentile
    
    % Áp dụng ràng buộc dựa trên gradient
    for i = 2:rows-1
        for j = 2:cols-1
            if abs(delta_k(i,j)) > 1 && grad_magnitude(i,j) < grad_threshold
                % Nếu thay đổi lớn nhưng gradient thấp, hạn chế thay đổi
                neighbors = delta_k(i-1:i+1, j-1:j+1);
                median_neighbor = median(neighbors(:));
                
                % Chỉ cho phép thay đổi không quá 1 bước so với median của lân cận
                if abs(delta_k(i,j) - median_neighbor) > 1
                    delta_k_constrained(i,j) = median_neighbor + sign(delta_k(i,j) - median_neighbor);
                end
            end
        end
    end
end
    
    % Xử lý mask invalid
    if isempty(mask_invalid)
        mask_invalid = false(rows, cols);
    else
        if ~isequal(size(mask_invalid), [rows, cols])
            error('MaskInvalid phải có cùng kích thước với unwrapped_phase_input');
        end
    end
    
    % Mask tổng hợp (vùng không được hiệu chỉnh)
    protection_mask = boundary_mask | mask_invalid;
    
    if verbose
        fprintf('Bắt đầu quá trình hiệu chỉnh lặp với ràng buộc biên...\n');
        fprintf('Image size: %dx%d\n', rows, cols);
        fprintf('Filter size: [%d %d], Epsilon: %.2e, Max iterations: %d\n', ...
                filter_size(1), filter_size(2), epsilon, max_iterations);
        fprintf('Boundary condition: %s, Boundary width: %d\n', boundary_condition, boundary_width);
        fprintf('Protected pixels: %d (%.2f%%)\n', sum(protection_mask(:)), 100*sum(protection_mask(:))/(rows*cols));
    end
    
    % Vòng lặp chính
    for iter = 1:max_iterations
        % Bước 1: Xử lý điều kiện biên trước khi lọc
        phase_with_boundary = apply_boundary_condition(current_phase, boundary_condition, filter_size);
        
        % Bước 2: Áp dụng bộ lọc trung vị với xử lý biên
        filtered_phase = medfilt2(phase_with_boundary, filter_size, 'symmetric');
        
        % Cắt về kích thước ban đầu nếu cần
        if ~isequal(size(filtered_phase), [rows, cols])
            filtered_phase = filtered_phase(1:rows, 1:cols);
        end
        
        % Bước 3: Tính toán sự khác biệt về "thứ tự vân" 
        % delta_k = Round[(Phi_filtered - Phi_current) / 2π]
        delta_k = round((filtered_phase - current_phase) / (2*pi));
        
        % Bước 4: Áp dụng các ràng buộc
        % Giới hạn |delta_k|
        delta_k = sign(delta_k) .* min(abs(delta_k), max_delta_k);
        
        % Bảo vệ vùng biên và các pixel không hợp lệ
        delta_k(protection_mask) = 0;
        
        % Bước 5: Kiểm tra tính liên tục không gian (spatial continuity constraint)
        delta_k = apply_spatial_continuity_constraint(delta_k, current_phase);
        
        % Tính toán metric hội tụ (RMS của delta_k chỉ trên vùng được phép thay đổi)
        active_pixels = ~protection_mask;
        if sum(active_pixels(:)) > 0
            rms_delta_k = sqrt(mean((delta_k(active_pixels)).^2));
        else
            rms_delta_k = 0;
        end
        
        convergence_history(end+1) = rms_delta_k;
        num_iterations = iter;
        
        if verbose
            num_corrections = sum(delta_k(:) ~= 0);
            fprintf('Iteration %d: RMS(delta_k) = %.6f, Corrections: %d, Unique values: %d\n', ...
                    iter, rms_delta_k, num_corrections, length(unique(delta_k(:))));
        end
        
        % Kiểm tra điều kiện hội tụ
        if iter > 1
            % Kiểm tra xem delta_k có thay đổi không
            if isequal(delta_k, previous_delta_k)
                if verbose
                    fprintf('Hội tụ đạt được: delta_k không thay đổi (iteration %d)\n', iter);
                end
                break;
            end
            
            % Kiểm tra xem thay đổi có nhỏ hơn epsilon không
            if rms_delta_k < epsilon
                if verbose
                    fprintf('Hội tụ đạt được: RMS(delta_k) < epsilon (iteration %d)\n', iter);
                end
                break;
            end
            
            % Kiểm tra thay đổi tương đối giữa các lần lặp
            relative_change = abs(convergence_history(end) - convergence_history(end-1)) / ...
                             (convergence_history(end-1) + eps);
            if relative_change < epsilon
                if verbose
                    fprintf('Hội tụ đạt được: Thay đổi tương đối < epsilon (iteration %d)\n', iter);
                end
                break;
            end
        end
        
        % Bước 3: Hiệu chỉnh pha với ràng buộc biên
        % Phi_corrected = Phi_current + delta_k * 2π
        current_phase = current_phase + delta_k * (2*pi);
        
        % Khôi phục giá trị biên gốc nếu cần
        if preserve_boundary
            current_phase(protection_mask) = original_phase(protection_mask);
        end
        
        % Lưu delta_k hiện tại để so sánh ở lần lặp tiếp theo
        previous_delta_k = delta_k;
        
        % Kiểm tra nếu đạt số lần lặp tối đa
        if iter == max_iterations
            if verbose
                fprintf('Cảnh báo: Đạt số lần lặp tối đa (%d) mà chưa hội tụ hoàn toàn\n', max_iterations);
            end
        end
    end
    
    corrected_unwrapped_phase = current_phase;
    
    if verbose
        fprintf('Hoàn thành sau %d lần lặp\n', num_iterations);
        fprintf('RMS cuối cùng của delta_k: %.6f\n', convergence_history(end));
    end
end

% Hàm hỗ trợ: Phân tích kết quả hội tụ
function plot_convergence_analysis(convergence_history)
    figure;
    subplot(2,1,1);
    plot(1:length(convergence_history), convergence_history, 'b-o', 'LineWidth', 2);
    xlabel('Số lần lặp');
    ylabel('RMS(delta_k)');
    title('Quá trình hội tụ');
    grid on;
    
    subplot(2,1,2);
    if length(convergence_history) > 1
        semilogy(1:length(convergence_history), convergence_history, 'r-s', 'LineWidth', 2);
        xlabel('Số lần lặp');
        ylabel('RMS(delta_k) (log scale)');
        title('Quá trình hội tụ (thang logarit)');
        grid on;
    end
end

%% thêm ngày 29-6-25
function W = reconstructZernikeAdvanced(indices, coefficients, gridSize)
%reconstructZernikeAdvanced Tái tạo bề mặt Zernike từ chỉ số và hệ số.
%
%   Hàm này hoạt động ở 2 chế độ:
%   1. CHẾ ĐỘ (n,m): Nếu 'indices' là ma trận Kx2, mỗi hàng là một cặp [n,m].
%   2. CHẾ ĐỘ j: Nếu 'indices' là vector Kx1, mỗi phần tử là một số thứ tự j.
%
%   Input:
%       indices (matrix hoặc vector): Ma trận Kx2 các cặp [n,m] HOẶC vector Kx1
%                                     các số thứ tự j.
%       coefficients (vector): Vector Kx1 hoặc 1xK chứa các hệ số tương ứng.
%       gridSize (integer, optional): Kích thước lưới. Mặc định là 256.
%
%   Output:
%       W (matrix): Bề mặt wavefront cuối cùng.

    % --- 0. Xử lý đầu vào tùy chọn ---
    if nargin < 3, gridSize = 256; end

    % --- 1. Kiểm tra tính hợp lệ của đầu vào ---
    if size(indices, 1) ~= numel(coefficients) && size(indices, 2) ~= numel(coefficients)
         error('Số lượng chỉ số phải bằng số lượng hệ số.');
    end
    
    % --- 2. Tạo lưới tọa độ ---
    x = linspace(-1, 1, gridSize);
    y = linspace(-1, 1, gridSize);
    [X, Y] = meshgrid(x, y);
    [t, r] = cart2pol(X, Y);
    
    % --- 3. Xử lý và tính toán ---
    W = zeros(gridSize, gridSize);
    num_terms = numel(coefficients);
    
    % --- PHÁT HIỆN CHẾ ĐỘ DỰA TRÊN KÍCH THƯỚC CỦA 'indices' ---
    
    if size(indices, 2) == 2 && size(indices,1) == num_terms % CHẾ ĐỘ (n,m)
        fprintf('--- Chế độ (n,m) được kích hoạt ---\n');
        nm_pairs = indices;
        for k = 1:num_terms
            n = nm_pairs(k, 1);
            m = nm_pairs(k, 2);
            C = coefficients(k);
            if C ~= 0
                fprintf('Đang thêm đa thức (n=%d, m=%d) với hệ số %.2f\n', n, m, C);
                W = W + C * zernike(r, t, n, m);
            end
        end
        
    elseif isvector(indices) % CHẾ ĐỘ SỐ THỨ TỰ j
        fprintf('--- Chế độ số thứ tự (j) được kích hoạt ---\n');
        j_indices = indices(:); % Đảm bảo là vector cột
        for k = 1:num_terms
            j = j_indices(k);
            C = coefficients(k);
            if C ~= 0
                % Chuyển đổi j -> (n,m) bằng hàm phụ
                [n, m] = map_j_to_nm(j);
                fprintf('Đang thêm đa thức j=%d (n=%d, m=%d) với hệ số %.2f\n', j, n, m, C);
                W = W + C * zernike(r, t, n, m);
            end
        end
        
    else
        error("Định dạng của 'indices' không hợp lệ. Phải là ma trận Kx2 hoặc vector Kx1.");
    end

    % --- 4. Mask ---
%     W(r > 1) = NaN; 
    fprintf('Hoàn thành tái tạo bề mặt.\n');
end
function [n, m] = map_j_to_nm(j)
%map_j_to_nm Chuyển đổi chỉ số Zernike tuần tự j sang cặp (n,m).
%   Sử dụng thứ tự chuẩn ANSI.

    if j <= 0 || floor(j) ~= j
        error('Chỉ số j phải là một số nguyên dương.');
    end
    
    count = 1;
    n_current = 0;
    while true
        for m_current = -n_current:2:n_current
            if count == j
                n = n_current;
                m = m_current;
                return; % Trả về kết quả và thoát hàm
            end
            count = count + 1;
        end
        n_current = n_current + 1;
    end
end
