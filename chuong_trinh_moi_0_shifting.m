%%
% 
% 
% =========================================================================
% SCRIPT CHÍNH: MÔ PHỎNG, TÁI TẠO VÀ PHÂN TÍCH PHA TỪ OFF-AXIS HOLOGRAM
% =========================================================================
% chạy oke 12/8/25

%% 1. KHỞI TẠO
clc;
clear;
close all;
fprintf('Bắt đầu quy trình mô phỏng và tái tạo...\n');
% Các mức nhiễu cần kiểm tra
noise_levels = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30];
rms_errors = zeros(size(noise_levels));


    % --- KHỞI TẠO lại các tham số (copy phần đầu script của bạn, hoặc gọi từ hàm setup) ---


%% 2. MÔ PHỎNG HOLOGRAM
fprintf('--> Bước 1: Mô phỏng Hologram...\n');
% --- Thiết lập thông số ---
M = 512; % Kích thước ảnh (chiều cao)
N = 512; % Kích thước ảnh (chiều rộng)

fx = 40 / N; % Tần số sóng mang
fy = -60 / M;

params = struct();
params = set_default_params(params);
auto_fft = 0;
% Bề mặt ground truth
% --- Tạo đối tượng pha gốc (Ground Truth) ---
% Lựa chọn: 'peaks', 'sin', 'gaussian', 'zernike', 'step',
%           'sharp_peak', 'multi_sharp_peak', 'sinc_spike',
%           'microsphere_array','test','test_ls','residual'
params.groundTruth= 'high_noise';
noise_level = 0.15;
% Bộ lọc Fourier
params.filter_type = 'rectangle';         % 'circle' | 'rectangle'
params.filter_radius = 40;             % (px) bán kính bộ lọc tròn
params.filter_width = 150;             % (px) chiều rộng HCN
params.filter_height = 120;            % (px) chiều cao HCN

% Ức chế DC (zero-order)
params.dc_suppression_radius = 25;     % (px) bán kính loại bỏ trung tâm

% Tham số ảnh
params.lambda = 632.8e-9;              % bước sóng ánh sáng (m)
params.pixel_size = 3.45e-6;           % kích thước điểm ảnh (m)

% Pha
params.unwrap_method = 'step';  % hoặc 'quality_guided', 'hybrid'
params.phase_smoothing = true;
params.smoothing_sigma = 2;            % độ mượt hậu xử lý (Gaussian)

% Debug/hiển thị
params.show_figures = true;
params.verbose = true;



% --- Tạo đối tượng pha gốc (Ground Truth) ---
% Lựa chọn: 'peaks', 'sin', 'gaussian', 'zernike', 'step',
%           'sharp_peak', 'multi_sharp_peak', 'sinc_spike',
%           'microsphere_array','test'
phase_type = params.groundTruth; 

% --- Tạo lưới tọa độ chung (có thể dùng cho nhiều trường hợp) ---
% Một số trường hợp cần lưới tọa độ từ -1 đến 1, số khác cần từ 1 đến N/M.
% Chúng ta sẽ tạo lưới tọa độ bên trong từng case khi cần để đảm bảo tính đúng đắn.

fprintf('Đang tạo đối tượng pha loại: %s với kích thước M=%d, N=%d\n', phase_type, M, N);

switch lower(phase_type)
    case 'peaks'
        noise_gt = 0.3;
        [X, Y] = meshgrid(linspace(-1, 1, N), linspace(-1, 1, M));
        phi_ground_truth = 2 * peaks(3*X, 3*Y) ;

        raw_noise = 2*randn(size(phi_ground_truth));
        % smooth_noise = imgaussfilt(raw_noise, 5); % kernel 3x3
        [x, sigma] = meshgrid(linspace(0, 1, N), linspace(0, pi/5, N));

        % Define the constant 'a'
        a = N * pi / 2;

        % Define the zero-mean Gaussian noise component
        % This is a random term for each point, with a standard deviation of sigma
        eta = randn(N, N) .* sigma;
        phi_ground_truth = phi_ground_truth + eta;

        % mặt phẳng tham chiếu
        % Áp dụng trực tiếp phương trình của bạn lên toàn bộ lưới tọa độ
    case 'gaussian'
        [X, Y] = meshgrid(linspace(-1, 1, N), linspace(-1, 1, M));
        sigma = 0.5;
        phi_ground_truth = exp(-(X.^2 + Y.^2) / (2 * sigma^2)) * 2 * pi;
        % mặt phẳng tham chiếu
        % Áp dụng trực tiếp phương trình của bạn lên toàn bộ lưới tọa độ
    case 'sin'
        [X, Y] = meshgrid(linspace(-1, 1, N), linspace(-1, 1, M));
        freq_x = 2;  freq_y = 0;
        phi_ground_truth = 2 * pi * sin(2 * pi * freq_x * X) .* cos(2 * pi * freq_y * Y);
        % mặt phẳng tham chiếu
        % Áp dụng trực tiếp phương trình của bạn lên toàn bộ lưới tọa độ
    case 'zernike'
        % Ví dụ:
        indices = 1:10;
        zernike_coeffs = [
            0, 0, 0, 1, 2, 3, 1, 2, 3, 0.5, 1, 0.2, 0, 0, 0,...
            0, 0, 0, 0, 0, 0, 0, 0, 0.00000208, 0, 0, 0, 0,...
            0, 0, 0, 0, 0, 0, 0
            ];
        zernike_coeffs = zernike_coeffs';
        order = 15; % Bậc cao nhất của Zernike

        % <<< SỬA ĐỔI: Sử dụng biến N thay vì giá trị gán cứng
        % Giả định rằng mặt sóng Zernike là hình tròn nội tiếp trong một lưới vuông.
        % Chúng ta sẽ dùng chiều nhỏ hơn của M và N để đảm bảo.
        grid_size = min(M, N);

        wavefront = reconstruct_wavefront(zernike_coeffs, order, grid_size);

        % <<< SỬA ĐỔI: Cần resize wavefront về đúng kích thước M, N nếu cần
        if size(wavefront, 1) ~= M || size(wavefront, 2) ~= N
            phase_object = imresize(wavefront, [M, N]);
        else
            phase_object = wavefront;
        end

        % Phần còn lại của code Zernike...
        coeff = zeros(1, 2);
        coeff(1) = 20; coeff(2) = 10;
        [output_coeff, z_recon_map] = ZernikeLegendreFit(phase_object, "2indices", coeff);
        fprintf('He so tái tạo: %.1f \n', output_coeff{1});
        [output_coeff_no_tilt, z_recon_no_tilt] = removeTiltFromZernike(output_coeff, "2indices", z_recon_map);
        phi_ground_truth = z_recon_no_tilt;
        % mặt phẳng tham chiếu
        % Áp dụng trực tiếp phương trình của bạn lên toàn bộ lưới tọa độ
        phase_offset = 2 * pi * (fx *X + fy * Y);
    case 'step'
        phase_jump_magnitude = 5 * pi;
        phi_ground_truth = zeros(M, N);
        % <<< SỬA ĐỔI: middle_column đã dùng N nên không cần sửa
        middle_column = round(N / 2);
        phi_ground_truth(:, middle_column:end) = phase_jump_magnitude;
        % mặt phẳng tham chiếu
        % Áp dụng trực tiếp phương trình của bạn lên toàn bộ lưới tọa độ
        [X, Y] = meshgrid(linspace(-1, 1, N), linspace(-1, 1, M));
        phase_offset = 2 * pi * (fx *X + fy * Y);

    case 'sharp_peak'
        % <<< SỬA ĐỔI: Bỏ các biến rows, cols gán cứng và dùng trực tiếp M, N
        [X, Y] = meshgrid(1:N, 1:M); % Dùng N cho chiều rộng, M cho chiều cao
        cx = N/2;
        cy = M/2;

        Z_base = 0.001 * ((X - cx).^2 + (Y - cy).^2);
        sigma = 20;
        amplitude = 10 * pi;
        Z_peak = amplitude * exp(- ((X - cx).^2 + (Y - cy).^2) / (2 * sigma^2));
        phi_ground_truth = Z_base + Z_peak;
        % mặt phẳng tham chiếu
        % Áp dụng trực tiếp phương trình của bạn lên toàn bộ lưới tọa độ
        phase_offset = 2 * pi * (fx *X + fy * Y);
    case 'multi_sharp_peak'
        % <<< SỬA ĐỔI: Bỏ các biến rows, cols gán cứng và dùng trực tiếp M, N
        [X, Y] = meshgrid(1:N, 1:M);
        center_x = N/2;
        center_y = M/2;

        Z_base = 0.0005 * ((X - center_x).^2 + (Y - center_y).^2);

        % <<< SỬA ĐỔI: Định nghĩa vị trí các đỉnh theo tỷ lệ của kích thước ảnh
        % Điều này đảm bảo bố cục các đỉnh không đổi khi M, N thay đổi
        peaks(1).cx = round(N * (100/512)); peaks(1).cy = round(M * (150/600));
        peaks(1).amplitude = 12 * pi;
        peaks(1).sigma = 30;

        peaks(2).cx = round(N * (400/512)); peaks(2).cy = round(M * (380/600));
        peaks(2).amplitude = 10 * pi;
        peaks(2).sigma = 30;

        peaks(3).cx = round(N * (256/512)); peaks(3).cy = round(M * (256/600));
        peaks(3).amplitude = 15 * pi;
        peaks(3).sigma = 30;

        peaks(4).cx = round(N * (120/512)); peaks(4).cy = round(M * (400/600));
        peaks(4).amplitude = 9 * pi;
        peaks(4).sigma = 30;

        Z_peaks_total = zeros(M, N); % Khởi tạo theo đúng kích thước M, N
        for i = 1:length(peaks)
            p = peaks(i);
            Z_current_peak = p.amplitude * exp(- ((X - p.cx).^2 + (Y - p.cy).^2) / (2 * p.sigma^2));
            Z_peaks_total = Z_peaks_total + Z_current_peak;
        end

        phi_ground_truth = Z_base + Z_peaks_total;
        phase_offset = 2 * pi * (fx *X + fy * Y);

    case 'sinc_spike'
        % Case này đã được viết tốt, dùng M, N nên không cần sửa
        [X, Y] = meshgrid(linspace(-4, 4, N), linspace(-4, 4, M));
        r = sqrt(X.^2 + Y.^2);
        Z_sinc = sin(pi * r) ./ (pi * r);
        Z_sinc(isnan(Z_sinc)) = 1;  % sinc(0) = 1
        Z_sinc = Z_sinc * 2 * pi;
        sigma = 0.12;
        amplitude = 6 * pi;
        Z_spike = amplitude * exp(-(X.^2 + Y.^2) / (2 * sigma^2));
        phi_ground_truth = Z_sinc + Z_spike;
        % mặt phẳng tham chiếu
        % Áp dụng trực tiếp phương trình của bạn lên toàn bộ lưới tọa độ
        phase_offset = 2 * pi * (fx *X + fy * Y);
    case 'microsphere_array'
        % Case này cũng đã được viết tốt, dùng M, N nên không cần sửa
        image = zeros(M, N);
        gridSize = 3;
        A = 10;
        sigma = 5;
        spacingX = N / (gridSize + 1);
        spacingY = M / (gridSize + 1);
        [X, Y] = meshgrid(1:N, 1:M);
        for row = 1:gridSize
            for col = 1:gridSize
                centerX = col * spacingX;
                centerY = row * spacingY;
                gaussian = A * exp(- ((X - centerX).^2 + (Y - centerY).^2) / (2*sigma^2));
                image = image + gaussian;
            end
        end
        phi_ground_truth = image;
        % mặt phẳng tham chiếu
        % Áp dụng trực tiếp phương trình của bạn lên toàn bộ lưới tọa độ
        phase_offset = 2 * pi * (fx *X + fy * Y);
    case "test"

        %   N = 200;
        [x, sigma] = meshgrid(linspace(0, 1, N), linspace(0, pi/5, N));

        % Define the constant 'a'
        a = N * pi / 2;

        % Define the zero-mean Gaussian noise component
        % This is a random term for each point, with a standard deviation of sigma
        eta = randn(N, N) .* sigma;

        % Calculate the unwrapped phase distribution
        phi = a * x.^2 + eta;
        phi_ground_truth = phi;
    case 'test_ls'
        % Thông số ảnh

        noise_level = 0.05;   % nhiễu Gaussian
        phase_scale = 0.05;   % điều chỉnh độ dốc pha

        % Lưới toạ độ
        [x, y] = meshgrid(1:N, 1:M);

        % Tạo bề mặt parabol (pha gốc)
        phi_true = phase_scale * ((x - N/2).^2 + (y - M/2).^2);

        % Thêm nhiễu Gaussian
        phi_noisy = phi_true + noise_level * randn(M, N);
        phi_ground_truth = phi_noisy;

    case 'residual'
        [x, y] = meshgrid(linspace(-1, 1, M), linspace(-1, 1, N));

        % Tạo bề mặt pha gốc (unwrapped phase)
        phase_unwrapped = (2*x).^2 +y.^2;

%         % Bao pha vào [-pi, pi]
%         phase_wrapped = mod(phase_unwrapped + pi, 2*pi) - pi;

        % Chèn điểm residual: tại vị trí (50,50) và (70,30)
%         phase_unwrapped(50, 50) = phase_unwrapped(50, 50) + pi; % tạo discontinuity
%         phase_unwrapped(70, 30) = phase_unwrapped(70, 30) - pi;
%         phase_unwrapped(:, N/2+1:end) = pi;  % Nhảy bậc π tại giữa ảnh
        phi_ground_truth = phase_unwrapped;
    case "high_gradient"
        [x, y] = meshgrid(0:N-1, 0:M-1);

        A = 5 * pi;           % Biên độ lớn hơn π
        fx = 0.04;            % f_x đủ lớn để gây wrap mạnh

        phi_ground_truth = A * sin(2 * pi * fx * x);  % Bề mặt pha
    case "high_noise"
        [x, sigma] = meshgrid(linspace(0, 1, N), linspace(0, pi/5, N));

        % Define the constant 'a'
        a = N/4 * pi / 2;

        % Define the zero-mean Gaussian noise component
        % This is a random term for each point, with a standard deviation of sigma
%         eta = randn(N, N) .* sigma;

        % Calculate the unwrapped phase distribution
        phi_ground_truth = a * x.^2 ;
    otherwise
        error('Loại pha chưa được định nghĩa!');
end

% --- Hiển thị kết quả (tùy chọn) ---
figure;
imagesc(phi_ground_truth);
axis image;
colorbar;
title(['Đối tượng pha: ', strrep(phase_type, '_', ' ')]);

% --- Tạo hologram từ pha gốc ---
hologram = generate_test_hologram(M, N, fx, fy, phi_ground_truth, noise_level);
hologram_abs = mat2gray(hologram);
imwrite(hologram_abs, 'hologram.bmp');


%% 3. TÁI TẠO PHA BẰNG PHƯƠNG PHÁP BIẾN ĐỔI FOURIER (FFT)
fprintf('--> Bước 2: Tái tạo pha Wrapped bằng FFT...\n');
if(auto_fft)
    [wrappedPhase, ~] = reconstruct_phase_auto(hologram, params);
else
    [wrappedPhase, ~] = reconstruct_phase_interactively(hologram, struct());
end
% [wrappedPhase, ~] = remove_tilt_from_wrapped2(wrappedPhase); % Tùy chọn xóa nghiêng

%% 4. ƯỚC LƯỢNG PHA BẰNG PHƯƠNG PHÁP PHÂN TÍCH VÂN
fprintf('--> Bước 3: Ước lượng pha thô bằng phân tích vân...\n');
% Làm mảnh và gán bậc vân
skeleton_image = skeletonize_zhang_suen(hologram_abs, true);

% % --- Bước 1: Nhị phân hóa ảnh bằng Otsu ---
% 
% fprintf('Bước 1/3: Nhị phân hóa ảnh bằng phương pháp Otsu...\n');
% thresh = graythresh(hologram_abs);
% BW_Original = imbinarize(hologram_abs, thresh);
% 
% fprintf('Ngưỡng Otsu: %.4f\n', thresh);
% fprintf('Số pixel foreground: %d\n', sum(BW_Original(:)));
% skeleton_image = mzs_thinning(BW_Original);
% figure;
% imshow(skeleton_image);
% title('Skeleton improved', 'FontSize', 12);


skeleton_image = xoa_ria(skeleton_image);


% Đã có ảnh skeleton hóa tên là 'bw' và lambda



[~, labels, img] = assign_fringe_order(skeleton_image, true);

% [~, labels, img] = assign_fringe_order_improved(skeleton_image, true);

% Tái tạo bề mặt từ vân
[phi_est, ~] = reconSurface_linearPushed(img, labels, 632.8e-9, 'None', false);





%% 5. GIẢI BỌC PHA VÀ TINH CHỈNH
fprintf('--> Bước 4: Giải bọc pha và tinh chỉnh kết quả...\n');
% --- Giải bọc pha sử dụng pha ước lượng ---
[phi_est, wrappedPhase, phi_ground_truth] = crop_multiple_to_smallest(phi_est, wrappedPhase, phi_ground_truth);

[finalUnwrappedPhase, kMap] = unwrapUsingEstimate(phi_est, wrappedPhase);
fprintf("chay k map");
  [kMap, phi_plane_kMap] = remove_plane_manual(kMap);
  kMap = round( kMap);
  fprintf("ket thuc kmap");



% cutoff_ratio = 0.5;
% [kMap, ~] = remove_tilt_simple(kMap, cutoff_ratio);
% 
% figure;
% subplot(1,2,1);
% surf(kMap,"EdgeColor","none"); colorbar;
% title("Final truoc khi refine");
% 
% kMap = iterative_k_correction(kMap);
% 
% subplot(1,2,2);
% surf(kMap,"EdgeColor","none");
% title("Final truoc khi refine");
% 
[finalUnwrappedPhase, ~, ~] = correct_sparse_artifacts_iterative(finalUnwrappedPhase, ...
    'BoundaryCondition', 'symmetric', 'BoundaryWidth', 2, 'MaxIterations', 150);


% % Gọi hàm refine
% [finalUnwrappedPhase, artifact_mask] = refine_sparse_artifacts(finalUnwrappedPhase, ...
%     'WindowSize', 5, ...
%     'Threshold', pi, ...
%     'MaxIterations', 5, ...
%     'Verbose', true);


% [finalUnwrappedPhase, ~, ~] = correct_sparse_artifacts_iterative_v2(finalUnwrappedPhase, ...
%     'BoundaryCondition', 'symmetric', 'BoundaryWidth', 2, 'MaxIterations', 150);
% finalUnwrappedPhase = iterative_median_unwrap(finalUnwrappedPhase);
%%
% [finalUnwrappedPhase, fitted_plane] = remove_tilt_least_squares_gemini(finalUnwrappedPhase);


cutoff_ratio = 0.5;
% [finalUnwrappedPhase, ~] = remove_tilt_simple(finalUnwrappedPhase, cutoff_ratio);

finalUnwrappedPhase = finalUnwrappedPhase(6:end-5, 6:end-5);
[finalUnwrappedPhase, phi_est_aligned, wrappedPhase_aligned, phi_ground_truth_aligned] = ...
                        crop_multiple_to_smallest(finalUnwrappedPhase, phi_est, wrappedPhase, phi_ground_truth);

%% 5.5 CÁC THUẬT TOÁN UNWRAPPING KHÁC
unwrapped_Phase_LS_DCT = unwrapping.unwrapPhase(wrappedPhase, 'ls', 'dct'); % LS với DCT
unwrapped_Phase_TIE_FFT = unwrapping.unwrapPhase(wrappedPhase, 'tie', 'fft'); % TIE với FFT
unwrapped_Phase_noncontinue = unwrapping.unwrapPhase(wrappedPhase, 'linh'); % Phương pháp của a Linh
unwrapped_Phase_2dweight = unwrapping.unwrapPhase(wrappedPhase, '2dweight'); % 2D weighted phase unwrapping
unwrapped_Phase_2dweight = goldstein_unwrap(wrappedPhase);

% loại bỏ nghiêng
% [unwrapped_Phase_LS_DCT, ~] = remove_tilt_simple(unwrapped_Phase_LS_DCT, cutoff_ratio);
% [unwrapped_Phase_TIE_FFT, ~] = remove_tilt_simple(unwrapped_Phase_TIE_FFT, cutoff_ratio);
% [unwrapped_Phase_noncontinue, ~] = remove_tilt_simple(unwrapped_Phase_noncontinue, cutoff_ratio);
% [unwrapped_Phase_2dweight, fitted_plane] = remove_tilt_simple(unwrapped_Phase_2dweight, cutoff_ratio);

  [finalUnwrappedPhase, unwrapped_Phase_LS_DCT, unwrapped_Phase_TIE_FFT, unwrapped_Phase_noncontinue, unwrapped_Phase_2dweight, ...
      phi_ground_truth_aligned, kMap] = ...
                        crop_multiple_to_smallest(finalUnwrappedPhase, unwrapped_Phase_LS_DCT, unwrapped_Phase_TIE_FFT,...
                        unwrapped_Phase_noncontinue, unwrapped_Phase_2dweight,phi_ground_truth, kMap);


  %% OfFset lại bề mặt bằng cách trừ đi mặt phẳng tham chiếu
  [unwrapped_Phase_LS_DCT, phi_plane] = remove_plane_manual(unwrapped_Phase_LS_DCT);
phase_offset = phi_plane;
%   unwrapped_Phase_LS_DCT = unwrapped_Phase_LS_DCT - phase_offset;
  unwrapped_Phase_TIE_FFT = unwrapped_Phase_TIE_FFT - phase_offset;
  unwrapped_Phase_noncontinue = unwrapped_Phase_noncontinue - phase_offset;
  unwrapped_Phase_2dweight = unwrapped_Phase_2dweight - phase_offset;
  finalUnwrappedPhase = finalUnwrappedPhase - phase_offset;
% offset về 0
  unwrapped_Phase_LS_DCT  =  unwrapped_Phase_LS_DCT- min(unwrapped_Phase_LS_DCT(:));
  unwrapped_Phase_TIE_FFT  =  unwrapped_Phase_TIE_FFT- min(unwrapped_Phase_TIE_FFT(:));
  unwrapped_Phase_noncontinue  =  unwrapped_Phase_noncontinue- min(unwrapped_Phase_noncontinue(:));
  unwrapped_Phase_2dweight  =  unwrapped_Phase_2dweight- min(unwrapped_Phase_2dweight(:));
  finalUnwrappedPhase  =  finalUnwrappedPhase- min(finalUnwrappedPhase(:));

%% 6. PHÂN TÍCH SAI SỐ
fprintf('--> Bước 5: Tính toán và hiển thị sai số...\n');

% Sai số RMS - và tuyệt đối
calculateAndCompareErrors(phi_ground_truth_aligned, unwrapped_Phase_LS_DCT,...
                unwrapped_Phase_TIE_FFT, unwrapped_Phase_noncontinue,...
                unwrapped_Phase_2dweight,finalUnwrappedPhase);


% Hien thi ket qua
% --- Hiển thị bản đồ sai số ---
figure("Name","Kết quả LS DCT ");
surf(unwrapped_Phase_LS_DCT, 'EdgeColor', 'none');
title("Kết quả LS DCT");
xlabel('x'); ylabel('y'); zlabel('(rad)');
colormap; colorbar; 
figure("Name","Kết quả thuật toán TIE FFT");
surf(unwrapped_Phase_TIE_FFT, 'EdgeColor', 'none');
title("Kết quả thuật toán TIE FFT");
xlabel('x'); ylabel('y'); zlabel('(rad)');
colormap; colorbar; 
figure("Name","Sử dụng thuật toán following non-continuous path");
surf(unwrapped_Phase_noncontinue, 'EdgeColor', 'none');
title("Sử dụng thuật toán Following non-continuous path");
xlabel('x'); ylabel('y'); zlabel('(rad)');
colormap; colorbar; 

figure("Name","Kết quả 2D-weight");
surf(unwrapped_Phase_2dweight, 'EdgeColor', 'none');
title("Kết quả 2D-weight");
xlabel('x'); ylabel('y'); zlabel('(rad)');
colormap; colorbar; 

% % --- Hiển thị bản đồ sai số ---
% figure('Name', 'Phân Tích Sai Số Chi Tiết');
% % Sai số giữa pha cuối và pha gốc
% surf((finalUnwrappedPhase - phi_ground_truth_aligned), 'EdgeColor', 'none');
% title({'Bản Đồ Sai Số Tuyệt Đối', '(Pha Cuối vs. Pha Gốc)'});
% xlabel('x'); ylabel('y'); zlabel('Sai số (rad)');
% colormap; colorbar; 

%% 6. PHÂN TÍCH SAI SỐ (TIẾP THEO)
% --- Tính toán sai số cho các thuật toán khác ---
error_LS_DCT = unwrapped_Phase_LS_DCT - phi_ground_truth_aligned;
error_TIE_FFT = unwrapped_Phase_TIE_FFT - phi_ground_truth_aligned;
error_noncontinue = unwrapped_Phase_noncontinue - phi_ground_truth_aligned;
error_2dweight = unwrapped_Phase_2dweight - phi_ground_truth_aligned;
error_proposal = (finalUnwrappedPhase - phi_ground_truth_aligned);
% Chuẩn hoá lỗi
error_LS_DCT = error_LS_DCT-min(error_LS_DCT(:));
error_TIE_FFT = error_TIE_FFT-min(error_TIE_FFT(:));
error_noncontinue = error_noncontinue-min(error_noncontinue(:));
error_2dweight = error_2dweight-min(error_2dweight(:));
error_proposal = error_proposal-min(error_proposal(:));

% --- Hiển thị bản đồ sai số của 5 thuật toán trên cùng 1 figure ---
figure('Name', 'So Sánh Toàn Diện Bản Đồ Sai Số Của Các Thuật Toán', 'WindowState', 'maximized');

% --- HÀNG 1: BIỂU ĐỒ SAI SỐ 3D ---

% 1. Sai số của LS với DCT
subplot(2, 5, 1);
surf(error_LS_DCT, 'EdgeColor', 'none');
title({'LS-DCT (3D)'});
xlabel('x'); ylabel('y'); zlabel('Sai số (rad)');
axis tight;
colormap(gca, 'jet');

% 2. Sai số của TIE với FFT
subplot(2, 5, 2);
surf(error_TIE_FFT, 'EdgeColor', 'none');
title({'TIE-FFT (3D)'});
xlabel('x'); ylabel('y');
axis tight;
colormap(gca, 'jet');

% 3. Sai số của phương pháp path-following
subplot(2, 5, 3);
surf(error_noncontinue, 'EdgeColor', 'none');
title({'Path-following (3D)'});
xlabel('x'); ylabel('y');
axis tight;
colormap(gca, 'jet');

% 4. Sai số của 2D weighted
subplot(2, 5, 4);
surf(error_2dweight, 'EdgeColor', 'none');
title({'2D Weighted (3D)'});
xlabel('x'); ylabel('y');
axis tight;
colormap(gca, 'jet');

% 5. Sai số của Proposal
subplot(2, 5, 5);
surf(error_proposal, 'EdgeColor', 'none');
title({'Proposal (3D)'});
xlabel('x'); ylabel('y');
axis tight;
colormap(gca, 'jet');

% --- HÀNG 2: BẢN ĐỒ SAI SỐ 2D ---

% 6. Sai số của LS với DCT (2D)
subplot(2, 5, 6);
imagesc(error_LS_DCT);
title({'LS-DCT (2D)'});
xlabel('x'); ylabel('y');
axis image; % Giữ đúng tỉ lệ ảnh
colorbar;
colormap(gca, 'jet');

% 7. Sai số của TIE với FFT (2D)
subplot(2, 5, 7);
imagesc(error_TIE_FFT);
title({'TIE-FFT (2D)'});
xlabel('x'); ylabel('y');
axis image;
colorbar;
colormap(gca, 'jet');

% 8. Sai số của phương pháp path-following (2D)
subplot(2, 5, 8);
imagesc(error_noncontinue);
title({'Path-following (2D)'});
xlabel('x'); ylabel('y');
axis image;
colorbar;
colormap(gca, 'jet');

% 9. Sai số của 2D weighted (2D)
subplot(2, 5, 9);
imagesc(error_2dweight);
title({'2D Weighted (2D)'});
xlabel('x'); ylabel('y');
axis image;
colorbar;
colormap(gca, 'jet');

% 10. Sai số của Proposal (2D)
subplot(2, 5, 10);
imagesc(error_proposal);
title({'Proposal (2D)'});
xlabel('x'); ylabel('y');
axis image;
colorbar;
colormap(gca, 'jet');

% Thêm tiêu đề chung cho toàn bộ figure
sgtitle('So Sánh Bản Đồ Sai Số Tuyệt Đối (vs. Ground Truth)');

 
%%

%% 7. HIỂN THỊ KẾT QUẢ TỔNG QUAN
fprintf('--> Bước 6: Hiển thị kết quả cuối cùng...\n');
create_overview_visualization(phi_ground_truth_aligned, phi_est_aligned, ...
    wrappedPhase_aligned, finalUnwrappedPhase, kMap);
%% 8. HIỂN THỊ MẶT CẮT NGANG SAI SỐ
fprintf('--> Bước 7: Vẽ đồ thị mặt cắt ngang sai số...\n');

% Lấy chỉ số của hàng ở giữa từ một trong các ma trận sai số
[num_rows, ~] = size(error_LS_DCT);
middle_row_index = round(num_rows / 2);

% Trích xuất dữ liệu mặt cắt ngang (hàng giữa) từ mỗi ma trận sai số
cross_section_final = error_proposal(middle_row_index, :);
cross_section_LS_DCT = error_LS_DCT(middle_row_index, :);
cross_section_TIE_FFT = error_TIE_FFT(middle_row_index, :);
cross_section_noncontinue = error_noncontinue(middle_row_index, :);
cross_section_2dweight = error_2dweight(middle_row_index, :);
 
% --- Vẽ đồ thị 2D so sánh các mặt cắt ngang ---
figure('Name', 'Mặt Cắt Ngang Sai Số');
hold on; % Giữ các đồ thị được vẽ trên cùng một trục


plot(cross_section_LS_DCT, 'DisplayName', 'LS-DCT', 'LineStyle', '--');
plot(cross_section_TIE_FFT, 'DisplayName', 'TIE-FFT', 'Color','g');
plot(cross_section_noncontinue, 'DisplayName', 'Phương pháp Path-following', "LineWidth",2);
plot(cross_section_2dweight, 'DisplayName', '2D Weighted', 'LineWidth', 1.5,'LineStyle', '-.');
plot(cross_section_final, 'DisplayName', 'Proposal', 'LineWidth', 1);
hold off; % Thả trục

% --- Thêm các chi tiết cho đồ thị ---
title(['Mặt Cắt Ngang Sai Số Tại Hàng y = ' num2str(middle_row_index)]);
xlabel('Chỉ số cột (x)');
ylabel('Sai số tuyệt đối (rad)');
legend('show', 'Location', 'best'); % Hiển thị chú giải ở vị trí tốt nhất
grid on; % Bật lưới để dễ đọc
axis tight; % Điều chỉnh trục cho vừa vặn
%% 8.5 Hiển thị mặt cắt ngang kết quả
fprintf('--> Bước 7: Vẽ đồ thị mặt cắt ngang sai số...\n');

% Lấy chỉ số của hàng ở giữa từ một trong các ma trận sai số
[num_rows, ~] = size(error_LS_DCT);
middle_row_index = round(num_rows / 2);
% offset data ve moc 0
finalUnwrappedPhase = finalUnwrappedPhase - min(finalUnwrappedPhase(:));
unwrapped_Phase_LS_DCT = unwrapped_Phase_LS_DCT - min(unwrapped_Phase_LS_DCT(:));
unwrapped_Phase_TIE_FFT = unwrapped_Phase_TIE_FFT - min(unwrapped_Phase_TIE_FFT(:));
unwrapped_Phase_noncontinue = unwrapped_Phase_noncontinue - min(unwrapped_Phase_noncontinue(:));
unwrapped_Phase_2dweight = unwrapped_Phase_2dweight - min(unwrapped_Phase_2dweight(:));
phi_ground_truth_aligned = phi_ground_truth_aligned - min(phi_ground_truth_aligned(:));

wrappedPhase_aligned = wrappedPhase_aligned - min(wrappedPhase_aligned(:));

% Trích xuất dữ liệu mặt cắt ngang (hàng giữa) từ mỗi ma trận sai số
cross_section_proposal_method = finalUnwrappedPhase(middle_row_index, :);
cross_section_LS_DCT_method = unwrapped_Phase_LS_DCT(middle_row_index, :);
cross_section_TIE_FFT_method = unwrapped_Phase_TIE_FFT(middle_row_index, :);
cross_section_noncontinue_method = unwrapped_Phase_noncontinue(middle_row_index, :);
cross_section_2dweight_medthod = unwrapped_Phase_2dweight(middle_row_index, :);
cross_section_ground_truth = phi_ground_truth_aligned(middle_row_index,:); 
cross_section_wrappedPhase = wrappedPhase_aligned(middle_row_index,:);
% --- Vẽ đồ thị 2D so sánh các mặt cắt ngang ---
figure('Name', 'Mặt Cắt Ngang Kết quả');
hold on; % Giữ các đồ thị được vẽ trên cùng một trục

plot(cross_section_LS_DCT_method, 'DisplayName', 'LS-DCT', 'LineStyle', '--');
plot(cross_section_TIE_FFT_method, 'DisplayName', 'TIE-FFT', 'LineStyle', ':');
plot(cross_section_noncontinue_method, 'DisplayName', 'noncontinue', 'LineStyle', '-.');
plot(cross_section_2dweight_medthod, 'DisplayName', '2D Weighted', 'LineWidth', 2);
plot(cross_section_ground_truth, 'DisplayName','Ground Truth', 'LineWidth', 1.5,"LineStyle","-.");
plot(cross_section_proposal_method, 'DisplayName', 'Proposal', 'LineWidth', 1);

% plot(cross_section_wrappedPhase, 'DisplayName','Wrapped Phase', 'LineWidth', 1, "LineStyle","-.");


hold off;  

% --- Thêm các chi tiết cho đồ thị ---
title(['Mặt Cắt Ngang Kết quả Tại Hàng y = ' num2str(middle_row_index)]);
xlabel('Chỉ số cột (x)');
ylabel('(rad)');
legend('show', 'Location', 'best'); % Hiển thị chú giải ở vị trí tốt nhất
grid on; % Bật lưới để dễ đọc
axis tight; % Điều chỉnh trục cho vừa vặn


fprintf('\nQuy trình đã hoàn thành!\n');

%% ========================================================================
function [wrappedPhase, params] = reconstruct_phase_interactively(hologram, params)
% RECONSTRUCT_PHASE_INTERACTIVELY_MASK Tái tạo pha từ hologram bằng cách
% dùng MẶT NẠ để lọc phổ bậc +1 một cách tương tác.
%
%   Input:
%       hologram - Ảnh hologram đầu vào (có thể là ảnh màu hoặc ảnh xám).
%       params   - Một struct chứa các tham số (tùy chọn).
%
%   Output:
%       wrappedPhase - Pha đã tái tạo (bị gói trong khoảng [-pi, pi]).
%       params       - Struct tham số được cập nhật (tùy chọn).

% 1. Chuyển đổi hologram sang ảnh xám nếu cần thiết.
if size(hologram, 3) == 3
    hologramGray = rgb2gray(hologram);
else
    hologramGray = hologram;
end

[numRows, numCols] = size(hologramGray);

% 2. Thực hiện biến đổi Fourier 2D và dịch chuyển thành phần tần số 0 về tâm.
fourierTransform = fftshift(fft2(double(hologramGray)));

% 3. Hiển thị phổ Fourier để người dùng lựa chọn.
figure('Name', 'Fourier Spectrum - Select +1 Order');
imshow(log(1 + abs(fourierTransform)), []);
title('Vẽ một hình chữ nhật quanh phổ bậc +1 rồi double-click');
xlabel('Tần số không gian (u)');
ylabel('Tần số không gian (v)');

% 4. Cho phép người dùng chọn vùng quan tâm (ROI) bằng tay.
[~, xRec, yRec, widthRec, heightRec] = myDrawRec();

% 5. TẠO MỘT MẶT NẠ (MASK) TỪ VÙNG ĐÃ CHỌN
%    Tạo một ma trận toàn số 0...
mask = zeros(numRows, numCols);
%    ...và đặt vùng chữ nhật đã chọn thành 1.
mask(yRec:yRec+heightRec-1, xRec:xRec+widthRec-1) = 1;

% 6. ÁP DỤNG MẶT NẠ VÀ DỊCH CHUYỂN VỀ TÂM
%    Nhân phổ gốc với mặt nạ để loại bỏ các tần số bên ngoài vùng chọn.
filteredSpectrum = fourierTransform .* mask;


% 7. Thực hiện biến đổi Fourier ngược để tái tạo trường sóng phức.
complexField = ifft2(ifftshift(filteredSpectrum));

% 8. Lấy pha từ trường phức.
wrappedPhase = angle(complexField);

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
function [unwrappedPhase, kMap] = unwrapUsingEstimate(estimatedPhase, wrappedPhase)
% Giải Wrapped pha `wrappedPhase` dựa trên pha ước lượng `estimatedPhase`.
wrappedEstimate = wrapToPi(estimatedPhase);
kMap = round((estimatedPhase - wrappedEstimate) / (2*pi));
% 2. Smooth k using median filter (to suppress isolated outliers)
kMap = medfilt2(kMap, [3 3]);
unwrappedPhase = wrappedPhase + 2*pi * kMap;
end

% -------------------------------------------------------------------------

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


figure("Name","Gốc"); surf(phi_gt, 'EdgeColor', 'none'); title('Gốc'); axis tight; view(45, 30); colorbar;
figure("Name","Pha ước lượng"); surf(phi_est, 'EdgeColor', 'none'); title('Pha Ước lượng'); axis tight; view(45, 30); colorbar;
figure('Name', "Pha wrapped"); surf(phi_wrapped, 'EdgeColor', 'none'); title('Pha Wrapped'); axis tight; view(45, 30); colorbar;
figure("Name","Kết quả cuối cùng"); surf(phi_final, 'EdgeColor', 'none'); title('Kết quả Cuối cùng'); axis tight; view(45, 30); colorbar;
figure("Name", "Bản đồ K"); surf(kMap, 'EdgeColor', 'none'); title('Bản đồ K (Fringe Order)'); axis tight; view(45, 30); colormap(gca, parula); colorbar;

% figure(); imagesc(phi_gt); title('Gốc (2D)'); axis image; colorbar;
% figure(); imagesc(phi_est); title('Pha Ước lượng (2D)'); axis image; colorbar;
% figure(); imagesc(phi_wrapped); title('Pha Wrapped (2D)'); axis image; colorbar;
% figure(); imagesc(phi_final); title('Kết quả Cuối cùng (2D)'); axis image; colorbar;
% figure(); imagesc(kMap); title('Bản đồ K (2D)'); axis image; colormap(gca, parula); colorbar;
end

function [phi_corrected, phi_plane] = remove_plane_manual(phi)
%REMOVE_PLANE_MANUAL Cho phép người dùng chọn điểm hoặc vẽ HCN để nội suy và loại mặt phẳng nghiêng
% [phi_corrected, phi_plane] = remove_plane_manual(phi)
% - phi: bản đồ pha đầu vào
% - phi_corrected: bản đồ sau khi loại nghiêng
% - phi_plane: mặt phẳng đã nội suy

[N, M] = size(phi);
[X, Y] = meshgrid(1:M, 1:N);

% Kiểm tra và xử lý NaN/Inf trong dữ liệu đầu vào
if any(~isfinite(phi(:)))
    warning('Dữ liệu chứa NaN hoặc Inf. Đang thay thế bằng giá trị trung bình...');
    phi_mean = nanmean(phi(:));
    phi(~isfinite(phi)) = phi_mean;
end

% % --- Hiển thị ảnh ban đầu để người dùng chọn phương thức ---
% figure;
% surf(phi, "EdgeColor", "none");
% colormap jet;
% colorbar;
% title('Bản đồ pha gốc');
% 
% figure;
% imagesc(phi);
% axis image;
% colormap jet;
% colorbar;
% title('Bản đồ pha gốc');

% --- Hộp thoại lựa chọn phương thức ---
% choice = questdlg('Chọn phương thức để xác định mặt phẳng:', ...
%     'Lựa chọn nội suy', ...
%     'Chọn điểm', 'Vẽ HCN', 'Chọn điểm');
choice = "Vẽ HCN";
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
        %         rect = getrect; % [xmin ymin width height]
        %
        %         % Lấy tọa độ 4 góc từ hình chữ nhật
        %         xmin = rect(1);
        %         ymin = rect(2);
        %         width = rect(3);
        %         height = rect(4);
        %         x_pts = [xmin; xmin + width; xmin + width; xmin];
        %         y_pts = [ymin; ymin; ymin + height; ymin + height];
        % Lấy kích thước của ma trận phi
        [rows, cols] = size(phi);

        % Xác định tọa độ x (cột) và y (hàng) của 4 góc
        % Thứ tự: trên-trái, trên-phải, dưới-phải, dưới-trái
        offset = 5;
        x_pts = [offset;    cols-offset; cols-offset; offset];
        y_pts = [offset;    offset;    rows-offset; rows-offset];
        width = cols -2*offset;
        height = rows - 2*offset;
        
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

% --- Kiểm tra và làm sạch tọa độ điểm ---
% Đảm bảo tọa độ nằm trong phạm vi hợp lệ
x_pts = max(1, min(M, x_pts));
y_pts = max(1, min(N, y_pts));

% --- Lấy giá trị Z tại các điểm đã chọn ---
z_pts = interp2(phi, x_pts, y_pts);

% Kiểm tra và loại bỏ các điểm có giá trị NaN
valid_pts = isfinite(x_pts) & isfinite(y_pts) & isfinite(z_pts);

if sum(valid_pts) < 3
    warning('Không đủ điểm hợp lệ để fit mặt phẳng (cần ít nhất 3 điểm). Trả về dữ liệu gốc.');
    phi_corrected = phi;
    phi_plane = zeros(N, M);
    return;
end

% Lọc các điểm hợp lệ
x_pts = x_pts(valid_pts);
y_pts = y_pts(valid_pts);
z_pts = z_pts(valid_pts);

% % --- Hiển thị lại ảnh với các điểm đã chọn ---
% figure;
% imagesc(phi);
% axis image;
% colormap jet;
% hold on;
% plot(x_pts, y_pts, 'rx', 'MarkerSize', 12, 'LineWidth', 2);
% 
% if strcmp(choice, 'Vẽ HCN')
%     % Vẽ lại hình chữ nhật để xác nhận
%     rect_x = [x_pts' x_pts(1)];
%     rect_y = [y_pts' y_pts(1)];
%     plot(rect_x, rect_y, 'r-', 'LineWidth', 2);
% end
% 
% for i = 1:length(x_pts)
%     text(x_pts(i) + 5, y_pts(i), sprintf('%d', i), ...
%         'Color', 'w', 'FontSize', 10, 'FontWeight', 'bold');
% end
% title('Pha gốc với các điểm nội suy đã chọn');
% hold off;

% --- Fit mặt phẳng với xử lý lỗi ---
try
    % Phương pháp 1: Sử dụng fit() với dữ liệu đã làm sạch
    tbl = table(x_pts, y_pts, z_pts, 'VariableNames', {'x', 'y', 'z'});
    fit_model = fit([tbl.x, tbl.y], tbl.z, 'poly11'); % poly11: f(x,y) = p00 + p10*x + p01*y

    % Tạo mặt phẳng đã khớp trên toàn bộ lưới tọa độ
    phi_plane = fit_model(X, Y);


end

% Kiểm tra kết quả phi_plane
if any(~isfinite(phi_plane(:)))
    warning('Mặt phẳng fit chứa NaN hoặc Inf. Đang thay thế...');
    phi_plane(~isfinite(phi_plane)) = 0;
end

% --- Trừ mặt phẳng (nghiêng) khỏi pha gốc ---
phi_corrected = phi - phi_plane;

% % --- Hiển thị kết quả ---
% figure;
% sgtitle('Kết quả loại bỏ mặt phẳng nghiêng');
% 
% subplot(1,3,1);
% imagesc(phi);
% axis image;
% colormap turbo;
% colorbar;
% title('Pha gốc');
% 
% subplot(1,3,2);
% imagesc(phi_plane);
% axis image;
% colormap turbo;
% colorbar;
% title('Mặt phẳng đã fit');
% 
% subplot(1,3,3);
% imagesc(phi_corrected);
% axis image;
% colormap turbo;
% colorbar;
% title('Pha đã loại nghiêng');
% 
% % In thông tin về quá trình fit
% fprintf('Đã sử dụng %d điểm để fit mặt phẳng.\n', length(x_pts));
% fprintf('Phạm vi giá trị pha gốc: [%.3f, %.3f]\n', min(phi(:)), max(phi(:)));
% fprintf('Phạm vi giá trị pha đã hiệu chỉnh: [%.3f, %.3f]\n', min(phi_corrected(:)), max(phi_corrected(:)));

end

function [phi_corrected, plane_est] = remove_tilt_from_wrapped_by_rect(phi_wrapped)
% REMOVE_TILT_FROM_WRAPPED_BY_RECT - Loại bỏ mặt phẳng nghiêng từ ảnh pha wrapped.
%
% Hàm này cho phép người dùng vẽ một hình chữ nhật trên một vùng được coi là phẳng
% của ảnh pha. Sau đó, nó sẽ nội suy một mặt phẳng từ 4 góc của hình chữ nhật
% và trừ mặt phẳng này khỏi toàn bộ ảnh để loại bỏ độ nghiêng.
%
% Inputs:
%   phi_wrapped - ảnh pha đã wrap trong khoảng [-pi, pi].
%
% Outputs:
%   phi_corrected - ảnh pha đã loại bỏ mặt phẳng nghiêng (kết quả vẫn được wrap).
%   plane_est     - mặt phẳng nghiêng đã được ước lượng (a*x + b*y + c).

% --- 1. Kiểm tra dữ liệu đầu vào ---
if nargin < 1
    error('Cần ít nhất 1 tham số đầu vào: ảnh pha wrapped.');
end
if ~ismatrix(phi_wrapped) || ~isnumeric(phi_wrapped)
    error('phi_wrapped phải là một ma trận số.');
end

[rows, cols] = size(phi_wrapped);
[X, Y] = meshgrid(1:cols, 1:rows);

try
    % --- 2. Hiển thị ảnh và yêu cầu người dùng vẽ hình chữ nhật ---
    fprintf('Vui lòng vẽ một hình chữ nhật trên vùng nền phẳng.\n');
    fprintf('Sau khi vẽ xong, double-click vào bên trong hình chữ nhật để xác nhận.\n');

    % Hiển thị ảnh 3D để người dùng có cái nhìn tổng quan
    figure('Name', 'Ảnh Pha 3D Gốc');
    surf(phi_wrapped, "EdgeColor", "none");
    colormap(jet); colorbar;
    title("Ảnh pha wrapped ban đầu (3D)");

    % Mở cửa sổ ảnh 2D để người dùng vẽ
    fig_draw = figure('Name', 'Vẽ vùng chọn');
    imagesc(phi_wrapped);
    axis image;
    colormap(gca, gray);
    colorbar;
    title('Vẽ hình chữ nhật (double-click để xác nhận)');

    % Cho phép người dùng vẽ hình chữ nhật
    h_rect = drawrectangle('Color', 'g', 'LineWidth', 1, 'StripeColor', 'm');
    wait(h_rect); % Chờ cho đến khi người dùng double-click

    rect_pos = round(h_rect.Position); % Lấy tọa độ [x, y, width, height]
    close(fig_draw); % Đóng cửa sổ vẽ sau khi xác nhận

    % Kiểm tra tính hợp lệ của hình chữ nhật
    if rect_pos(3) < 3 || rect_pos(4) < 3
        error('Vùng chọn quá nhỏ. Vui lòng chọn một vùng lớn hơn.');
    end

    % --- 3. Lấy tọa độ và giá trị pha tại 4 góc ---
    x1 = max(1, rect_pos(1));
    y1 = max(1, rect_pos(2));
    x2 = min(cols, x1 + rect_pos(3) - 1);
    y2 = min(rows, y1 + rect_pos(4) - 1);

    corner_x = [x1, x2, x1, x2]; % 4 góc: trái-trên, phải-trên, trái-dưới, phải-dưới
    corner_y = [y1, y1, y2, y2];

    % Lấy giá trị pha tại 4 điểm góc
    corner_phases = zeros(4, 1);
    for i = 1:4
        corner_phases(i) = phi_wrapped(corner_y(i), corner_x(i));
    end

    % --- 4. Fit mặt phẳng từ 4 điểm góc ---
    % Giải hệ phương trình: a*x + b*y + c = z
    A = [corner_x(:), corner_y(:), ones(4, 1)];

    % Sử dụng toán tử '\' để tìm các hệ số [a; b; c]
    coeffs = A \ corner_phases(:);

    % --- 5. Tính toán và hiệu chỉnh ---
    a = coeffs(1);
    b = coeffs(2);
    c = coeffs(3);
    plane_est = a*X + b*Y + c; % Tạo mặt phẳng nghiêng trên toàn ảnh

    % Trừ mặt phẳng nghiêng và wrap lại kết quả trong khoảng [-pi, pi]
    phi_corrected = wrapToPi(phi_wrapped - plane_est);

    % --- 6. Hiển thị kết quả ---
    fprintf('Đã loại bỏ nghiêng thành công.\n');
    fprintf('Tham số mặt phẳng: a=%.6f, b=%.6f, c=%.6f\n', a, b, c);

    % Figure so sánh 2D
    figure('Name', 'Kết quả loại bỏ nghiêng (2D)', 'NumberTitle', 'off');
    subplot(2, 2, 1);
    imagesc(phi_wrapped);
    axis image; title('Pha Wrapped gốc'); colormap(jet); colorbar;

    subplot(2, 2, 2);
    imagesc(phi_corrected);
    axis image; title('Sau khi loại nghiêng'); colormap(jet); colorbar;

    subplot(2, 2, 3);
    imagesc(plane_est);
    axis image; title('Mặt phẳng nghiêng ước lượng'); colormap(jet); colorbar;

    subplot(2, 2, 4);
    imagesc(wrapToPi(phi_wrapped - phi_corrected));
    axis image; title('Độ lệch đã loại bỏ (wrap)'); colormap(jet); colorbar;

    % Figure so sánh 3D
    figure('Name', 'Kết quả loại bỏ nghiêng (3D)', 'NumberTitle', 'off');
    subplot(1, 2, 1);
    mesh(X, Y, phi_wrapped);
    title('Wrapped phase (gốc)');
    xlabel('x'); ylabel('y'); zlabel('Pha');
    colormap(jet); colorbar; view(45, 30);

    subplot(1, 2, 2);
    mesh(X, Y, phi_corrected);
    title('Sau khi loại nghiêng');
    xlabel('x'); ylabel('y'); zlabel('Pha');
    colormap(jet); colorbar; view(45, 30);

catch ME
    % Xử lý lỗi nếu người dùng đóng cửa sổ hoặc có lỗi khác xảy ra
    fprintf('Lỗi xảy ra trong quá trình xử lý: %s\n', ME.message);
    fprintf('Hàm đã bị hủy. Trả về ảnh gốc.\n');
    phi_corrected = phi_wrapped;
    plane_est = zeros(size(phi_wrapped));

    % Đảm bảo đóng figure vẽ nếu nó còn tồn tại
    if exist('fig_draw', 'var') && isvalid(fig_draw)
        close(fig_draw);
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
    addParameter(p, 'FilterSize', [3 3], @(x) isnumeric(x) && length(x) == 2);
    addParameter(p, 'Epsilon', 1e-6, @(x) isnumeric(x) && x > 0);
    addParameter(p, 'MaxIterations', 100, @(x) isnumeric(x) && x > 0);
    addParameter(p, 'Verbose', false, @islogical);
    addParameter(p, 'BoundaryCondition', 'symmetric', @(x) ischar(x) && ismember(x, {'zero', 'symmetric', 'replicate', 'circular'}));
    addParameter(p, 'BoundaryWidth', 5, @(x) isnumeric(x) && x >= 0);
    addParameter(p, 'PreserveBoundary', true, @islogical);
    addParameter(p, 'MaxDeltaK', 2, @(x) isnumeric(x) && x > 0);
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

%% thêm ngày 29-6-25
%% ngày 1-7-25


function [skeleton_image, binary_image] = skeletonize_zhang_suen(input_image, display_result)
% SKELETONIZE_ZHANG_SUEN Thực hiện skeletonization bằng thuật toán Zhang-Suen
%
% Hàm này sử dụng thuật toán Zhang-Suen để tạo khung xương (skeleton) từ ảnh đầu vào.
% Thuật toán Zhang-Suen là một phương pháp thinning song song, bảo toàn topology
% và tạo ra skeleton có độ dày 1 pixel.
%
% INPUT:
%   input_image    - Ảnh đầu vào (grayscale hoặc binary)
%   display_result - (Optional) true/false để hiển thị kết quả (default: true)
%
% OUTPUT:
%   skeleton_image - Ảnh skeleton (binary)
%   binary_image   - Ảnh binary trung gian sau bước nhị phân hóa
%
% EXAMPLE:
%   skeleton = skeletonize_zhang_suen(input_img);
%   [skeleton, binary] = skeletonize_zhang_suen(input_img, false);

% --- Xử lý tham số đầu vào ---
if nargin < 1
    error('Thiếu tham số đầu vào: input_image');
end

if nargin < 2
    display_result = true; % Mặc định hiển thị kết quả
end

% --- Kiểm tra đầu vào ---
if isempty(input_image)
    error('Ảnh đầu vào không được để trống');
end

if ~isnumeric(input_image)
    error('Ảnh đầu vào phải là ma trận số');
end

% Chuyển đổi sang ảnh xám nếu cần
if size(input_image, 3) == 3
    input_image = rgb2gray(input_image);
    fprintf('Đã chuyển đổi ảnh RGB sang grayscale\n');
end

try
    fprintf('Bắt đầu quá trình skeletonization...\n');

    % --- Bước 1: Nhị phân hóa ảnh bằng Otsu ---
    fprintf('Bước 1/3: Nhị phân hóa ảnh bằng phương pháp Otsu...\n');
    thresh = graythresh(input_image);
    BW_Original = imbinarize(input_image, thresh);

    fprintf('Ngưỡng Otsu: %.4f\n', thresh);
    fprintf('Số pixel foreground: %d\n', sum(BW_Original(:)));

    % --- Bước 2: Skeletonize bằng Zhang-Suen ---
    fprintf('Bước 2/3: Áp dụng thuật toán Zhang-Suen...\n');
    BW_Thinned = BW_Original;
    [rows, cols] = size(BW_Thinned);
    changing = true;
    iteration = 0;

    while changing
        iteration = iteration + 1;
        changing = false;
        BW_Del = true(rows, cols);

        % --- Step 1 của Zhang-Suen ---
        for i = 2:rows-1
            for j = 2:cols-1
                P = BW_Thinned(i-1:i+1, j-1:j+1);
                P = P(:)';
                % Sắp xếp theo thứ tự: P1(center), P2, P3, P4, P5, P6, P7, P8, P9, P2(lặp)
                P = [P(5), P(2), P(3), P(6), P(9), P(8), P(7), P(4), P(1), P(2)];

                if P(1) == 1  % Nếu pixel trung tâm là foreground
                    neighbors = sum(P(2:9));  % Số lượng neighbor foreground
                    transitions = sum(P(2:9) == 0 & P(3:10) == 1);  % Số transition 0->1

                    % Điều kiện Zhang-Suen Step 1
                    if neighbors >= 2 && neighbors <= 6 && transitions == 1 ...
                            && P(2)*P(4)*P(6) == 0 && P(4)*P(6)*P(8) == 0
                        BW_Del(i,j) = false;
                        changing = true;
                    end
                end
            end
        end
        BW_Thinned = BW_Thinned & BW_Del;

        % --- Step 2 của Zhang-Suen ---
        BW_Del = true(rows, cols);
        for i = 2:rows-1
            for j = 2:cols-1
                P = BW_Thinned(i-1:i+1, j-1:j+1);
                P = P(:)';
                P = [P(5), P(2), P(3), P(6), P(9), P(8), P(7), P(4), P(1), P(2)];

                if P(1) == 1
                    neighbors = sum(P(2:9));
                    transitions = sum(P(2:9) == 0 & P(3:10) == 1);

                    % Điều kiện Zhang-Suen Step 2
                    if neighbors >= 2 && neighbors <= 6 && transitions == 1 ...
                            && P(2)*P(4)*P(8) == 0 && P(2)*P(6)*P(8) == 0
                        BW_Del(i,j) = false;
                        changing = true;
                    end
                end
            end
        end
        BW_Thinned = BW_Thinned & BW_Del;

        % Hiển thị tiến trình mỗi 10 iterations
        if mod(iteration, 10) == 0
            fprintf('  Iteration %d: %d pixels còn lại\n', iteration, sum(BW_Thinned(:)));
        end

        % Tránh vòng lặp vô hạn
        if iteration > 1000
            warning('Đã đạt giới hạn iteration (1000). Dừng thuật toán.');
            break;
        end
    end

    fprintf('Hoàn thành sau %d iterations\n', iteration);
    fprintf('Số pixel skeleton: %d\n', sum(BW_Thinned(:)));

    % --- Bước 3: Hiển thị kết quả ---
    if display_result
        fprintf('Bước 3/3: Hiển thị kết quả...\n');

        figure('Name', 'Kết quả Skeletonization Zhang-Suen', 'NumberTitle', 'off');

        % Hiển thị so sánh
        subplot(1, 3, 1);
        imshow(input_image);
        title('Ảnh gốc', 'FontSize', 12);

        subplot(1, 3, 2);
        imshow(BW_Original);
        title('Ảnh nhị phân (Otsu)', 'FontSize', 12);

        subplot(1, 3, 3);
        imshow(BW_Thinned);
        title('Skeleton (Zhang-Suen)', 'FontSize', 12);

        % Điều chỉnh layout
        sgtitle('Quá trình Skeletonization', 'FontSize', 14, 'FontWeight', 'bold');
    end

    % --- Trả về kết quả ---
    skeleton_image = BW_Thinned;
    binary_image = BW_Original;

    % Thống kê cuối cùng
    fprintf('\n=== THỐNG KÊ KẾT QUẢ ===\n');
    fprintf('Kích thước ảnh: %d x %d\n', rows, cols);
    fprintf('Pixel gốc (foreground): %d (%.2f%%)\n', sum(BW_Original(:)), 100*sum(BW_Original(:))/(rows*cols));
    fprintf('Pixel skeleton: %d (%.2f%%)\n', sum(BW_Thinned(:)), 100*sum(BW_Thinned(:))/(rows*cols));
    fprintf('Tỷ lệ nén: %.2fx\n', sum(BW_Original(:))/sum(BW_Thinned(:)));
    fprintf('Số iterations: %d\n', iteration);
    fprintf('========================\n');

catch ME
    % Xử lý lỗi
    error_msg = sprintf('Lỗi trong quá trình skeletonize Zhang-Suen:\n%s\n\nChi tiết:\n%s', ...
        ME.message, ME.getReport());
    error(error_msg);
end

end

function [fringe_order, fringe_labels, processed_image] = assign_fringe_order(input_image, display_result)
% ASSIGN_FRINGE_ORDER Gán bậc vân cho ảnh hologram đã được skeletonize
%
% Hàm này thực hiện gán nhãn bậc vân dựa trên vị trí tương đối so với tâm ảnh.
% Vân gần tâm nhất được gán bậc 0, các vân phía trên có bậc dương tăng dần,
% các vân phía dưới có bậc âm giảm dần.
%
% INPUT:
%   input_image    - Ảnh binary đã được skeletonize
%   display_result - (Optional) true/false để hiển thị kết quả (default: true)
%
% OUTPUT:
%   fringe_order     - Số lượng vân được phát hiện
%   fringe_labels    - Vector chứa nhãn bậc vân của từng vùng liên thông
%   processed_image  - Ảnh đã được cắt biên và xử lý
%
% EXAMPLE:
%   [order, labels, img] = assign_fringe_order(skeleton_image);
%   [order, labels, img] = assign_fringe_order(skeleton_image, false); % Không hiển thị

% --- Xử lý tham số đầu vào ---
if nargin < 1
    error('Thiếu tham số đầu vào: input_image');
end

if nargin < 2
    display_result = true; % Mặc định hiển thị kết quả
end

% --- Kiểm tra đầu vào ---
if isempty(input_image)
    error('Ảnh đầu vào không được để trống');
end

if ~islogical(input_image) && ~(isnumeric(input_image) && all(input_image(:) == 0 | input_image(:) == 1))
    error('Ảnh đầu vào phải là ảnh binary (logical hoặc 0/1)');
end

% Chuyển đổi sang logical nếu cần
if ~islogical(input_image)
    input_image = logical(input_image);
end

try
    % --- Bước 1: Cắt biên ảnh để tránh ảnh hưởng vùng biên ---
    offset = 1;
    [orig_H, orig_W] = size(input_image);

    % Kiểm tra kích thước ảnh
    if orig_H <= 2*offset || orig_W <= 2*offset
        warning('Ảnh quá nhỏ để cắt biên. Sử dụng ảnh gốc.');
        bw_crop = input_image;
        offset = 0;
    else
        bw_crop = input_image(offset+1:end-offset, offset+1:end-offset);
    end

    [H, W] = size(bw_crop);

    % --- Bước 2: Tìm các vùng liên thông (vân) ---

    cc = bwconncomp(bw_crop);

    if cc.NumObjects == 0
        warning('Không tìm thấy vân nào trong ảnh');
        fringe_order = 0;
        fringe_labels = [];
        processed_image = bw_crop;
        return;
    end

    labeled_matrix = labelmatrix(cc);
    stats = regionprops(cc, 'Centroid', 'BoundingBox');

    % --- Bước 3: Tìm nhóm gần tâm nhất làm gốc ---
    centroids = cat(1, stats.Centroid);
    image_center = [W/2, H/2];
    dist = vecnorm(centroids - image_center, 2, 2);
    [~, idx_center] = min(dist);

    % --- Bước 4: Khởi tạo và gán nhãn ---
    labels = nan(cc.NumObjects, 1);
    labels(idx_center) = 0; % Nhóm gốc đặt nhãn 0

    queue = idx_center; % Hàng đợi để duyệt lan truyền nhãn
    processed_groups = false(cc.NumObjects, 1);
    processed_groups(idx_center) = true;

    % --- Bước 5: Lan truyền nhãn ---
    while ~isempty(queue)
        current_group = queue(1);
        queue(1) = [];

        current_label = labels(current_group);
        pixels = cc.PixelIdxList{current_group};
        [rows, cols] = ind2sub([H, W], pixels);

        visited_gid = []; % Tránh xét lại nhóm cùng vòng lặp

        for i = 1:length(rows)
            r = rows(i);
            c = cols(i);

            % Lan truyền lên trên theo cột
            for y = r-1:-1:1
                gid = labeled_matrix(y, c);
                if gid > 0 && ~processed_groups(gid) && ~ismember(gid, visited_gid)
                    labels(gid) = current_label + 1; % Nhãn tăng dần lên trên
                    queue(end+1) = gid;
                    processed_groups(gid) = true;
                    visited_gid(end+1) = gid;
                    break;
                elseif gid > 0 && processed_groups(gid)
                    break;
                end
            end

            % Lan truyền xuống dưới theo cột
            for y = r+1:H
                gid = labeled_matrix(y, c);
                if gid > 0 && ~processed_groups(gid) && ~ismember(gid, visited_gid)
                    labels(gid) = current_label - 1; % Nhãn giảm dần xuống dưới
                    queue(end+1) = gid;
                    processed_groups(gid) = true;
                    visited_gid(end+1) = gid;
                    break;
                elseif gid > 0 && processed_groups(gid)
                    break;
                end
            end
        end
    end

    % --- Bước 6: Chuẩn hóa nhãn thành số nguyên dương bắt đầu từ 1 ---
    valid_labels = labels(~isnan(labels));

    if isempty(valid_labels)
        warning('Không thể gán nhãn cho bất kỳ vân nào');
        fringe_order = 0;
        fringe_labels = [];
        processed_image = bw_crop;
        return;
    end

    unique_labels = unique(valid_labels);
    labels_new = nan(size(labels));
    for i = 1:length(unique_labels)
        labels_new(labels == unique_labels(i)) = i;
    end
    labels = labels_new;

    % --- Bước 7: Hiển thị kết quả (nếu được yêu cầu) ---
    if display_result
        figure('Name', 'Kết quả gán bậc vân', 'NumberTitle', 'off');
        imshow(bw_crop);
        hold on;

        for k = 1:cc.NumObjects
            if ~isnan(labels(k))
                pixels = cc.PixelIdxList{k};
                [rows, cols] = ind2sub([H, W], pixels);
                coords = [cols, rows]; % [x, y]

                % Tính khoảng cách từ tâm ảnh để đặt nhãn ở vị trí gần tâm nhất
                dists = sqrt((coords(:,1) - image_center(1)).^2 + (coords(:,2) - image_center(2)).^2);
                [~, min_idx] = min(dists);
                label_pos = coords(min_idx, :);

                text(label_pos(1), label_pos(2), num2str(labels(k)), ...
                    'Color', 'r', 'FontSize', 11, 'FontWeight', 'bold', ...
                    'HorizontalAlignment', 'center');
            end
        end

        title('Gán bậc vân', 'FontSize', 12);
        hold off;
    end

    % --- Bước 8: Trả về kết quả ---
    fringe_order = cc.NumObjects;
    fringe_labels = labels;
    processed_image = bw_crop;

    % Hiển thị thống kê
    fprintf('Đã phát hiện %d vân\n', fringe_order);
    fprintf('Số vân được gán nhãn: %d\n', sum(~isnan(labels)));
    if ~isempty(valid_labels)
        fprintf('Phạm vi bậc vân: %d đến %d\n', min(unique_labels), max(unique_labels));
    end

catch ME
    % Xử lý lỗi
    error_msg = sprintf('Lỗi trong quá trình gán bậc vân:\n%s', ME.message);
    error(error_msg);
end

end
function [fringe_order, fringe_labels, processed_image] = assign_fringe_order_improved(input_image, display_result)
% ASSIGN_FRINGE_ORDER_IMPROVED Gán bậc vân, cải tiến để xử lý vân dính liền.
%
% Hàm này sử dụng Watershed transform để tách các vân bị dính liền trước khi
% gán bậc vân. Điều này giúp nhận diện chính xác hơn các vân riêng lẻ.
%
% INPUT:
%   input_image    - Ảnh binary đã được skeletonize
%   display_result - (Optional) true/false để hiển thị kết quả (default: true)
%
% OUTPUT:
%   fringe_order     - Số lượng vân được phát hiện
%   fringe_labels    - Vector chứa nhãn bậc vân của từng vùng liên thông
%   processed_image  - Ảnh đã được xử lý (đã tách vân)

% --- Xử lý tham số đầu vào ---
if nargin < 1
    error('Thiếu tham số đầu vào: input_image');
end
if nargin < 2
    display_result = true; % Mặc định hiển thị kết quả
end

% --- Kiểm tra đầu vào ---
if isempty(input_image)
    error('Ảnh đầu vào không được để trống');
end
if ~islogical(input_image) && ~(isnumeric(input_image) && all(input_image(:) == 0 | input_image(:) == 1))
    error('Ảnh đầu vào phải là ảnh binary (logical hoặc 0/1)');
end
if ~islogical(input_image)
    input_image = logical(input_image);
end

try
    % --- Bước 1: Cắt biên ảnh ---
    offset = 1;
    [orig_H, orig_W] = size(input_image);
    if orig_H <= 2*offset || orig_W <= 2*offset
        warning('Ảnh quá nhỏ để cắt biên. Sử dụng ảnh gốc.');
        bw_crop = input_image;
        offset = 0;
    else
        bw_crop = input_image(offset+1:end-offset, offset+1:end-offset);
    end
    [H, W] = size(bw_crop);

    % --- *** CẢI TIẾN: BƯỚC TÁCH VÂN DÍNH LIỀN BẰNG WATERSHED *** ---
    % 1. Đảo ngược ảnh (vân thành 0, nền thành 1) và tính distance transform
    D = bwdist(~bw_crop);

    % 2. Áp dụng Watershed để tìm đường phân chia.
    %    - Phủ định D để các vùng trung tâm của nền trở thành "lưu vực".
    %    - Watershed sẽ tạo ra các "con đê" (giá trị 0) tại nơi các lưu vực gặp nhau.
    L = watershed(-D);

    % 3. Cắt các vân gốc tại đường phân chia của watershed.
    %    Những pixel có nhãn 0 trong L là đường phân chia.
    %    Đặt các pixel này thành 0 (đen) trong ảnh gốc để tách các vùng.
    bw_separated = bw_crop;
    bw_separated(L == 0) = 0;
    % --- *** KẾT THÚC PHẦN CẢI TIẾN *** ---


    % --- Bước 2: Tìm các vùng liên thông (trên ảnh đã được tách) ---
    cc = bwconncomp(bw_separated); % Sử dụng ảnh đã được tách vân
    if cc.NumObjects == 0
        warning('Không tìm thấy vân nào trong ảnh');
        fringe_order = 0;
        fringe_labels = [];
        processed_image = bw_separated;
        return;
    end
    labeled_matrix = labelmatrix(cc);
    stats = regionprops(cc, 'Centroid', 'BoundingBox');

    % --- Bước 3: Tìm nhóm gần tâm nhất làm gốc ---
    centroids = cat(1, stats.Centroid);
    image_center = [W/2, H/2];
    dist = vecnorm(centroids - image_center, 2, 2);
    [~, idx_center] = min(dist);

    % --- Bước 4: Khởi tạo và gán nhãn ---
    labels = nan(cc.NumObjects, 1);
    labels(idx_center) = 0;
    queue = idx_center;
    processed_groups = false(cc.NumObjects, 1);
    processed_groups(idx_center) = true;

    % --- Bước 5: Lan truyền nhãn (giữ nguyên logic) ---
    while ~isempty(queue)
        current_group = queue(1);
        queue(1) = [];
        current_label = labels(current_group);
        pixels = cc.PixelIdxList{current_group};
        [rows, cols] = ind2sub([H, W], pixels);
        visited_gid = [];
        for i = 1:length(rows)
            r = rows(i);
            c = cols(i);
            % Lan truyền lên trên
            for y = r-1:-1:1
                gid = labeled_matrix(y, c);
                if gid > 0 && ~processed_groups(gid) && ~ismember(gid, visited_gid)
                    labels(gid) = current_label + 1;
                    queue(end+1) = gid;
                    processed_groups(gid) = true;
                    visited_gid(end+1) = gid;
                    break;
                elseif gid > 0 && processed_groups(gid)
                    break;
                end
            end
            % Lan truyền xuống dưới
            for y = r+1:H
                gid = labeled_matrix(y, c);
                if gid > 0 && ~processed_groups(gid) && ~ismember(gid, visited_gid)
                    labels(gid) = current_label - 1;
                    queue(end+1) = gid;
                    processed_groups(gid) = true;
                    visited_gid(end+1) = gid;
                    break;
                elseif gid > 0 && processed_groups(gid)
                    break;
                end
            end
        end
    end

    % --- Bước 6: Chuẩn hóa nhãn thành số nguyên dương ---
    valid_labels = labels(~isnan(labels));
    if isempty(valid_labels)
        warning('Không thể gán nhãn cho bất kỳ vân nào');
        fringe_order = 0;
        fringe_labels = [];
        processed_image = bw_separated;
        return;
    end
    % Chuyển đổi bậc vân tương đối (vd: -2, -1, 0, 1, 2) thành bậc tuyệt đối (1, 2, 3, 4, 5)
    [~, ~, final_labels_indices] = unique(valid_labels);
    labels_new = nan(size(labels));
    labels_new(~isnan(labels)) = final_labels_indices;
    labels = labels_new;


    % --- Bước 7: Hiển thị kết quả ---
    if display_result
        figure('Name', 'Kết quả gán bậc vân (Cải tiến)', 'NumberTitle', 'off');
        imshow(bw_separated); % Hiển thị ảnh đã tách vân
        hold on;
        for k = 1:cc.NumObjects
            if ~isnan(labels(k))
                pixels = cc.PixelIdxList{k};
                [rows, cols] = ind2sub([H, W], pixels);
                coords = [cols, rows]; % [x, y]
                dists = sqrt((coords(:,1) - image_center(1)).^2 + (coords(:,2) - image_center(2)).^2);
                [~, min_idx] = min(dists);
                label_pos = coords(min_idx, :);
                text(label_pos(1), label_pos(2), num2str(labels(k)), ...
                    'Color', 'yellow', 'FontSize', 9, 'FontWeight', 'bold', ...
                    'HorizontalAlignment', 'center');
            end
        end
        title('Gán bậc vân sau khi dùng Watershed', 'FontSize', 12);
        hold off;
    end

    % --- Bước 8: Trả về kết quả ---
    fringe_order = cc.NumObjects;
    fringe_labels = labels;
    processed_image = bw_separated;

    fprintf('Đã phát hiện %d vân (sau khi tách)\n', fringe_order);
    fprintf('Số vân được gán nhãn: %d\n', sum(~isnan(labels)));
    if ~isempty(valid_labels)
        fprintf('Phạm vi bậc vân: %d đến %d\n', min(labels(~isnan(labels))), max(labels(~isnan(labels))));
    end
catch ME
    error_msg = sprintf('Lỗi trong quá trình gán bậc vân:\n%s', ME.message);
    error(error_msg);
end
end
function [recons_surface, figure_handle] = reconSurface_linearPushed(BW, fringe_labels, lambda, tilt_option, show_figure)
% RECONSURFACE_LINEARPUSHED Tái tạo bề mặt 3D từ ảnh vân giao thoa
%
% Cú pháp:
%   [recons_surface, figure_handle] = reconSurface_linearPushed(BW, fringe_labels, lambda, tilt_option, show_figure)
%
% Tham số đầu vào:
%   BW            - Ảnh nhị phân đã cắt biên (logical matrix)
%   fringe_labels - Vector chứa nhãn của các vân (double array)
%   lambda        - Bước sóng ánh sáng (double)
%   tilt_option   - Tùy chọn xử lý ('None', 'Remove tilt', 'Invert', 'Remove + Invert')
%   show_figure   - Có hiển thị figure hay không (logical, optional, default: true)
%
% Tham số đầu ra:
%   recons_surface - Ma trận bề mặt 3D đã tái tạo
%   figure_handle  - Handle của figure (nếu show_figure = true)
%
% Ví dụ:
%   [surface, fig] = reconSurface_linearPushed(BW, [1,2,3,4,5], 632.8e-9, 'Remove tilt');

% Xử lý tham số đầu vào
if nargin < 5
    show_figure = true;
end

% Kiểm tra tham số đầu vào
if isempty(fringe_labels)
    error('Bạn cần gán nhãn vân trước khi nội suy.');
end

if ~islogical(BW)
    error('BW phải là ảnh nhị phân (logical matrix).');
end

% Thiết lập khoảng cách giữa các vân
khoang_cach_van = lambda/2;

% Tìm các thành phần liên thông
cc = bwconncomp(BW);
L = labelmatrix(cc);

% Khởi tạo các mảng điểm 3D
num_labels = max(L(:));
X = []; Y = []; Z = [];

for i = 1:num_labels
    % Tìm các điểm thuộc vân có nhãn i
    [y, x] = find(L == i);

    if i <= length(fringe_labels)
        % Tính độ cao z dựa trên nhãn vân
        z = ones(size(x)) * (fringe_labels(i) - 1) * khoang_cach_van;
        X = [X; x];
        Y = [Y; y];
        Z = [Z; z];
    end
end


% ==========================================================
% ===== CHÈN ĐOẠN CODE HIỂN THỊ VÀO ĐÂY ======================
figure('Name', 'Bề mặt trước khi nội suy (Point Cloud)');
scatter3(X, Y, Z, 10, Z, 'filled');
title('Đám mây điểm 3D trước khi nội suy');
xlabel('X (px)');
ylabel('Y (px)');
zlabel('Độ cao (m)');
colorbar;
axis ij; % Lật trục Y để khớp với tọa độ ảnh
view(-30, 20); % Đặt góc nhìn cho dễ quan sát
% ==========================================================

% Kiểm tra xem có dữ liệu để nội suy không
if isempty(X)
    error('Không có dữ liệu để nội suy. Kiểm tra lại fringe_labels và BW.');
end

% Nội suy để tạo mặt 3D mượt
[xq, yq] = meshgrid(1:size(BW,2), 1:size(BW,1));
F = scatteredInterpolant(X, Y, Z, 'natural', 'nearest');
Zq = F(xq, yq);
Zq(~isfinite(Zq)) = 0;

% %
% Z_grid_cubic = griddata(X, Y, Z, xq, yq, 'cubic');
% Z_grid_cubic(~isfinite(Z_grid_cubic)) = 0;
% 
% % 6. Làm mượt hậu xử lý cho cubic
% Z_cubic_smooth = imgaussfilt(Z_grid_cubic, 2);
% Zq = Z_cubic_smooth;
% %

% Chuyển từ mét sang radian
phi_rad = (4 * pi / lambda) * Zq;
Zq = phi_rad;

% Cắt biên để hiển thị tốt hơn
margin = 1;
if size(Zq,1) > 2*margin && size(Zq,2) > 2*margin
    Z_crop = Zq(margin:end-margin, margin:end-margin);
else
    Z_crop = Zq;
    warning('Kích thước ảnh quá nhỏ để cắt biên.');
end

[M, N] = size(Z_crop);
[xGrid, yGrid] = meshgrid(1:N, 1:M);
x = xGrid(:);
y = yGrid(:);
z = Z_crop(:);

% Xử lý theo lựa chọn của người dùng
switch tilt_option
    case 'None'
        Z_processed = Z_crop;

    case 'Remove tilt'
        good = ~isnan(z);
        if sum(good) < 3
            warning('Không đủ điểm hợp lệ để loại bỏ độ nghiêng.');
            Z_processed = Z_crop;
        else
            A = [x, y, ones(size(x))];
            coeff = A(good,:) \ z(good);
            Z_fit = reshape(A * coeff, size(Z_crop));
            Z_processed = Z_crop - Z_fit;
        end

    case 'Invert'
        Z_processed = max(Z_crop(:)) - Z_crop;

    case 'Remove + Invert'
        good = ~isnan(z);
        if sum(good) < 3
            warning('Không đủ điểm hợp lệ để loại bỏ độ nghiêng.');
            Z_leveled = Z_crop;
        else
            A = [x, y, ones(size(x))];
            coeff = A(good,:) \ z(good);
            Z_fit = reshape(A * coeff, size(Z_crop));
            Z_leveled = Z_crop - Z_fit;
        end
        Z_processed = max(Z_leveled(:)) - Z_leveled;

    otherwise
        warning('Tùy chọn không hợp lệ. Sử dụng "None".');
        Z_processed = Z_crop;
end

% Chuẩn hóa bắt đầu từ 0
Z_offset = Z_processed - min(Z_processed(:));

% Gán kết quả đầu ra
recons_surface = Z_offset;

% Hiển thị bề mặt 3D nếu được yêu cầu
if show_figure
    figure_handle = figure;
    surf(xGrid, yGrid, Z_offset);
    shading interp;
    xlabel('X (px)');
    ylabel('Y (px)');
    zlabel('rad');
    title(['3D Surface Linear (Option: ', tilt_option, ')']);
    colormap parula;
    colorbar;
else
    figure_handle = [];
end

end

function refined = xoa_ria(binaryImg)
% refineSkeleton - Tinh chỉnh ảnh skeleton nhị phân
% Đầu vào:
%   binaryImg - ảnh nhị phân đã skeleton hóa
% Đầu ra:
%   refined - ảnh skeleton sau khi loại nhiễu và nối các đoạn đứt

% --- Bước 1: Loại bỏ các nhánh nhỏ (râu ria)
cleaned = bwmorph(binaryImg, 'spur', 1);  % loại bỏ các nhánh nhỏ lẻ

% --- Bước 3: Lấy lại skeleton sau khi closing
skeleton = bwmorph(cleaned, 'skel', 3);    % skeleton hóa lại

% --- Bước 4: Loại bỏ tiếp râu ria còn lại (nếu có)
pruned = bwmorph(skeleton, 'spur', 1);      % chỉ xóa các spur nhỏ nhất

% --- Bước 5: Xoá các điểm trắng đơn lẻ (nhiễu nhỏ)
refined = bwareaopen(pruned, 2);            % giữ lại các vùng >= 2 pixels
end
%%
function [wrappedPhase, params] = reconstruct_phase_auto(hologram, params)
% Tái tạo pha từ hologram bằng cách lọc trong miền tần số với lựa chọn tự động.
%
% Chức năng sẽ tự động tìm phổ bậc +1 ở nửa trên của miền tần số,
% tạo một bộ lọc (tròn hoặc HCN) và tiến hành tái tạo pha.
%
% Tham số (params) có thể chứa:
% params.filter_type: 'circle' hoặc 'rectangle' (mặc định: 'circle')
% params.filter_radius: Bán kính của bộ lọc tròn (mặc định: 40)
% params.filter_width: Chiều rộng bộ lọc HCN (mặc định: 80)
% params.filter_height: Chiều cao bộ lọc HCN (mặc định: 80)
% params.dc_suppression_radius: Bán kính để loại bỏ thành phần DC (mặc định: 25)

% --- Kiểm tra và đặt giá trị mặc định cho params ---
if ~exist('params', 'var')
    params = struct();
end
if ~isfield(params, 'filter_type')
    params.filter_type = 'circle'; % 'circle' hoặc 'rectangle'
end
if ~isfield(params, 'filter_radius')
    params.filter_radius = 50; % Bán kính của bộ lọc tròn
end
if ~isfield(params, 'filter_width')
    params.filter_width = 100; % Chiều rộng bộ lọc HCN
end
if ~isfield(params, 'filter_height')
    params.filter_height = 100; % Chiều cao bộ lọc HCN
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

% Loại bỏ thành phần DC (bậc 0) để tránh chọn nhầm
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

% Vẽ hình dạng bộ lọc tương ứng
if strcmp(params.filter_type, 'circle')
    theta = 0:0.01:2*pi;
    x_circle = params.filter_radius * cos(theta) + u_max;
    y_circle = params.filter_radius * sin(theta) + v_max;
    plot(x_circle, y_circle, 'g', 'LineWidth', 2);
    title(['Phổ bậc +1 (Tròn) tại (', num2str(u_max), ', ', num2str(v_max), ')']);
else % rectangle
    rect_x = u_max - params.filter_width/2;
    rect_y = v_max - params.filter_height/2;
    rectangle('Position', [rect_x, rect_y, params.filter_width, params.filter_height], ...
        'EdgeColor', 'g', 'LineWidth', 2);
    title(['Phổ bậc +1 (HCN) tại (', num2str(u_max), ', ', num2str(v_max), ')']);
end
hold off;

% --- Tạo bộ lọc và trích xuất phổ ---

% Tạo mask tương ứng với loại bộ lọc
if strcmp(params.filter_type, 'circle')
    % Bộ lọc hình tròn
    roi_mask = sqrt((U - u_max).^2 + (V - v_max).^2) <= params.filter_radius;
else
    % Bộ lọc hình chữ nhật
    roi_mask = (abs(U - u_max) <= params.filter_width/2) & ...
        (abs(V - v_max) <= params.filter_height/2);
end

% Áp dụng mask để chỉ giữ lại phổ bậc +1
filteredContent = fourierTransform .* roi_mask;

% % Dịch chuyển vùng phổ đã chọn về lại tâm của ma trận
% dich_chuyen = 0;
% % Tính toán độ dịch chuyển cần thiết
% v_shift = v0 - v_max;
% u_shift = u0 - u_max - dich_chuyen;
% 
% % Dùng circshift để dịch chuyển
% filteredSpectrum = circshift(filteredContent, [v_shift, u_shift]);

filteredSpectrum = filteredContent;
% --- Hiển thị kết quả phổ sau khi lọc và dịch chuyển ---
figure('Name','Phổ sau khi xử lý');
imshow(log(1 + abs(filteredSpectrum)), []);
title(['Phổ bậc +1 (', params.filter_type, ') sau khi lọc và dịch về tâm']);

% --- Tái tạo trường sóng phức và lấy pha ---
finalPhaseComplex = ifft2(ifftshift(filteredSpectrum));

% Lấy pha từ trường phức
wrappedPhase = angle(finalPhaseComplex);
end


%% theem 6/7/25 - thêm đa thức zernike

function hologram = generate_test_hologram(M, N, fx, fy, phase_object, noise_level)
% Tạo ra một hologram nhiễu xạ Fresnel đơn giản.
%
% Input:
%   M, N: Kích thước của hologram
%   fx, fy: Tần số sóng mang theo hai chiều x và y
%   phase_object: Ma trận 2D đại diện cho pha của vật thể
%
% Output:
%   hologram: Ma trận 2D của hologram được tạo ra

[X, Y] = meshgrid(1:N, 1:M);
[x, sigma] = meshgrid(linspace(0, 1, N), linspace(0, pi/5, N));

% Define the zero-mean Gaussian noise component
% This is a random term for each point, with a standard deviation of sigma
I_noise = noise_level * randn(N, N) .* sigma;

% Cường độ nền và điều biến
a = 1.0; % Background intensity
b = 0.8; % Modulation depth

% Sóng mang phẳng (plane wave carrier)
carrier = 2 * pi * (fx * X + fy * Y);

% Công thức tạo hologram
% g = a + b * cos(sóng_mang + pha_vật)
hologram = a + b .* cos(carrier + phase_object) + I_noise;

end
%%
function surface = reconstruct_wavefront(zernike_coeffs, order, grid_size)
    % Hàm tái tạo bề mặt sóng từ các hệ số đa thức Zernike
    % zernike_coeffs: Hệ số đa thức Zernike
    % order: Bậc của các đa thức Zernike
    % grid_size: Kích thước của lưới điểm trên mặt phẳng 

    % Khởi tạo mặt phẳng lưới
    [X, Y] = meshgrid(linspace(-1, 1, grid_size), linspace(-1, 1, grid_size));
    R = sqrt(X.^2 + Y.^2);
    Theta = atan2(Y, X);

    % Khởi tạo bề mặt sóng
    surface = zeros(size(X));

    % Lặp qua các bậc và hệ số để tái tạo bề mặt sóng
    index = 1;
    for n = 0:order
        for m = -n:2:n
            if index <= length(zernike_coeffs)
                % Lấy hệ số tương ứng
                Z = zernike_coeffs(index);
                
                % Tính giá trị Zernike polynomial
                Zmn = ZernikePolynomial(n, m, R, Theta);
                
                % Cộng dồn vào bề mặt sóng
              %  surface = surface + Z * Zmn;
                surface = surface + Z * Zmn;

                index = index + 1;
            end
        end
    end
end

function Zmn = ZernikePolynomial(n, m, R, Theta)
    % Hàm tính Zernike polynomial
    % n: Bậc n của Zernike
    % m: Bậc m của Zernike
    % R: Bán kính
    % Theta: Góc theta
    
    % Tính radial Zernike polynomial
    RadialPoly = zeros(size(R));
    for k = 0:floor((n-abs(m))/2)
        c = (-1)^k * factorial(n-k) / (factorial(k) * factorial((n + abs(m))/2 - k) * factorial((n - abs(m))/2 - k));
        RadialPoly = RadialPoly + c * R.^(n - 2*k);
    end
    
    % Tính Zernike polynomial
    if m >= 0
        Zmn = RadialPoly .* cos(m * Theta);
    else
        Zmn = RadialPoly .* sin(abs(m) * Theta);
    end
end

%% Tính toán đa thức Zernike
function radial = zernike_radial(r,n,m)
    % Functions required for use: elliptical_crop
%     hàm tính toán đa thức Zernike
%     Đầu vào
%         r:      bán kính
%         n:      bậc quang sai
%         m:      bậc phương vị
%     Đầu ra:
%         giá trị Đa thức Zernike

    if mod(n-m,2) == 1
        error('n-m must be even');
    end
    if n < 0 || m < 0
        error('n and m must both be positive in radial function')
    end
    if floor(n) ~= n || floor(m) ~= m
        error('n and m must both be integers')
    end
    if n == m
        radial = r.^n;
    elseif n - m == 2
        radial = n*zernike_radial(r,n,n)-(n-1)*zernike_radial(r,n-2,n-2);
    else
        H3 = (-4*((m+4)-2)*((m+4)-3)) / ((n+(m+4)-2)*(n-(m+4)+4));
        H2 = (H3*(n+(m+4))*(n-(m+4)+2)) / (4*((m+4)-1))  +  ((m+4)-2);
        H1 = ((m+4)*((m+4)-1) / 2)  -  (m+4)*H2  +  (H3*(n+(m+4)+2)*(n-(m+4))) / (8);
        radial = H1*zernike_radial(r,n,m+4) + (H2+H3 ./ r.^2).*zernike_radial(r,n,m+2);
        
        % Fill in NaN values that may have resulted from DIV/0 in prior
        % line. Evaluate these points directly (non-recursively) as they
        % are scarce if present.
        
        if sum(sum(isnan(radial))) > 0
            [row, col] = find(isnan(radial));
            c=1;
            while c<=length(row)
                x = 0;
                for k = 0:(n-m)/2
                    ((-1)^k*factorial(n-k))/(factorial(k)*factorial((n+m)/2-k)*factorial((n-m)/2-k))*0^(n-2*k);
                    x = x + ((-1)^k*factorial(n-k))/(factorial(k)*factorial((n+m)/2-k)*factorial((n-m)/2-k))*0^(n-2*k);
                end
                radial(row(c),col(c)) = x;
                c=c+1;
            end
        end

    end

end
%%
function [output_coeff_no_tilt, z_recon_no_tilt] = removeTiltFromZernike(output_coeff, index_type, z_map, center_j, center_i)
% Hàm loại bỏ nghiêng từ hệ số Zernike đã fit
%
% Inputs:
%   output_coeff - Hệ số từ ZernikeLegendreFit
%   index_type - "2indices" hoặc "fringe"
%   z_map - Bản đồ surface gốc (để tái tạo)
%   center_j, center_i - Tọa độ center
%
% Outputs:
%   output_coeff_no_tilt - Hệ số đã loại bỏ nghiêng
%   z_recon_no_tilt - Bề mặt tái tạo không nghiêng

if nargin < 5
    [center_j, center_i] = FindCenter(z_map);
end

if lower(index_type) == "2indices"
    % Với 2indices: nghiêng là m=1, n=0
    % amn(2,1) = a_10 (nghiêng theo x)
    % bmn(2,1) = b_10 (nghiêng theo y)
    
    amn = output_coeff{1};
    bmn = output_coeff{2};
    
    % Tạo copy và loại bỏ nghiêng
    amn_no_tilt = amn;
    bmn_no_tilt = bmn;
    
    % Loại bỏ tilt terms (m=1, n=0)
    amn_no_tilt(2, 1) = 0;  % a_10 = 0
    bmn_no_tilt(2, 1) = 0;  % b_10 = 0
    
    output_coeff_no_tilt = cell(1,2);
    output_coeff_no_tilt{1} = amn_no_tilt;
    output_coeff_no_tilt{2} = bmn_no_tilt;
    
elseif lower(index_type) == "fringe"
    % Với fringe indexing: 
    % j=2 -> x-tilt, j=3 -> y-tilt
    
    coeff_j = output_coeff{1};
    coeff_j_no_tilt = coeff_j;
    
    % Loại bỏ tilt terms
    if length(coeff_j) >= 2
        coeff_j_no_tilt(2) = 0;  % x-tilt
    end
    if length(coeff_j) >= 3
        coeff_j_no_tilt(3) = 0;  % y-tilt
    end
    
    output_coeff_no_tilt = cell(1,1);
    output_coeff_no_tilt{1} = coeff_j_no_tilt;
end

% Tái tạo bề mặt từ hệ số đã loại bỏ nghiêng
z_recon_no_tilt = reconstructZernikeSurface(output_coeff_no_tilt, index_type, z_map, center_j, center_i);

end

function z_recon = reconstructZernikeSurface(output_coeff, index_type, z_map, center_j, center_i)
% Hàm tái tạo bề mặt từ hệ số Zernike

[x_pix, y_pix] = size(z_map);
r_pix = x_pix/2 * 0.99;

[i_mesh, j_mesh] = meshgrid((1:x_pix), (1:y_pix));
[theta_map, rho_map] = cart2pol(i_mesh-center_i, j_mesh-center_j);
u_map = rho_map(:)/r_pix;
u2_map = u_map.^2;
% u2_map(u2_map > 0.99) = NaN;

if lower(index_type) == "2indices"
    amn = output_coeff{1};
    bmn = output_coeff{2};
    
    [m_max, n_max] = size(amn);
    m_max = m_max - 1;
    n_max = n_max - 1;
    
    z_recon = zeros(x_pix, y_pix);
    
    for m = 0:m_max
        % Tính Jacobi polynomials
        Pmns_map = jacobiZernike_table(n_max+1, m, reshape(u2_map, 1, [])');
        Pmns_map = Pmns_map' .* (reshape(u_map, 1, []).^m);
        
        for n = 0:n_max
            if m + 2*n <= n_max % Điều kiện hợp lệ
                Pmn_map = reshape(Pmns_map(n+1, :), x_pix, y_pix);
                
                % Thêm contribution từ amn và bmn
                a = amn(m+1, n+1);
                b = bmn(m+1, n+1);
                
                if m == 0
                    z_recon = z_recon + a * Pmn_map;
                else
                    a_map = a * cos(m * theta_map) .* Pmn_map;
                    b_map = b * sin(m * theta_map) .* Pmn_map;
                    z_recon = z_recon + a_map + b_map;
                end
            end
        end
    end
    
elseif lower(index_type) == "fringe"
    coeff_j = output_coeff{1};
    j_max = length(coeff_j);
    
    z_recon = zeros(x_pix, y_pix);
    
    for j = 1:j_max
        if coeff_j(j) ~= 0
            [n, m] = fringe22index(j);
            
            % Tính Jacobi polynomial
            Pmn_map = jacobiZernike_table(n+1, abs(m), reshape(u2_map, 1, [])');
            Pmn_map = reshape(Pmn_map(n+1, :), x_pix, y_pix);
            Pmn_map = Pmn_map .* (rho_map/r_pix).^abs(m);
            
            if m == 0
                z_recon = z_recon + coeff_j(j) * Pmn_map;
            elseif m > 0
                z_recon = z_recon + coeff_j(j) * cos(m * theta_map) .* Pmn_map;
            else % m < 0
                z_recon = z_recon + coeff_j(j) * sin(abs(m) * theta_map) .* Pmn_map;
            end
        end
    end
end

% Mask các vùng ngoài aperture
% z_recon(u_map > 0.99) = NaN;

end

%% Các hàm hỗ trợ (copy từ code gốc)
function [avg_i, avg_j] = FindCenter(Z)
i_index = 1:size(Z,1); 
j_index = 1:size(Z,2);

[valid_i, valid_j] = find(~isnan(Z));
avg_i = mean(i_index(valid_i),'all');
avg_j = mean(j_index(valid_j),'all');
end

function JZ=jacobiZernike_table(k,m,xe)
JZ = zeros(size(xe,1),k+1);

JZ(:,1)=ones(size(xe,1),1); % n=0
if k > 0
    JZ(:,2)=(m+2)*xe-(m+1);     % n=1
end

for n = 1:k-1
    s = m+2*n;
    an = -(s+1)*((s-n).^2+n.^2+s)./(n+1)./(s-n+1)./s;
    bn = (s+2)*(s+1)./(n+1)./(s-n+1);
    cn = (s+2)*(s-n)*n./(n+1)./(s-n+1)./s;
    JZ(:,n+2)=(an+bn*xe).*JZ(:,n+1)-cn*JZ(:,n);
end
end

function [n_, m_] = fringe22index(j)
for n = 0:1:j*2+2
    for m = -n:1:n+1
        if m < 0
            sgn_m = -1;
        else
            sgn_m = 1;
        end
        temp = (1+(n+abs(m))/2)^2 - 2*abs(m) + (1-sgn_m)/2;
        if j == temp
            m_ = m;
            n_ = (n-abs(m_))/2;
            return
        end
    end
end
end
%% thêm 9-7-25
%% thêm 9-7-25
function varargout = crop_multiple_to_smallest(varargin)
    % Giả định tất cả các biến là 2D ma trận
    n = nargin;
    sizes = cellfun(@size, varargin, 'UniformOutput', false);

    % Tìm kích thước nhỏ nhất theo từng chiều
    min_rows = min(cellfun(@(s) s(1), sizes));
    min_cols = min(cellfun(@(s) s(2), sizes));

    varargout = cell(1, n);
    for i = 1:n
        mat = varargin{i};
        [m, n_] = size(mat);
        
        % Tính chỉ số cắt đều 4 phía
        row_start = floor((m - min_rows)/2) + 1;
        col_start = floor((n_ - min_cols)/2) + 1;
        row_end = row_start + min_rows - 1;
        col_end = col_start + min_cols - 1;
        
        varargout{i} = mat(row_start:row_end, col_start:col_end);
    end
end


function [phase_detrended, tilt_plane] = remove_tilt_simple(phase_map, cutoff_ratio)
% remove_tilt_simple: Loại bỏ tilt từ ảnh pha bằng cách ước lượng từ tần số thấp
%
% INPUT:
%   phase_map - ảnh pha đầu vào (2D)
%   cutoff_ratio - tỉ lệ tần số thấp giữ lại (default: 0.05)
%
% OUTPUT:
%   phase_detrended - pha sau khi loại bỏ tilt
%   tilt_plane - mặt phẳng tilt đã ước lượng

if nargin < 2
    cutoff_ratio = 0.05;
end

[rows, cols] = size(phase_map);

% Biến đổi Fourier
F = fftshift(fft2(double(phase_map)));

% Tạo filter Gaussian (mượt hơn circular filter)
[X, Y] = meshgrid(1:cols, 1:rows);
cx = ceil(cols/2);
cy = ceil(rows/2);
R = sqrt((X - cx).^2 + (Y - cy).^2);
Rmax = max(cx, cy);

% Gaussian filter thay vì circular
sigma = cutoff_ratio * Rmax / 2;
mask = exp(-(R.^2) / (2 * sigma^2));

% Lọc tần số thấp
F_low = F .* mask;
low_freq_phase = real(ifft2(ifftshift(F_low)));

% Fit mặt phẳng từ low-frequency component
[Xg, Yg] = meshgrid(1:cols, 1:rows);
A = [Xg(:), Yg(:), ones(rows*cols, 1)];
coeffs = A \ low_freq_phase(:);

% Tạo mặt phẳng tilt
tilt_plane = coeffs(1)*Xg + coeffs(2)*Yg + coeffs(3);

% Loại bỏ tilt
phase_detrended = phase_map - tilt_plane;

end
%% - theem 10-7-25

function skeleton = mzs_thinning(binary_image)
    % MZS (Modified ZS) Thinning Algorithm
    % Input: binary_image - binary image (0 for background, 1 for foreground)
    % Output: skeleton - thinned skeleton of the input image
    
    % Convert to logical if needed
    if ~islogical(binary_image)
        binary_image = logical(binary_image);
    end
    
    % Initialize
    skeleton = binary_image;
    [rows, cols] = size(skeleton);
    
    % Main thinning loop
    changed = true;
    iteration = 0;
    
    while changed
        iteration = iteration + 1;
        fprintf('Iteration %d\n', iteration);
        
        changed = false;
        
        % First sub-iteration
        candidates1 = find_deletion_candidates(skeleton, 1);
        if ~isempty(candidates1)
            for i = 1:size(candidates1, 1)
                row = candidates1(i, 1);
                col = candidates1(i, 2);
                skeleton(row, col) = 0;
                changed = true;
            end
        end
        
        % Second sub-iteration
        candidates2 = find_deletion_candidates(skeleton, 2);
        if ~isempty(candidates2)
            for i = 1:size(candidates2, 1)
                row = candidates2(i, 1);
                col = candidates2(i, 2);
                skeleton(row, col) = 0;
                changed = true;
            end
        end
        
        % Safety check to prevent infinite loop
        if iteration > 1000
            warning('Maximum iterations reached. Stopping thinning process.');
            break;
        end
    end
    
    fprintf('Thinning completed after %d iterations\n', iteration);
end

function candidates = find_deletion_candidates(image, subiteration)
    % Find pixels that are candidates for deletion
    
    [rows, cols] = size(image);
    candidates = [];
    
    % Process each pixel
    for i = 2:rows-1
        for j = 2:cols-1
            % Check if current pixel is foreground
            if image(i, j) == 1
                % Check subfield condition (even pixels only)
                if mod(i + j, 2) == 0
                    % Extract 3x3 neighborhood
                    neighborhood = image(i-1:i+1, j-1:j+1);
                    
                    % Check deletion conditions
                    if check_deletion_conditions(neighborhood, subiteration)
                        candidates = [candidates; i, j];
                    end
                end
            end
        end
    end
end

function can_delete = check_deletion_conditions(neighborhood, subiteration)
    % Check if a pixel can be deleted based on MZS conditions
    
    % Map neighborhood to paper notation
    % p9 p2 p3
    % p8 p1 p4
    % p7 p6 p5
    
    p1 = neighborhood(2, 2);  % Center pixel
    p2 = neighborhood(1, 2);  % North
    p3 = neighborhood(1, 3);  % Northeast
    p4 = neighborhood(2, 3);  % East
    p5 = neighborhood(3, 3);  % Southeast
    p6 = neighborhood(3, 2);  % South
    p7 = neighborhood(3, 1);  % Southwest
    p8 = neighborhood(2, 1);  % West
    p9 = neighborhood(1, 1);  % Northwest
    
    % Calculate B(p1) - number of foreground pixels in 8-neighborhood
    B_p1 = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
    
    % Calculate C(p1) - number of 8-connected components
    C_p1 = calculate_connectivity(p2, p3, p4, p5, p6, p7, p8, p9);
    
    % Common conditions for both sub-iterations
    condition_b = (C_p1 == 1);  % Exactly one 8-connected component
    
    if subiteration == 1
        % First sub-iteration conditions
        condition_c = (B_p1 >= 2 && B_p1 <= 7);
        condition_d = (p2 * p4 * p6 == 0);
        condition_e = (p4 * p6 * p8 == 0);
        
        can_delete = condition_b && condition_c && condition_d && condition_e;
        
    else  % subiteration == 2
        % Second sub-iteration conditions
        condition_c = (B_p1 >= 1 && B_p1 <= 7);
        condition_d = (p2 * p4 * p8 == 0);
        condition_e = (p2 * p6 * p8 == 0);
        
        can_delete = condition_b && condition_c && condition_d && condition_e;
        
        % Additional condition to preserve 2x2 squares
        if can_delete && B_p1 == 1
            % Find the single black neighbor
            neighbors = [p2, p3, p4, p5, p6, p7, p8, p9];
            neighbor_positions = [
                -1, 0;   % p2
                -1, 1;   % p3
                0, 1;    % p4
                1, 1;    % p5
                1, 0;    % p6
                1, -1;   % p7
                0, -1;   % p8
                -1, -1   % p9
            ];
            
            % Check if the single neighbor is a diagonal pixel
            diagonal_indices = [2, 4, 6, 8];  % p3, p5, p7, p9
            
            for k = 1:length(neighbors)
                if neighbors(k) == 1
                    if ismember(k, diagonal_indices)
                        % This is a diagonal neighbor
                        % Apply additional preservation condition
                        % (This would require checking the neighbor's neighborhood)
                        % For simplicity, we'll preserve such pixels
                        can_delete = false;
                    end
                    break;
                end
            end
        end
    end
end

function C = calculate_connectivity(p2, p3, p4, p5, p6, p7, p8, p9)
    % Calculate number of 8-connected components using the formula from paper
    % C(p1) = ~p2 & (p3 | p4) + ~p4 & (p5 | p6) + ~p6 & (p7 | p8) + ~p8 & (p9 | p2)
    
    term1 = (~p2) & (p3 | p4);
    term2 = (~p4) & (p5 | p6);
    term3 = (~p6) & (p7 | p8);
    term4 = (~p8) & (p9 | p2);
    
    C = term1 + term2 + term3 + term4;
end

% Demo function to test the algorithm
function demo_mzs()
    % Create test images
    
    % Test 1: Simple square
    fprintf('Testing MZS on 4x4 square pattern:\n');
    square = [
        0 0 0 0 0 0;
        0 1 1 1 1 0;
        0 1 1 1 1 0;
        0 1 1 1 1 0;
        0 1 1 1 1 0;
        0 0 0 0 0 0
    ];
    
    figure(1);
    subplot(1,2,1);
    imshow(square, 'InitialMagnification', 500);
    title('Original Square');
    
    skeleton_square = mzs_thinning(square);
    subplot(1,2,2);
    imshow(skeleton_square, 'InitialMagnification', 500);
    title('MZS Thinned Square');
    
    % Test 2: L-shaped pattern
    fprintf('\nTesting MZS on L-shaped pattern:\n');
    L_shape = [
        0 0 0 0 0 0 0 0;
        0 1 1 1 1 1 1 0;
        0 1 1 1 1 1 1 0;
        0 1 1 0 0 0 0 0;
        0 1 1 0 0 0 0 0;
        0 1 1 0 0 0 0 0;
        0 1 1 1 1 1 1 0;
        0 1 1 1 1 1 1 0;
        0 0 0 0 0 0 0 0
    ];
    
    figure(2);
    subplot(1,2,1);
    imshow(L_shape, 'InitialMagnification', 300);
    title('Original L-shape');
    
    skeleton_L = mzs_thinning(L_shape);
    subplot(1,2,2);
    imshow(skeleton_L, 'InitialMagnification', 300);
    title('MZS Thinned L-shape');
    
    % Test 3: Diagonal line
    fprintf('\nTesting MZS on diagonal line:\n');
    diagonal = zeros(10, 10);
    for i = 1:8
        diagonal(i, i) = 1;
        diagonal(i, i+1) = 1;
        diagonal(i+1, i) = 1;
    end
    
    figure(3);
    subplot(1,2,1);
    imshow(diagonal, 'InitialMagnification', 300);
    title('Original Diagonal');
    
    skeleton_diagonal = mzs_thinning(diagonal);
    subplot(1,2,2);
    imshow(skeleton_diagonal, 'InitialMagnification', 300);
    title('MZS Thinned Diagonal');
    
    % Print statistics
    fprintf('\nStatistics:\n');
    fprintf('Square: %d -> %d pixels\n', sum(square(:)), sum(skeleton_square(:)));
    fprintf('L-shape: %d -> %d pixels\n', sum(L_shape(:)), sum(skeleton_L(:)));
    fprintf('Diagonal: %d -> %d pixels\n', sum(diagonal(:)), sum(skeleton_diagonal(:)));
end

% Performance evaluation functions
function evaluate_performance(original, skeleton)
    % Calculate performance metrics as defined in the paper
    
    % Thinning Rate (TR)
    TR = calculate_thinning_rate(skeleton);
    
    % Connectivity Measure (CM)
    CM = calculate_connectivity_measure(skeleton);
    
    % Sensitivity Measure (SM)
    SM = calculate_sensitivity_measure(skeleton);
    
    % Execution time would be measured during actual execution
    
    fprintf('Performance Metrics:\n');
    fprintf('Thinning Rate (TR): %.6f\n', TR);
    fprintf('Connectivity Measure (CM): %d\n', CM);
    fprintf('Sensitivity Measure (SM): %d\n', SM);
    fprintf('Original pixels: %d\n', sum(original(:)));
    fprintf('Skeleton pixels: %d\n', sum(skeleton(:)));
end

function TR = calculate_thinning_rate(skeleton)
    % Calculate thinning rate based on triangle count
    [rows, cols] = size(skeleton);
    
    TM1 = 0;
    for i = 1:rows
        for j = 1:cols
            if skeleton(i, j) == 1
                TM1 = TM1 + count_triangles(skeleton, i, j);
            end
        end
    end
    
    TM2 = 4 * (max(rows, cols) - 1)^2;
    TR = 1 - TM1 / TM2;
end

function triangle_count = count_triangles(image, i, j)
    % Count triangles as defined in the paper
    [rows, cols] = size(image);
    
    triangle_count = 0;
    
    if i > 1 && j > 1 && i < rows && j < cols
        % Get neighborhood
        p1 = image(i, j);
        p2 = image(i-1, j);
        p3 = image(i-1, j+1);
        p4 = image(i, j+1);
        p5 = image(i+1, j+1);
        p6 = image(i+1, j);
        p7 = image(i+1, j-1);
        p8 = image(i, j-1);
        p9 = image(i-1, j-1);
        
        % Count triangles
        triangle_count = p1 * ((p8 * p9) + (p9 * p2) + (p2 * p3) + (p3 * p4));
    end
end

function CM = calculate_connectivity_measure(skeleton)
    % Count end points and discrete points
    [rows, cols] = size(skeleton);
    CM = 0;
    
    for i = 2:rows-1
        for j = 2:cols-1
            if skeleton(i, j) == 1
                % Count neighbors
                neighborhood = skeleton(i-1:i+1, j-1:j+1);
                B = sum(neighborhood(:)) - 1;  % Exclude center pixel
                
                if B < 2
                    CM = CM + 1;
                end
            end
        end
    end
end

function SM = calculate_sensitivity_measure(skeleton)
    % Count cross-points (pixels with more than 2 connections)
    [rows, cols] = size(skeleton);
    SM = 0;
    
    for i = 2:rows-1
        for j = 2:cols-1
            if skeleton(i, j) == 1
                % Calculate A(p) - number of transitions
                neighborhood = skeleton(i-1:i+1, j-1:j+1);
                p = [neighborhood(1,2), neighborhood(1,3), neighborhood(2,3), ...
                     neighborhood(3,3), neighborhood(3,2), neighborhood(3,1), ...
                     neighborhood(2,1), neighborhood(1,1), neighborhood(1,2)];
                
                A = 0;
                for k = 1:8
                    if p(k) ~= p(k+1)
                        A = A + 1;
                    end
                end
                A = A / 2;
                
                if A > 2
                    SM = SM + 1;
                end
            end
        end
    end
end

%%
function phase_corrected = iterative_median_unwrap(phase_wrapped, varargin)
%ITERATIVE_MEDIAN_UNWRAP Sử dụng lọc trung vị lặp để sửa lỗi 2π
%
% Inputs:
%   phase_wrapped - Ma trận pha đã wrapped (rad)
%   varargin - Các tham số tùy chọn:
%              'window_size' - Kích thước cửa sổ lọc (default: 3)
%              'max_iter' - Số lần lặp tối đa (default: 10)
%              'threshold' - Ngưỡng dừng (default: 0.01)
%              'verbose' - Hiển thị thông tin (default: false)
%
% Output:
%   phase_corrected - Ma trận pha đã sửa lỗi 2π

% Xử lý tham số đầu vào
p = inputParser;
addParameter(p, 'window_size', 5, @(x) isnumeric(x) && x > 0);
addParameter(p, 'max_iter', 200, @(x) isnumeric(x) && x > 0);
addParameter(p, 'threshold', 0.0000001, @(x) isnumeric(x) && x > 0);
addParameter(p, 'verbose', false, @islogical);
parse(p, varargin{:});

window_size = p.Results.window_size;
max_iter = p.Results.max_iter;
threshold = p.Results.threshold;
verbose = p.Results.verbose;

% Khởi tạo
phase_corrected = phase_wrapped;
[rows, cols] = size(phase_wrapped);

if verbose
    fprintf('Bắt đầu thuật toán lọc trung vị lặp...\n');
    fprintf('Kích thước ma trận: %d x %d\n', rows, cols);
end

% Vòng lặp chính
for iter = 1:max_iter
    phase_old = phase_corrected;
    
    % Tính gradient của pha
    [grad_x, grad_y] = gradient(phase_corrected);
    
    % Phát hiện các điểm có gradient lớn (nghi ngờ có lỗi 2π)
    grad_magnitude = sqrt(grad_x.^2 + grad_y.^2);
    suspect_mask = grad_magnitude > pi; % Ngưỡng phát hiện lỗi 2π
    
    % Áp dụng lọc trung vị cho các vùng nghi ngờ
    phase_filtered = medfilt2(phase_corrected, [window_size, window_size]);
    
    % Tính sự khác biệt giữa pha gốc và pha sau lọc
    diff_phase = phase_corrected - phase_filtered;
    
    % Xác định lỗi 2π và sửa chữa
    for i = 1:rows
        for j = 1:cols
            if suspect_mask(i, j)
                % Tính số lần 2π cần sửa
                k = round(diff_phase(i, j) / (2*pi));
                if abs(k) > 0
                    phase_corrected(i, j) = phase_corrected(i, j) - k * 2*pi;
                end
            end
        end
    end
    
    % Kiểm tra điều kiện dừng
    change = norm(phase_corrected - phase_old, 'fro') / norm(phase_old, 'fro');
    
    if verbose
        fprintf('Lần lặp %d: Thay đổi = %.6f\n', iter, change);
    end
    
    if change < threshold
        if verbose
            fprintf('Hội tụ sau %d lần lặp\n', iter);
        end
        break;
    end
end

if verbose && iter == max_iter
    fprintf('Đã đạt số lần lặp tối đa (%d)\n', max_iter);
end

end
function phase_corrected = iterative_k_correction(phase_wrapped, varargin)
%ITERATIVE_K_CORRECTION Sử dụng lọc trung vị lặp để sửa các bước nhảy k bất thường
%
% Inputs:
%   phase_wrapped - Ma trận pha đã wrapped (rad)
%   varargin - Các tham số tùy chọn:
%              'window_size' - Kích thước cửa sổ lọc (default: 3)
%              'max_iter' - Số lần lặp tối đa (default: 10)
%              'threshold' - Ngưỡng dừng (default: 0.01)
%              'k_threshold' - Ngưỡng phát hiện k bất thường (default: 2)
%              'verbose' - Hiển thị thông tin (default: false)
%
% Output:
%   phase_corrected - Ma trận pha đã sửa lỗi 2π

% Xử lý tham số đầu vào
p = inputParser;
addParameter(p, 'window_size', 3, @(x) isnumeric(x) && x > 0);
addParameter(p, 'max_iter', 100, @(x) isnumeric(x) && x > 0);
addParameter(p, 'threshold', 0.01, @(x) isnumeric(x) && x > 0);
addParameter(p, 'k_threshold', 2, @(x) isnumeric(x) && x > 0);
addParameter(p, 'verbose', false, @islogical);
parse(p, varargin{:});

window_size = p.Results.window_size;
max_iter = p.Results.max_iter;
threshold = p.Results.threshold;
k_threshold = p.Results.k_threshold;
verbose = p.Results.verbose;

% Khởi tạo
phase_corrected = phase_wrapped;
[rows, cols] = size(phase_wrapped);

if verbose
    fprintf('Bắt đầu thuật toán sửa k bất thường...\n');
    fprintf('Kích thước ma trận: %d x %d\n', rows, cols);
end

% Vòng lặp chính
for iter = 1:max_iter
    phase_old = phase_corrected;
    
    % Bước 1: Tính ma trận k từ pha hiện tại
    k_matrix = compute_k_matrix(phase_corrected, verbose && iter == 1);
    
    % Bước 2: Áp dụng lọc trung vị lên ma trận k
    k_filtered = medfilt2(k_matrix, [window_size, window_size]);
    
    % Bước 3: Phát hiện các k bất thường
    k_diff = abs(k_matrix - k_filtered);
    abnormal_mask = k_diff > k_threshold;
    
    % Bước 4: Sửa các k bất thường
    k_corrected = k_matrix;
    k_corrected(abnormal_mask) = k_filtered(abnormal_mask);
    
    % Bước 5: Tái tạo pha từ k đã sửa
    phase_corrected = reconstruct_phase_from_k(phase_wrapped, k_corrected);
    
    % Kiểm tra điều kiện dừng
    change = norm(phase_corrected - phase_old, 'fro') / norm(phase_old, 'fro');
    
    if verbose
        num_corrected = sum(abnormal_mask(:));
        fprintf('Lần lặp %d: Sửa %d điểm k, Thay đổi = %.6f\n', ...
            iter, num_corrected, change);
    end
    
    if change < threshold
        if verbose
            fprintf('Hội tụ sau %d lần lặp\n', iter);
        end
        break;
    end
end

if verbose && iter == max_iter
    fprintf('Đã đạt số lần lặp tối đa (%d)\n', max_iter);
end

end

function k_matrix = compute_k_matrix(phase, verbose)
%COMPUTE_K_MATRIX Tính ma trận các bước nhảy k
    [rows, cols] = size(phase);
    k_matrix = zeros(rows, cols);
    
    % Tính k theo hướng ngang (x)
    for i = 1:rows
        for j = 2:cols
            phase_diff = phase(i, j) - phase(i, j-1);
            k_matrix(i, j) = k_matrix(i, j-1) + round(phase_diff / (2*pi));
        end
    end
    
    % Tính k theo hướng dọc (y) và kết hợp
    k_matrix_y = zeros(rows, cols);
    for j = 1:cols
        for i = 2:rows
            phase_diff = phase(i, j) - phase(i-1, j);
            k_matrix_y(i, j) = k_matrix_y(i-1, j) + round(phase_diff / (2*pi));
        end
    end
    
    % Kết hợp k từ cả hai hướng (trung bình có trọng số)
    weight_x = 0.5;
    weight_y = 0.5;
    k_matrix = weight_x * k_matrix + weight_y * k_matrix_y;
    k_matrix = round(k_matrix);
    
    if verbose
        fprintf('Thống kê ma trận k:\n');
        fprintf('- Giá trị k min/max: %d / %d\n', min(k_matrix(:)), max(k_matrix(:)));
        fprintf('- Số điểm k != 0: %d (%.2f%%)\n', ...
            sum(k_matrix(:) ~= 0), 100*sum(k_matrix(:) ~= 0)/numel(k_matrix));
    end
end

function phase_unwrapped = reconstruct_phase_from_k(phase_wrapped, k_matrix)
%RECONSTRUCT_PHASE_FROM_K Tái tạo pha từ ma trận k
    phase_unwrapped = phase_wrapped + 2*pi * k_matrix;
end


function [k_x, k_y] = compute_k_gradients(phase)
%COMPUTE_K_GRADIENTS Tính gradient của k theo cả hai hướng
    [rows, cols] = size(phase);
    k_x = zeros(rows, cols);
    k_y = zeros(rows, cols);
    
    % Tính k_x (gradient theo hướng x)
    for i = 1:rows
        for j = 2:cols
            phase_diff = phase(i, j) - phase(i, j-1);
            k_x(i, j) = round(phase_diff / (2*pi));
        end
    end
    
    % Tính k_y (gradient theo hướng y)
    for j = 1:cols
        for i = 2:rows
            phase_diff = phase(i, j) - phase(i-1, j);
            k_y(i, j) = round(phase_diff / (2*pi));
        end
    end
end

% Hàm hỗ trợ: Tạo dữ liệu test với lỗi k cụ thể
function phase_test = generate_test_phase_with_k_errors(rows, cols)
%GENERATE_TEST_PHASE_WITH_K_ERRORS Tạo dữ liệu pha test với lỗi k
    [X, Y] = meshgrid(1:cols, 1:rows);
    
    % Tạo pha liên tục mượt
    phase_continuous = 0.3 * X + 0.2 * Y + 0.5 * sin(0.1*X) .* cos(0.1*Y);
    
    % Wrap pha về khoảng [-π, π]
    phase_wrapped = angle(exp(1i * phase_continuous));
    
    % Thêm lỗi k cụ thể (tạo các bước nhảy sai)
    error_locations = [
        round(rows*0.3), round(cols*0.3);
        round(rows*0.7), round(cols*0.4);
        round(rows*0.5), round(cols*0.8);
    ];
    
    phase_test = phase_wrapped;
    for i = 1:size(error_locations, 1)
        r = error_locations(i, 1);
        c = error_locations(i, 2);
        if r <= rows && c <= cols
            % Thêm lỗi ±2π hoặc ±4π
            error_k = randi([-2, 2]);
            if error_k ~= 0
                phase_test(r, c) = phase_test(r, c) + error_k * 2*pi;
                phase_test(r, c) = angle(exp(1i * phase_test(r, c)));
            end
        end
    end
end

%% Thêm ngày 11-7-25
function [corrected_unwrapped_phase, num_iterations, convergence_history] = correct_sparse_artifacts_iterative_v2(unwrapped_phase_input, varargin)
% Hàm cải tiến V2: Sử dụng lọc trung vị trên delta_k để đảm bảo tính nhất quán cục bộ,
% thay thế cho hàm 'apply_spatial_continuity_constraint' phức tạp.
%
% Inputs:
%   unwrapped_phase_input - Ma trận pha unwrapped đầu vào
%   varargin - Các tham số tùy chọn (tương tự phiên bản gốc)
%
% Outputs:
%   corrected_unwrapped_phase - Pha đã được hiệu chỉnh
%   num_iterations - Số lần lặp thực tế
%   convergence_history - Lịch sử hội tụ (RMS của delta_k)

% --- (Phần xử lý tham số đầu vào và khởi tạo giữ nguyên như hàm gốc) ---
p = inputParser;
addParameter(p, 'FilterSize', [5 5], @(x) isnumeric(x) && length(x) == 2);
addParameter(p, 'Epsilon', 1e-6, @(x) isnumeric(x) && x > 0);
addParameter(p, 'MaxIterations', 150, @(x) isnumeric(x) && x > 0);
addParameter(p, 'Verbose', false, @islogical);
addParameter(p, 'BoundaryCondition', 'symmetric', @(x) ischar(x) && ismember(x, {'zero', 'symmetric', 'replicate', 'circular'}));
addParameter(p, 'BoundaryWidth', 5, @(x) isnumeric(x) && x >= 0);
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
[rows, cols] = size(unwrapped_phase_input);
current_phase = unwrapped_phase_input;
original_phase = unwrapped_phase_input;
convergence_history = [];
num_iterations = 0;
previous_delta_k = [];
if preserve_boundary && boundary_width > 0
    boundary_mask = create_boundary_mask(rows, cols, boundary_width);
else
    boundary_mask = false(rows, cols);
end
if isempty(mask_invalid)
    mask_invalid = false(rows, cols);
else
    if ~isequal(size(mask_invalid), [rows, cols])
        error('MaskInvalid phải có cùng kích thước với unwrapped_phase_input');
    end
end
protection_mask = boundary_mask | mask_invalid;

if verbose
    fprintf('Bắt đầu quá trình hiệu chỉnh lặp V2 (lọc delta_k)...\n');
    fprintf('Image size: %dx%d\n', rows, cols);
    fprintf('Filter size: [%d %d], Epsilon: %.2e, Max iterations: %d\n', ...
        filter_size(1), filter_size(2), epsilon, max_iterations);
end

% Vòng lặp chính
for iter = 1:max_iterations
    % Bước 1: Áp dụng bộ lọc trung vị để tìm pha tham chiếu
    % (Sử dụng padding 'symmetric' trực tiếp trong medfilt2 để đơn giản hóa)
    filtered_phase = medfilt2(current_phase, filter_size, 'symmetric');
    
    % Bước 2: Tính toán sự khác biệt về "thứ tự vân" (delta_k)
    delta_k = round((filtered_phase - current_phase) / (2*pi));
    
    % Bước 3: Áp dụng các ràng buộc
    % Giới hạn |delta_k|
    delta_k = sign(delta_k) .* min(abs(delta_k), max_delta_k);
    
    % Bảo vệ vùng biên và các pixel không hợp lệ
    delta_k(protection_mask) = 0;
    
    %*********************************************************************
    %** THAY ĐỔI CỐT LÕI: Lọc delta_k để đảm bảo tính nhất quán cục bộ  **
    %** Thay thế hàm 'apply_spatial_continuity_constraint' phức tạp.     **
    %** Sử dụng bộ lọc [3 3] là đủ để loại bỏ các hiệu chỉnh đơn lẻ.     **
    %*********************************************************************
    delta_k = medfilt2(delta_k, [3 3], 'symmetric');

    % Tính toán metric hội tụ
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
        fprintf('Iteration %d: RMS(delta_k) = %.6f, Corrections: %d\n', ...
            iter, rms_delta_k, num_corrections);
    end
    
    % Kiểm tra điều kiện hội tụ (tương tự hàm gốc)
    if isequal(delta_k, previous_delta_k) || rms_delta_k < epsilon
        if verbose
            fprintf('Hội tụ đạt được tại vòng lặp %d.\n', iter);
        end
        break;
    end
    
    % Bước 4: Hiệu chỉnh pha
    current_phase = current_phase + delta_k * (2*pi);
    
    % Khôi phục giá trị biên gốc nếu cần (Hàm gốc đã có, giữ lại là tốt)
    if preserve_boundary
        current_phase(protection_mask) = original_phase(protection_mask);
    end
    
    previous_delta_k = delta_k;
    
    if iter == max_iterations
        if verbose
            fprintf('Cảnh báo: Đạt số lần lặp tối đa (%d).\n', max_iterations);
        end
    end
end
corrected_unwrapped_phase = current_phase;
if verbose
    fprintf('Hoàn thành sau %d lần lặp. RMS cuối cùng của delta_k: %.6f\n', num_iterations, convergence_history(end));
end

% Các hàm hỗ trợ (giữ nguyên)
    function boundary_mask = create_boundary_mask(rows, cols, width)
        boundary_mask = false(rows, cols);
        if width > 0
            boundary_mask(1:width, :) = true;
            boundary_mask(end-width+1:end, :) = true;
            boundary_mask(:, 1:width) = true;
            boundary_mask(:, end-width+1:end) = true;
        end
    end
end

%% mới thêm 12-7-25
function [refined_phase, artifact_mask] = refine_sparse_artifacts(unwrapped_phase, varargin)
% REFINE_SPARSE_ARTIFACTS - Làm mượt pha sau khi unwrap bằng cách xử lý điểm nhiễu rải rác
%
% Inputs:
%   unwrapped_phase - Ma trận pha đã unwrap (có thể chứa nhiễu)
%
% Tùy chọn (varargin):
%   'WindowSize'    - Kích thước cửa sổ lọc median (default: 3)
%   'Threshold'     - Ngưỡng phát hiện dị thường (gradient) (default: 2*pi)
%   'MaxIterations' - Số lần lặp refine (default: 3)
%   'Verbose'       - Hiển thị thông tin (default: false)
%
% Outputs:
%   refined_phase   - Pha đã được làm mượt
%   artifact_mask   - Mask nhị phân các điểm bị coi là artifact

    % ==== Tham số mặc định ====
    window_size = 3;
    threshold = 2*pi;  % khoảng nhảy >= 2pi được coi là nhiễu
    max_iter = 3;
    verbose = false;

    % ==== Đọc các tùy chọn ====
    for k = 1:2:length(varargin)
        switch lower(varargin{k})
            case 'windowsize', window_size = varargin{k+1};
            case 'threshold', threshold = varargin{k+1};
            case 'maxiterations', max_iter = varargin{k+1};
            case 'verbose', verbose = varargin{k+1};
        end
    end

    refined_phase = unwrapped_phase;
    artifact_mask = false(size(unwrapped_phase));
    
    if verbose
        fprintf('--- Refining sparse artifacts ---\n');
    end

    % ==== Lặp refine nhiều lần ====
    for iter = 1:max_iter
        % Tính gradient theo x, y
        [gx, gy] = gradient(refined_phase);
        grad_mag = sqrt(gx.^2 + gy.^2);

        % Xác định điểm nghi ngờ có gradient bất thường
        current_artifact_mask = grad_mag > threshold;

        % Loại bỏ viền (vì không lọc được)
        current_artifact_mask([1 end], :) = false;
        current_artifact_mask(:, [1 end]) = false;

        % Gộp mask
        artifact_mask = artifact_mask | current_artifact_mask;

        % Áp dụng lọc median
        filtered_phase = medfilt2(refined_phase, [window_size window_size], 'symmetric');

        % Thay giá trị tại các điểm bị nghi ngờ bằng giá trị đã lọc
        refined_phase(current_artifact_mask) = filtered_phase(current_artifact_mask);

        if verbose
            fprintf('Iteration %d: %d pixels refined\n', iter, sum(current_artifact_mask(:)));
        end

        % Nếu không còn điểm nào cần refine thì kết thúc sớm
        if sum(current_artifact_mask(:)) == 0
            break;
        end
    end

    if verbose
        fprintf('Refine done. Total refined pixels: %d\n', sum(artifact_mask(:)));
    end
end

%% thêm hàm params 15-7-25
function params = set_default_params(params)

    if ~exist('params', 'var') || isempty(params)
        params = struct();
    end

    def = struct(...
        'filter_type', 'circle', ...
        'filter_radius', 50, ...
        'filter_width', 100, ...
        'filter_height', 100, ...
        'dc_suppression_radius', 25, ...
        'lambda', 632.8e-9, ...
        'pixel_size', 3.45e-6, ...
        'unwrap_method', 'least_square', ...
        'phase_smoothing', true, ...
        'smoothing_sigma', 2, ...
        'show_figures', true, ...
        'verbose', true ...
    );

    fn = fieldnames(def);
    for i = 1:length(fn)
        if ~isfield(params, fn{i})
            params.(fn{i}) = def.(fn{i});
        end
    end

end

function W_est = reconstruct_surface_from_fringe_order(fringe_order, lambda)
% Tái tạo bề mặt W(x,y) từ bản đồ bậc vân (fringe_order)
% - fringe_order: ma trận chứa bậc vân tại các điểm skeleton (0 ở nơi không có vân)
% - lambda: bước sóng ánh sáng (μm)
% - W_est: bề mặt ước lượng được nội suy toàn ảnh

    % Tạo W_sparse = fringe_order * lambda/2
    W_sparse = fringe_order * (lambda / 2);

    % Lấy tọa độ và giá trị tại các điểm có bậc vân
    [Ys, Xs] = find(fringe_order > 0);
    Zs = W_sparse(sub2ind(size(W_sparse), Ys, Xs));

    % Tạo hàm nội suy
    F = scatteredInterpolant(Xs, Ys, Zs, 'natural', 'nearest');

    % Nội suy toàn bộ bề mặt
    [Xgrid, Ygrid] = meshgrid(1:size(fringe_order, 2), 1:size(fringe_order, 1));
    W_est = F(Xgrid, Ygrid);
end
%% thêm ngày 19-7-25
function [recons_surface, figure_handle] = reconSurface_row(BW, fringe_labels, lambda, tilt_option, show_figure)
% RECONSURFACE_LINEARPUSHED Tái tạo bề mặt 3D từ ảnh vân giao thoa
% === PHIÊN BẢN SỬ DỤNG NỘI SUY 1D THEO TỪNG HÀNG (THỬ NGHIỆM) ===

% Xử lý tham số đầu vào
if nargin < 5
    show_figure = true;
end

% Kiểm tra tham số đầu vào
if isempty(fringe_labels)
    error('Bạn cần gán nhãn vân trước khi nội suy.');
end
if ~islogical(BW)
    error('BW phải là ảnh nhị phân (logical matrix).');
end

% Thiết lập khoảng cách giữa các vân
khoang_cach_van = lambda / 2;

% Tìm các thành phần liên thông
cc = bwconncomp(BW);
L = labelmatrix(cc);

% Khởi tạo các mảng điểm 3D (sử dụng phiên bản đã tối ưu)
num_labels = max(L(:));
num_pixels_to_process = 0;
for i = 1:min(num_labels, length(fringe_labels))
    num_pixels_to_process = num_pixels_to_process + length(cc.PixelIdxList{i});
end
X = zeros(num_pixels_to_process, 1);
Y = zeros(num_pixels_to_process, 1);
Z = zeros(num_pixels_to_process, 1);
currentIndex = 1;

for i = 1:num_labels
    if i <= length(fringe_labels)
        [y_coords, x_coords] = find(L == i);
        num_pts_in_fringe = length(x_coords);
        if num_pts_in_fringe > 0
            z_value = (fringe_labels(i) - 1) * khoang_cach_van;
            endIndex = currentIndex + num_pts_in_fringe - 1;
            X(currentIndex:endIndex) = x_coords;
            Y(currentIndex:endIndex) = y_coords;
            Z(currentIndex:endIndex) = z_value;
            currentIndex = currentIndex + num_pts_in_fringe;
        end
    end
end

% Hiển thị Point Cloud (tùy chọn)
% figure('Name', 'Bề mặt trước khi nội suy (Point Cloud)');
% scatter3(X, Y, Z, 10, Z, 'filled'); ...

% Kiểm tra xem có dữ liệu để nội suy không
if isempty(X)
    error('Không có dữ liệu để nội suy. Kiểm tra lại fringe_labels và BW.');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% THAY THẾ PHẦN NỘI SUY 2D BẰNG NỘI SUY 1D THEO TỪNG HÀNG %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
height = size(BW, 1);
width = size(BW, 2);
Zq = nan(height, width); % Khởi tạo ma trận kết quả với NaN
x_query = 1:width;       % Các điểm x cần nội suy trên mỗi hàng

for y = 1:height
    % Tìm dữ liệu (x, z) trên hàng y hiện tại
    indices_on_row = find(Y == y);
    if ~isempty(indices_on_row)
        x_coords = X(indices_on_row);
        z_values = Z(indices_on_row);
        
        % Xử lý các điểm x trùng lặp bằng cách lấy trung bình
        [unique_x, ~, ic] = unique(x_coords);
        unique_z = accumarray(ic, z_values, [], @mean);

        % Cần ít nhất 2 điểm để nội suy tuyến tính
        if length(unique_x) >= 2
            % Nội suy 1D và ngoại suy để lấp đầy hàng
            interpolated_row = interp1(unique_x, unique_z, x_query, 'linear', 'extrap');
            Zq(y, :) = interpolated_row;
        end
    end
end
% Dọn dẹp các hàng không thể nội suy được (vẫn còn NaN)
Zq(~isfinite(Zq)) = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% KẾT THÚC PHẦN THAY THẾ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Chuyển từ mét sang radian (giữ lại logic gốc nếu bạn muốn)
phi_rad = (4 * pi / lambda) * Zq;
Zq = phi_rad;

% Cắt biên để hiển thị tốt hơn
margin = 1;
if size(Zq,1) > 2*margin && size(Zq,2) > 2*margin
    Z_crop = Zq(margin:end-margin, margin:end-margin);
else
    Z_crop = Zq;
    warning('Kích thước ảnh quá nhỏ để cắt biên.');
end
[M, N] = size(Z_crop);
[xGrid, yGrid] = meshgrid(1:N, 1:M);
x = xGrid(:);
y = yGrid(:);
z = Z_crop(:);

% Xử lý theo lựa chọn của người dùng
switch tilt_option
    case 'None'
        Z_processed = Z_crop;
    case 'Remove tilt'
        good = ~isnan(z);
        if sum(good) < 3
            warning('Không đủ điểm hợp lệ để loại bỏ độ nghiêng.');
            Z_processed = Z_crop;
        else
            A = [x, y, ones(size(x))];
            coeff = A(good,:) \ z(good);
            Z_fit = reshape(A * coeff, size(Z_crop));
            Z_processed = Z_crop - Z_fit;
        end
    case 'Invert'
        Z_processed = max(Z_crop(:)) - Z_crop;
    case 'Remove + Invert'
        good = ~isnan(z);
        if sum(good) < 3
            warning('Không đủ điểm hợp lệ để loại bỏ độ nghiêng.');
            Z_leveled = Z_crop;
        else
            A = [x, y, ones(size(x))];
            coeff = A(good,:) \ z(good);
            Z_fit = reshape(A * coeff, size(Z_crop));
            Z_leveled = Z_crop - Z_fit;
        end
        Z_processed = max(Z_leveled(:)) - Z_leveled;
    otherwise
        warning('Tùy chọn không hợp lệ. Sử dụng "None".');
        Z_processed = Z_crop;
end

% Chuẩn hóa bắt đầu từ 0
Z_offset = Z_processed - min(Z_processed(:));
% Gán kết quả đầu ra
recons_surface = Z_offset;
% Hiển thị bề mặt 3D nếu được yêu cầu
if show_figure
    figure_handle = figure;
    surf(xGrid, yGrid, Z_offset);
    shading interp;
    xlabel('X (px)');
    ylabel('Y (px)');
    zlabel('rad'); % Giữ nguyên vì logic chuyển đổi sang radian vẫn còn
    title(['3D Surface (Row-by-Row Interp., Option: ', tilt_option, ')']);
    colormap parula;
    colorbar;
else
    figure_handle = [];
end

end

%% thêm ngày 21-7-25 
% hàm tính và hiển thị sai số
function calculateAndCompareErrors(ground_truth, varargin)
% calculateAndCompareErrors - Tính sai số của một hoặc nhiều bề mặt 3D so với ground truth.
%
% Cú pháp:
%   calculateAndCompareErrors(ground_truth, result_1, result_2, ..., result_N)
%
% Mô tả:
%   Hàm này nhận đối số đầu tiên là 'ground_truth' (bề mặt 3D thực tế).
%   Các đối số tiếp theo (result_1, result_2, ...) là các bề mặt 3D kết quả
%   cần được đánh giá.
%
%   Hàm sẽ lặp qua từng bề mặt kết quả, tính toán và hiển thị:
%   - Sai số Bình phương Trung bình gốc (RMSE) so với ground_truth.
%   - Sai số tuyệt đối cho từng điểm dữ liệu tương ứng.
%   - Hiển thị trên biểu đồ 3D và 2D trực quan

% --- Kiểm tra số lượng đầu vào ---
if nargin < 2
    error('Lỗi: Cần ít nhất hai đầu vào (ground_truth và một bộ kết quả).');
end

disp('=============================================');
disp('    BÁO CÁO SO SÁNH SAI SỐ CÁC BỀ MẶT 3D');
disp('=============================================');

% Khởi tạo dữ liệu để vẽ biểu đồ
num_results = length(varargin);
rmse_values = zeros(1, num_results);
mae_values = zeros(1, num_results);
max_error_values = zeros(1, num_results);
result_names = cell(1, num_results);
valid_results = [];
all_absolute_errors = cell(1, num_results);

% Màu sắc cho biểu đồ
colors = lines(num_results);

% Lấy kích thước của ground truth
[m, n] = size(ground_truth);

% Lặp qua từng bộ kết quả được cung cấp trong varargin
for k = 1:num_results
    final_result = varargin{k};
    
    % In tiêu đề cho mỗi lần so sánh
    fprintf('\n-----------------------------------------\n');
    fprintf('###   PHÂN TÍCH BỀ MẶT 3D SỐ %d   ###\n', k);
    fprintf('-----------------------------------------\n');
    
    % --- Kiểm tra kích thước cho từng bộ kết quả ---
    if ~isequal(size(ground_truth), size(final_result))
        fprintf('!!! CẢNH BÁO: Bỏ qua Bề mặt %d do không cùng kích thước với ground_truth.\n', k);
        fprintf('    Ground truth: [%d x %d], Bề mặt %d: [%d x %d]\n', ...
                size(ground_truth, 1), size(ground_truth, 2), k, size(final_result, 1), size(final_result, 2));
        rmse_values(k) = NaN;
        mae_values(k) = NaN;
        max_error_values(k) = NaN;
        result_names{k} = sprintf('Bề mặt %d (Lỗi)', k);
        continue;
    end
    
    % --- Tính toán Sai số ---
    
    % 1. Sai số Tuyệt đối
    absolute_error = abs(ground_truth - final_result);
    mae = mean(absolute_error(:));
    max_abs_error = max(absolute_error(:));
    
    % 2. Sai số Bình phương Trung bình gốc (RMSE)
    squared_error = (ground_truth - final_result).^2;
    mean_squared_error = mean(squared_error(:));
    rms_error = sqrt(mean_squared_error);
    
    % Lưu dữ liệu để vẽ biểu đồ
    rmse_values(k) = rms_error;
    mae_values(k) = mae;
    max_error_values(k) = max_abs_error;
    result_names{k} = sprintf('Bề mặt %d', k);
    valid_results = [valid_results, k];
    all_absolute_errors{k} = absolute_error;
    
    % --- Hiển thị Kết quả cho bộ dữ liệu hiện tại ---
    fprintf('=> Kích thước bề mặt: [%d x %d] = %d điểm\n', m, n, m*n);
    fprintf('=> Sai số Bình phương Trung bình gốc (RMSE): %.6f\n', rms_error);
    fprintf('=> Sai số Tuyệt đối Trung bình (MAE): %.6f\n', mae);
    fprintf('=> Sai số Tuyệt đối Lớn nhất: %.6f\n', max_abs_error);
    fprintf('=> Phạm vi giá trị Ground Truth: [%.3f, %.3f]\n', min(ground_truth(:)), max(ground_truth(:)));
    fprintf('=> Phạm vi giá trị Bề mặt %d: [%.3f, %.3f]\n', k, min(final_result(:)), max(final_result(:)));
    
    % Hiển thị một số điểm sai số chi tiết
    fprintf('\n=> Sai số tại một số điểm đặc trưng:\n');
    sample_points = [1, 1; ceil(m/2), ceil(n/2); m, n; ceil(m/4), ceil(n/4); ceil(3*m/4), ceil(3*n/4)];
    for i = 1:size(sample_points, 1)
        row = sample_points(i, 1);
        col = sample_points(i, 2);
        if row <= m && col <= n
            fprintf('   - Điểm (%d,%d): |%.4f - %.4f| = %.4f\n', ...
                    row, col, ground_truth(row, col), final_result(row, col), absolute_error(row, col));
        end
    end
end

disp(' ');
disp('=============================================');
disp('             KẾT THÚC BÁO CÁO');
disp('=============================================');

% --- VẼ BIỂU ĐỒ ---
if ~isempty(valid_results)
    % Tạo figure chính với kích thước lớn hơn cho bề mặt 3D
    fig = figure('Position', [50, 50, 1400, 900]);
    
    % Tạo lưới tọa độ X, Y
    [X, Y] = meshgrid(1:n, 1:m);
    
    % 1. Hiển thị Ground Truth 3D
    subplot(2, 3, 1);
    surf(X, Y, ground_truth, 'EdgeColor', 'none');
    title('Ground Truth (Bề mặt 3D)', 'FontWeight', 'bold');
    xlabel('X'); ylabel('Y'); zlabel('Z');
    colorbar;
    view(45, 30);
    
    % 2. So sánh các sai số bằng bar chart
    subplot(2, 3, 2);
    valid_rmse = rmse_values(~isnan(rmse_values));
    valid_mae = mae_values(~isnan(mae_values));
    valid_max = max_error_values(~isnan(max_error_values));
    valid_names = result_names(~isnan(rmse_values));
    
    if ~isempty(valid_rmse)
        x_pos = 1:length(valid_rmse);
        bar_width = 0.25;
        
        bar(x_pos - bar_width, valid_rmse, bar_width, 'FaceColor', [0.2 0.6 0.8], 'DisplayName', 'RMSE');
        hold on;
        bar(x_pos, valid_mae, bar_width, 'FaceColor', [0.8 0.4 0.2], 'DisplayName', 'MAE');
        bar(x_pos + bar_width, valid_max, bar_width, 'FaceColor', [0.6 0.8 0.2], 'DisplayName', 'Max Error');
        
        xlabel('Bề mặt kết quả');
        ylabel('Giá trị sai số');
        title('So sánh các loại sai số');
        set(gca, 'XTickLabel', valid_names);
        xtickangle(45);
        legend('show');
        grid on;
        hold off;
    end
    
    % 3. Hiển thị bề mặt đầu tiên (nếu có)
    if ~isempty(valid_results)
        k = valid_results(1);
        subplot(2, 3, 3);
        surf(X, Y, varargin{k}, 'EdgeColor', 'none');
        title(sprintf('Bề mặt %d (Kết quả)', k), 'FontWeight', 'bold');
        xlabel('X'); ylabel('Y'); zlabel('Z');
        colorbar;
        view(45, 30);
    end
    
    % 4. Hiển thị bản đồ sai số (Error Map) của bề mặt đầu tiên
    if ~isempty(valid_results)
        k = valid_results(1);
        subplot(2, 3, 4);
        error_surface = all_absolute_errors{k};
        imagesc(error_surface);
        title(sprintf('Bản đồ Sai số Bề mặt %d', k), 'FontWeight', 'bold');
        xlabel('Cột (X)'); ylabel('Hàng (Y)');
        colorbar;
        colormap(gca, 'hot');
    end
    
    % 5. Histogram tổng hợp sai số
    subplot(2, 3, 5);
    hold on;
    for k = valid_results
        error_data = all_absolute_errors{k}(:);
        histogram(error_data, 30, 'FaceAlpha', 0.7, 'EdgeColor', 'none', ...
                 'DisplayName', sprintf('Bề mặt %d', k));
    end
    xlabel('Sai số tuyệt đối');
    ylabel('Tần suất');
    title('Phân bố Sai số các Bề mặt');
    legend('show');
    grid on;
    hold off;
    
    % 6. Cross-section comparison (cắt ngang tại giữa)
    subplot(2, 3, 6);
    mid_row = ceil(m/2);
    plot(1:n, ground_truth(mid_row, :), 'k-', 'LineWidth', 3, 'DisplayName', 'Ground Truth');
    hold on;
    
    for k = valid_results
        final_result = varargin{k};
        plot(1:n, final_result(mid_row, :), '--', 'LineWidth', 2, ...
             'Color', colors(k, :), 'DisplayName', sprintf('Bề mặt %d', k));
    end
    
    xlabel('Vị trí X');
    ylabel('Giá trị Z');
    title(sprintf('Cắt ngang tại hàng %d', mid_row));
    legend('show');
    grid on;
    hold off;
    
    % Tiêu đề chung cho figure
    sgtitle('Phân tích So sánh Bề mặt 3D', 'FontSize', 16, 'FontWeight', 'bold');
    
    % Nếu có nhiều bề mặt, tạo figure riêng để so sánh trực quan
    if length(valid_results) > 1
        fig2 = figure('Position', [100, 100, 1200, 400]);
        
        % Hiển thị tất cả bề mặt cạnh nhau
        num_plots = length(valid_results) + 1;
        cols = ceil(sqrt(num_plots));
        rows = ceil(num_plots / cols);
        
        % Ground Truth
        subplot(rows, cols, 1);
        surf(X, Y, ground_truth, 'EdgeColor', 'none');
        title('Ground Truth', 'FontWeight', 'bold');
        xlabel('X'); ylabel('Y'); zlabel('Z');
        view(45, 30);
        
        % Các bề mặt kết quả
        for i = 1:length(valid_results)
            k = valid_results(i);
            subplot(rows, cols, i+1);
            surf(X, Y, varargin{k}, 'EdgeColor', 'none');
            title(sprintf('Bề mặt %d\nRMSE: %.4f', k, rmse_values(k)), 'FontWeight', 'bold');
            xlabel('X'); ylabel('Y'); zlabel('Z');
            view(45, 30);
        end
        
        sgtitle('So sánh Trực quan Tất cả Bề mặt', 'FontSize', 14, 'FontWeight', 'bold');
    end
    
    % In bảng tóm tắt chi tiết
    fprintf('\n=== BẢNG TÓM TẮT SAI SỐ BỀ MẶT 3D ===\n');
    fprintf('%-12s | %-10s | %-10s | %-12s\n', 'Bề mặt', 'RMSE', 'MAE', 'Max Error');
    fprintf('%-12s | %-10s | %-10s | %-12s\n', '----------', '--------', '--------', '----------');
    for k = 1:num_results
        if ~isnan(rmse_values(k))
            fprintf('%-12s | %-10.6f | %-10.6f | %-12.6f\n', ...
                    sprintf('Bề mặt %d', k), rmse_values(k), mae_values(k), max_error_values(k));
        else
            fprintf('%-12s | %-10s | %-10s | %-12s\n', ...
                    sprintf('Bề mặt %d', k), 'Lỗi', 'Lỗi', 'Lỗi');
        end
    end
    fprintf('\nKích thước bề mặt: [%d x %d] = %d điểm dữ liệu\n', m, n, m*n);
    
    % Tìm bề mặt tốt nhất
    if ~isempty(valid_results)
        [~, best_rmse_idx] = min(rmse_values(valid_results));
        [~, best_mae_idx] = min(mae_values(valid_results));
        
        fprintf('\n=== ĐÁNH GIÁ ===\n');
        fprintf('Bề mặt tốt nhất theo RMSE: Bề mặt %d (RMSE = %.6f)\n', ...
                valid_results(best_rmse_idx), min(rmse_values(valid_results)));
        fprintf('Bề mặt tốt nhất theo MAE: Bề mặt %d (MAE = %.6f)\n', ...
                valid_results(best_mae_idx), min(mae_values(valid_results)));
    end
    fprintf('\n');
end
end

%% them ngay 14/8/2025
function im_unwrapped = goldstein_unwrap(phase_wrapped)
    % GOLDSTEIN_UNWRAP - Phase unwrapping theo phương pháp Goldstein
    % Input:
    %   IM  - ảnh phức (complex image), IM = mag .* exp(1i * wrapped_phase)
    % Output:
    %   im_unwrapped - ảnh pha đã unwrap

    % 1. Khởi tạo
    % Biên độ (magnitude) = 1 
    mag = ones(size(phase_wrapped));
    IM = mag .* exp(1i * phase_wrapped);   

    im_mag   = abs(IM);       % Magnitude
    im_phase = angle(IM);     % Wrapped phase
    im_mask  = ones(size(IM));

    % 2. Tính residues
    residue_charge = PhaseResidues_r1(im_phase, im_mask);

    % 3. Tạo branch cuts
    max_box_radius = 4;
    branch_cuts = BranchCuts_r1(residue_charge, max_box_radius, im_mask);

    % 4. Loại branch cuts khỏi mask
    im_mask(branch_cuts) = 0;
    im_mag1 = im_mag .* im_mask;

    % 5. Chọn điểm tham chiếu (tự động chọn magnitude lớn nhất)
    [r_dim, c_dim] = size(im_phase);
    im_mag1([1 r_dim], :) = 0;
    im_mag1(:, [1 c_dim]) = 0;
    [~, idx_max] = max(im_mag1(:));
    [rowref, colref] = ind2sub(size(im_mag1), idx_max);

    % 6. Unwrap
    im_unwrapped = FloodFill_r1(im_phase, im_mag, branch_cuts, im_mask, colref, rowref);
end
