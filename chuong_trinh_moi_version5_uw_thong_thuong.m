% =========================================================================
% SCRIPT CHÍNH: MÔ PHỎNG VÀ TÁI TẠO PHA TỪ OFF-AXIS HOLOGRAM
%
% MÔ TẢ:
% Script này thực hiện toàn bộ quy trình từ việc mô phỏng một hologram
% ngoài trục (off-axis), tái tạo lại pha đã bọc (wrapped phase) từ
% hologram đó, và cuối cùng là giải bọc pha (phase unwrapping) và
% phân tích sai số so với pha gốc (ground truth).
%
% =========================================================================

%% === KHỞI TẠO MÔI TRƯỜNG ===
clc;            % 
clear;          % 
close all;      % 

%% === 1. THIẾT LẬP CÁC THÔNG SỐ ===
fprintf('1. Đang thiết lập các thông số mô phỏng và vật lý...\n');
params = define_simulation_parameters();

% Tạo lưới tọa độ
x_vec = linspace(-1, 1, params.imageSize.X);
y_vec = linspace(-1, 1, params.imageSize.Y);
[X, Y] = meshgrid(x_vec, y_vec);

%% === 2. MÔ PHỎNG HOLOGRAM ===
fprintf('2. Đang tạo sóng vật thể, sóng tham chiếu và hologram...\n');
% Tạo sóng vật thể (Object Wave)
[Es, phi_ground_truth] = create_object_wave(params, X, Y);

% Tạo sóng tham chiếu (Reference Wave)
[E0, phi_ref] = create_reference_wave(params, X, Y);

% Mô phỏng ảnh giao thoa (Hologram) từ sự giao thoa của hai sóng
hologram = simulate_hologram(Es, E0);

% Chuẩn hóa và lưu ảnh hologram
hologram_normalized = mat2gray(hologram);
imwrite(hologram_normalized, 'hologram.bmp');

% Hiển thị các kết quả mô phỏng ban đầu
plot_simulation_inputs(phi_ground_truth, phi_ref, hologram);
fprintf('   -> Đã mô phỏng và lưu hologram.bmp\n');

%% === 3. TÁI TẠO PHA TỪ HOLOGRAM ===
fprintf('3. Đang tái tạo pha (wrapped phase) từ hologram...\n');
fprintf('   LƯU Ý: Vui lòng vẽ một hình chữ nhật quanh phổ bậc +1 và DOUBLE-CLICK để xác nhận.\n');

% Tái tạo pha bằng phương pháp lọc Fourier (tương tác)
[wrappedPhase, params] = reconstruct_phase_interactively(hologram, params);

% Tùy chọn: Loại bỏ độ nghiêng (tilt) khỏi mặt pha đã bọc
choice_tilt = questdlg('Bạn có muốn tự động loại bỏ độ nghiêng (tilt) khỏi mặt pha vừa tái tạo không?', ...
    'Xác nhận loại bỏ nghiêng', ...
    'Có', 'Không', 'Có');
if strcmp(choice_tilt, 'Có')
    fprintf('   -> Đang loại bỏ độ nghiêng...\n');
    [wrappedPhase, ~] = remove_tilt_from_wrapped2(wrappedPhase);
end

% Hiển thị kết quả tái tạo ban đầu
figure('Name', 'So sánh Pha Gốc và Pha Tái tạo (Wrapped)');
subplot(1, 2, 1);
imagesc(phi_ground_truth); axis image; colorbar; colormap(gca, jet);
title('Pha Gốc (Ground Truth)');
xlabel('x'); ylabel('y');
subplot(1, 2, 2);
imagesc(wrappedPhase); axis image; colorbar; colormap(gca, jet);
title('Pha Tái tạo (Wrapped)');
xlabel('x'); ylabel('y');
fprintf('   -> Hoàn thành tái tạo pha.\n');

%% === 4. TẢI VÀ CĂN CHỈNH DỮ LIỆU PHA ƯỚC LƯỢNG ===
% GHI CHÚ QUAN TRỌNG:
% Ở bước này, bạn cần tải dữ liệu pha ước lượng (phi_est) từ nguồn
% bên ngoài (ví dụ: một ứng dụng, một file .mat khác).
% Sau đó, bạn phải đảm bảo rằng tất cả các ma trận pha
% (phi_ground_truth, wrappedPhase, phi_est) có cùng kích thước
% để có thể so sánh và tính toán sai số.
fprintf('4. Đang chuẩn bị dữ liệu cho bước giải bọc pha và phân tích...\n');

% --- !!! NGƯỜI DÙNG CẦN CHỈNH SỬA TẠI ĐÂY !!! ---
% TODO: Thay thế dòng dưới đây bằng mã tải dữ liệu của bạn.
% Ví dụ: load('my_estimated_phase_data.mat', 'phi_est');
% Để script có thể chạy được, chúng ta tạm gán phi_est bằng phi_ground_truth
% (trường hợp ước lượng hoàn hảo).
phi_est = phi_ground_truth;
fprintf('   -> (Tạm thời) Đã gán phi_est = phi_ground_truth để chạy demo.\n');
% --- KẾT THÚC PHẦN CHỈNH SỬA ---

phi_ground_truth_aligned = phi_ground_truth;
wrappedPhase_aligned     = wrappedPhase;
phi_est_aligned          = phi_est; % Giả sử phi_est đã được căn chỉnh

% Kiểm tra lại kích thước để đảm bảo
assert(isequal(size(phi_ground_truth_aligned), size(wrappedPhase_aligned), size(phi_est_aligned)), ...
    'Kích thước các ma trận pha không đồng nhất. Vui lòng kiểm tra lại bước căn chỉnh.');
fprintf('   -> Đã thống nhất các biến _aligned.\n');

%% === 5. GIẢI BỌC PHA VÀ TÍNH TOÁN SAI SỐ ===
fprintf('5. Đang giải bọc pha sử dụng pha ước lượng...\n');

% Sử dụng các biến _aligned đã được thống nhất
finalUnwrappedPhase = poisson.unwrap_TIE_FD_FFT_iter(wrappedPhase_aligned);
[finalUnwrappedPhase, ~] = remove_plane_manual(finalUnwrappedPhase);
offset_val = 10;
[wrappedPhase_aligned, finalUnwrappedPhase, phi_ground_truth_aligned] = alignAndCropPhaseMaps...
    (wrappedPhase, finalUnwrappedPhase, phi_ground_truth, offset_val);
fprintf('   -> Giải bọc pha hoàn tất.\n');

fprintf('6. Đang tính toán các chỉ số sai số toàn diện...\n');
% Sử dụng các biến _aligned đã được thống nhất


%% === 5.5. HIỂN THỊ SAI LỆCH GIỮA PHA TRƯỚC VÀ SAU KHI GIẢI BỌC (PHẦN THÊM MỚI) ===
fprintf('5.5. Đang hiển thị giá trị sai lệch giữa pha sau và trước khi giải bọc...\n');

% Tính toán sai lệch (chính là bản đồ bậc vân 2*pi*k)
phase_difference = finalUnwrappedPhase - phi_ground_truth_aligned;

% Hiển thị kết quả
figure('Name', 'Sai lệch do Giải bọc pha (Unwrapped - Wrapped)');

% Bề mặt 3D của sai lệch
subplot(1, 2, 1);
surf(phase_difference, 'EdgeColor', 'none');
title({'Giá trị sai lệch (3D)', 'finalUnwrappedPhase - phi_ground_truth_aligned'});
xlabel('x'); ylabel('y');
axis tight;
view(45, 30);
colorbar;
colormap(gca, parula);
fprintf('   -> Đã tạo đồ thị 3D của sai lệch.\n');

% Hình ảnh 2D của sai lệch
subplot(1, 2, 2);
imagesc(phase_difference);
title({'Bản đồ sai lệch (2D)', 'finalUnwrappedPhase - phi_ground_truth_aligned'});
axis image;
colorbar;
colormap(gca, parula);
fprintf('   -> Đã tạo bản đồ 2D của sai lệch.\n');


%% === 6. HIỂN THỊ KẾT QUẢ PHÂN TÍCH VÀ LƯU ẢNH ===
fprintf('7. Đang hiển thị kết quả phân tích cuối cùng...\n');
% Hiển thị bảng tóm tắt sai số

% Tạo các hình ảnh phân tích chi tiết
fprintf('8. Đang tạo các hình ảnh phân tích chi tiết...\n');

% Visualization tổng quan - Sử dụng các biến _aligned
% Lưu ý: kMap không được định nghĩa, nên tôi giữ nguyên lời gọi hàm gốc
create_overview_visualization(phi_ground_truth_aligned, phi_est_aligned, ...
                            wrappedPhase_aligned, finalUnwrappedPhase); 

% % Phân tích sai số nâng cao - Sử dụng các biến _aligned
% create_advanced_error_analysis(finalUnwrappedPhase, phi_est_aligned, ...
%                               phi_ground_truth_aligned, error_metrics);

% Tùy chọn: Lưu tất cả các hình ảnh đã tạo
figs = findall(0, 'Type', 'figure');
choice_save = questdlg('Bạn có muốn lưu tất cả các hình ảnh phân tích không?', ...
    'Xác nhận lưu', ...
    'Có', 'Không', 'Có');
if strcmp(choice_save, 'Có')
    fprintf('9. Đang lưu các hình ảnh...\n');
    for i = 1:length(figs)
        fig = figs(i);
        figure(fig); % Đưa figure lên phía trước
        
        % Lấy tên figure để làm tên file
        name = get(fig, 'Name');
        if isempty(name)
            name = sprintf('Figure_%d', fig.Number);
        end
        
        % Chuyển đổi tên thành tên file hợp lệ
        name_valid = matlab.lang.makeValidName(name);
        filename = [name_valid '.png'];
        saveas(fig, filename);
        fprintf('   -> Đã lưu: %s\n', filename);
    end
end

fprintf('=========================================\n');
fprintf('           HOÀN THÀNH!\n');
fprintf('=========================================\n');


%% ========================================================================
%               CÁC HÀM PHỤ TRỢ (KHÔNG THAY ĐỔI)
% ========================================================================

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
    
    % Các loại vật thể: 'gaussian', 'peaks', 'zernike', "sinusoidal",
    %                                   "concentric_rings",
    %                                   "spiral",'custom'
    params.object.type = 'sinusoidal';
    
    % Cấu hình cho vật thể Zernike (LƯU Ý: cần hàm tao_da_thuc_zernike)
    params.object.zernike.indices = [4, 5, 6, 7, 8, 11];
    params.object.zernike.coefficients = [1.5, -0.8, 0.6, 0.7, -0.5, -1.2];
end
% -------------------------------------------------------------------------
% function [Es, phi_vat] = create_object_wave(params, X_norm, Y_norm)
%     % Tạo trường sóng vật thể từ pha, sử dụng tọa độ đã chuẩn hóa [-1, 1].
%     amp = params.object.amplitude;
%     switch params.object.type
%         case 'gaussian'
%             phi_vat = amp * exp(-5 * (X_norm.^2 + Y_norm.^2));
%         case 'peaks'
%             % Hàm peaks hoạt động tốt trên lưới tọa độ [-3, 3]
%             % Chúng ta co giãn X_norm, Y_norm cho phù hợp
%             phi_vat = amp/8 * peaks(3*X_norm, 3*Y_norm); 
%      
%         case 'sinusoidal'
%             freq_x = 2;
%             freq_y = 0;
%             phi_vat = amp * sin(2 * pi * freq_x * X_norm) .* cos(2 * pi * freq_y * Y_norm);
%         case 'concentric_rings'
%             R = sqrt(X_norm.^2 + Y_norm.^2);
%             rings = floor(R * 15); % 15 là số lượng vòng
%             phi_vat = amp * sin(rings);
%         case 'spiral'
%             R = sqrt(X_norm.^2 + Y_norm.^2);
%             Theta = atan2(Y_norm, X_norm);
%             num_turns = 5;
%             phi_vat = amp * (num_turns * Theta + 2*pi*R);
%         case 'custom'
%             P1 = 20 * exp(-((X_norm-0.4).^2 + (Y_norm-0.4).^2) / 0.1); % Đỉnh lồi
%             P2 = -18 * exp(-((X_norm+0.3).^2 + (Y_norm+0.2).^2) / 0.08); % Đỉnh lõm
%             phi_vat = P1+ P2;
%         otherwise
%             error("Loại vật thể '%s' không hỗ trợ.", params.object.type);
%     end
%     
%     % Tạo trường sóng phức từ pha
%     noise_level = 0.5; % Điều chỉnh độ lớn của nhiễu
%     noise = noise_level * randn(size(phi_vat));
%     phi_vat = phi_vat + noise;
%     Es = exp(1i * phi_vat);
% 
% end
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
    phi_vat = P1 + P2;
otherwise
    error("Loại vật thể '%s' không hỗ trợ.", params.object.type);
end

% Thêm bề mặt nhấp nhô tế vi thay vì nhiễu Gaussian
roughness_amplitude = 0.3; % Biên độ của độ nhấp nhô (có thể điều chỉnh)
roughness_frequency = 50;   % Tần số không gian của độ nhấp nhô

% Tạo nhiều tần số khác nhau để mô phỏng bề mặt tự nhiên
surface_roughness = zeros(size(X_norm));

% Thêm nhiều thành phần tần số với trọng số khác nhau
frequencies = [30, 50, 80, 120];
amplitudes = [0.4, 0.3, 0.2, 0.1]; % Giảm dần theo tần số cao

for i = 1:length(frequencies)
    freq = frequencies(i);
    amp_rough = amplitudes(i) * roughness_amplitude;
    
    % Tạo pattern ngẫu nhiên với tần số cụ thể
    phase_x = 2*pi*rand(); % Pha ngẫu nhiên
    phase_y = 2*pi*rand();
    
    surface_roughness = surface_roughness + ...
        amp_rough * sin(freq * X_norm + phase_x) .* sin(freq * Y_norm + phase_y) + ...
        amp_rough * cos(freq * X_norm * 1.3 + phase_x) .* cos(freq * Y_norm * 0.7 + phase_y);
end

% Thêm thành phần không gian tương quan (spatial correlation)
% Sử dụng hàm Gaussian 2D với kích thước kernel nhỏ để tạo correlation
correlation_size = 0.05; % Kích thước correlation (trong tọa độ chuẩn hóa)
[nx, ny] = size(X_norm);
dx = 2/(nx-1); % Bước lưới trong tọa độ chuẩn hóa
dy = 2/(ny-1);

% Tạo kernel Gaussian cho correlation
kernel_size = round(correlation_size/dx);
if kernel_size < 3, kernel_size = 3; end
if mod(kernel_size,2) == 0, kernel_size = kernel_size + 1; end

sigma = kernel_size/6; % Sigma của Gaussian kernel
kernel = fspecial('gaussian', kernel_size, sigma);

% Tạo nhiễu trắng và lọc để có correlation
white_noise = randn(size(X_norm));
correlated_roughness = imfilter(white_noise, kernel, 'replicate');
correlated_roughness = correlated_roughness * (roughness_amplitude * 0.2);

% Kết hợp tất cả thành phần roughness
total_roughness = surface_roughness + correlated_roughness;

% Thêm roughness vào pha
% phi_vat = phi_vat + total_roughness;

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
    noise = noise_level * randn(size(hologram));
    hologram_noisy = hologram + noise;
    fprintf('   -> Đã thêm nhiễu cộng Gauss vào hologram.\n');
    I = hologram_noisy;
end
%--------------------------------------------------------------------------
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
% Lưu ý: Hàm này yêu cầu 5 đầu vào, nhưng kMap không được truyền trong script chính.
% Để tránh lỗi, tôi sẽ kiểm tra số lượng đối số đầu vào.
    figure('Name', 'Tổng quan các bề mặt Pha', 'Position', [50, 50, 1400, 800]);
    
    sgtitle('So sánh các Bề mặt Pha', 'FontSize', 16, 'FontWeight', 'bold');
    
    has_kMap = nargin == 5;
    num_plots = 4;
    if has_kMap
        num_plots = 5;
    end

    subplot(2, num_plots, 1); surf(phi_gt, 'EdgeColor', 'none'); title('Gốc'); axis tight; view(45, 30); colorbar;
    subplot(2, num_plots, 2); surf(phi_est, 'EdgeColor', 'none'); title('Pha Ước lượng'); axis tight; view(45, 30); colorbar;
    subplot(2, num_plots, 3); surf(phi_wrapped, 'EdgeColor', 'none'); title('Pha Wrapped'); axis tight; view(45, 30); colorbar;
    subplot(2, num_plots, 4); surf(phi_final, 'EdgeColor', 'none'); title('Kết quả Cuối cùng'); axis tight; view(45, 30); colorbar;
    if has_kMap
        subplot(2, num_plots, 5); surf(kMap, 'EdgeColor', 'none'); title('Bản đồ K (Fringe Order)'); axis tight; view(45, 30); colormap(gca, parula); colorbar;
    end
    
    subplot(2, num_plots, num_plots + 1); imagesc(phi_gt); title('Gốc (2D)'); axis image; colorbar;
    subplot(2, num_plots, num_plots + 2); imagesc(phi_est); title('Pha Ước lượng (2D)'); axis image; colorbar;
    subplot(2, num_plots, num_plots + 3); imagesc(phi_wrapped); title('Pha Wrapped (2D)'); axis image; colorbar;
    subplot(2, num_plots, num_plots + 4); imagesc(phi_final); title('Kết quả Cuối cùng (2D)'); axis image; colorbar;
    if has_kMap
        subplot(2, num_plots, num_plots + 5); imagesc(kMap); title('Bản đồ K (2D)'); axis image; colormap(gca, parula); colorbar;
    end
end
% -------------------------------------------------------------------------
function create_advanced_error_analysis(finalUnwrappedPhase, phi_est_aligned, phi_ground_truth_aligned, error_metrics)
% Chỗ này bạn có thể thêm các hàm vẽ đồ thị phân tích sai số chi tiết
% Ví dụ: vẽ histogram của sai số, biểu đồ phân tán, etc.
end
% -------------------------------------------------------------------------
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
            title("Anh 3D anh wrapped phase");
            fig1 = figure;
            imagesc(phi_wrapped); 
            axis image; 
            colormap jet; 
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
            
            % Unwrap các giá trị phase của 4 góc
            % Sử dụng góc đầu tiên làm reference
            corner_phases_unwrapped = corner_phases;
            for i = 2:4
                % Tính hiệu số với góc trước đó
                diff_phase = corner_phases(i) - corner_phases_unwrapped(i-1);
                
                % Unwrap nếu có jump > pi
                if diff_phase > pi
                    corner_phases_unwrapped(i) = corner_phases(i) - 2*pi;
                elseif diff_phase < -pi
                    corner_phases_unwrapped(i) = corner_phases(i) + 2*pi;
                else
                    corner_phases_unwrapped(i) = corner_phases(i);
                end
            end
            
            % Kiểm tra nếu vẫn có discontinuity lớn, thử cách khác
            max_diff = max(abs(diff(corner_phases_unwrapped)));
            if max_diff > pi
                % Thử unwrap theo cách khác: so với trung bình
                mean_phase = mean(corner_phases);
                for i = 1:4
                    diff_mean = corner_phases(i) - mean_phase;
                    if diff_mean > pi
                        corner_phases_unwrapped(i) = corner_phases(i) - 2*pi;
                    elseif diff_mean < -pi
                        corner_phases_unwrapped(i) = corner_phases(i) + 2*pi;
                    else
                        corner_phases_unwrapped(i) = corner_phases(i);
                    end
                end
            end
            
            % Hiển thị các điểm góc trên ảnh
            hold on;
            plot(corner_x, corner_y, 'ro', 'MarkerSize', 8, 'LineWidth', 2, 'MarkerFaceColor', 'y');
            for i = 1:4
                text(corner_x(i)+5, corner_y(i), sprintf('%.3f', corner_phases_unwrapped(i)), ...
                     'Color', 'w', 'FontSize', 8, 'FontWeight', 'bold', 'BackgroundColor', 'k');
            end
            hold off;
            
            % Fit mặt phẳng từ 4 điểm góc
            A = [corner_x(:), corner_y(:), ones(4,1)];
            
            % Kiểm tra điều kiện của ma trận A
            if rank(A) < 3
                warning('Các điểm góc không đủ để xác định mặt phẳng duy nhất. Sử dụng least squares.');
                coeffs = A \ corner_phases_unwrapped(:);
            else
                coeffs = A \ corner_phases_unwrapped(:);
            end
            
            % In thông tin về 4 điểm góc
            fprintf('4 điểm góc đã chọn:\n');
            fprintf('Góc 1 (trái-trên): x=%d, y=%d, phase=%.4f\n', corner_x(1), corner_y(1), corner_phases_unwrapped(1));
            fprintf('Góc 2 (phải-trên): x=%d, y=%d, phase=%.4f\n', corner_x(2), corner_y(2), corner_phases_unwrapped(2));
            fprintf('Góc 3 (trái-dưới): x=%d, y=%d, phase=%.4f\n', corner_x(3), corner_y(3), corner_phases_unwrapped(3));
            fprintf('Góc 4 (phải-dưới): x=%d, y=%d, phase=%.4f\n', corner_x(4), corner_y(4), corner_phases_unwrapped(4));
            
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
    
    % --- Hiển thị ảnh ban đầu để người dùng chọn phương thức ---
    figure;
    surf(phi, "EdgeColor", "none"); 
    colormap jet; 
    colorbar;
    title('Bản đồ pha gốc');
    
    figure;
    imagesc(phi); 
    axis image; 
    colormap jet; 
    colorbar;
    title('Bản đồ pha gốc');
    
    % --- Hộp thoại lựa chọn phương thức ---
    choice = questdlg('Chọn phương thức để xác định mặt phẳng:', ...
                     'Lựa chọn nội suy', ...
                     'Chọn điểm', 'Vẽ HCN', 'Chọn điểm');
    
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
    
    % --- Hiển thị lại ảnh với các điểm đã chọn ---
    figure;
    imagesc(phi); 
    axis image; 
    colormap jet; 
    hold on;
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
    
    % --- Hiển thị kết quả ---
    figure;
    sgtitle('Kết quả loại bỏ mặt phẳng nghiêng');
    
    subplot(1,3,1);
    imagesc(phi); 
    axis image; 
    colormap turbo; 
    colorbar;
    title('Pha gốc');
    
    subplot(1,3,2);
    imagesc(phi_plane); 
    axis image; 
    colormap turbo; 
    colorbar;
    title('Mặt phẳng đã fit');
    
    subplot(1,3,3);
    imagesc(phi_corrected); 
    axis image; 
    colormap turbo; 
    colorbar;
    title('Pha đã loại nghiêng');
    
    % In thông tin về quá trình fit
    fprintf('Đã sử dụng %d điểm để fit mặt phẳng.\n', length(x_pts));
    fprintf('Phạm vi giá trị pha gốc: [%.3f, %.3f]\n', min(phi(:)), max(phi(:)));
    fprintf('Phạm vi giá trị pha đã hiệu chỉnh: [%.3f, %.3f]\n', min(phi_corrected(:)), max(phi_corrected(:)));
    
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
