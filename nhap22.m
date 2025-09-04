%% Script ví dụ để chạy hàm ZernikeLegendreFit
% Script này sẽ:
% 1. Tạo một bề mặt mẫu (z_map) từ sự kết hợp của các đa thức Zernike.
% 2. Chạy hàm ZernikeLegendreFit để tìm các hệ số.
% 3. Hiển thị kết quả dưới dạng văn bản và biểu đồ.

clear; clc; close all;

%% 1. Tạo dữ liệu bề mặt đầu vào (z_map)
fprintf('Đang tạo bề mặt mẫu...\n');

% Định nghĩa lưới tọa độ
pixel_count = 512; % Độ phân giải của bề mặt
[X, Y] = meshgrid(linspace(-1, 1, pixel_count));
[theta, rho] = cart2pol(X, Y);

% Tạo một bề mặt từ sự kết hợp của 3 đa thức Zernike (theo định nghĩa Fringe)
% Đây là các hệ số "thực" mà chúng ta muốn tìm lại
A5 = 1.2;  % Z_j=5 (Astigmatism dọc trục y)
A8 = -0.7; % Z_j=8 (Coma dọc trục x)
A11 = 0.5; % Z_j=11 (Sai số cầu bậc 3)

% Các phương trình Zernike (Fringe indexing)
Z5 = A5 * (rho.^2 .* sin(2*theta));
Z8 = A8 * ((3*rho.^3 - 2*rho) .* cos(theta));
Z11 = A11 * (6*rho.^4 - 6*rho.^2 + 1);

% Kết hợp chúng lại để tạo bề mặt cuối cùng
z_map = Z5 + Z8 + Z11;

% Đặt các giá trị bên ngoài vòng tròn đơn vị là NaN (Not-a-Number)
% vì Zernike chỉ được định nghĩa trong một vòng tròn
z_map(rho > 1) = NaN;

%% 2. Gọi hàm ZernikeLegendreFit
fprintf('Đang thực hiện khớp Zernike...\n');

% --- Cài đặt cho việc khớp ---
% Chúng ta sẽ sử dụng kiểu chỉ số "fringe" vì nó dễ sử dụng hơn
index_type = 'fringe';

% Chúng ta muốn khớp tới hệ số Zernike thứ 15.
% Điều này đủ để phát hiện các hệ số Z5, Z8, và Z11 mà chúng ta đã tạo.
coeff_max = 15; 

% Gọi hàm chính
% Chúng ta bỏ qua các tham số tùy chọn (J, K, center_j, center_i)
% để hàm tự động tính toán chúng.
[output_coeff, z_recon_map] = ZernikeLegendreFit(z_map, index_type, coeff_max);

%% 3. Hiển thị các hệ số đã tìm được
fprintf('Hoàn tất khớp Zernike. Các hệ số đã tìm được:\n');

% Lấy mảng hệ số từ cell array
fitted_coeffs = output_coeff{1};

% In các hệ số khác không đáng kể để kiểm tra
% (Chúng ta dùng một ngưỡng nhỏ để tránh in ra các giá trị nhiễu rất nhỏ)
for j = 1:length(fitted_coeffs)
    if abs(fitted_coeffs(j)) > 1e-4
        fprintf('  Hệ số c(%d) = %f\n', j, fitted_coeffs(j));
    end
end

fprintf('\nSo sánh với các hệ số gốc:\n');
fprintf('  A5  = %.1f (Tìm được: %.4f)\n', A5, fitted_coeffs(5));
fprintf('  A8  = %.1f (Tìm được: %.4f)\n', A8, fitted_coeffs(8));
fprintf('  A11 = %.1f (Tìm được: %.4f)\n', A11, fitted_coeffs(11));

%% 4. Vẽ biểu đồ kết quả
fprintf('Đang vẽ biểu đồ kết quả...\n');

figure('Name', 'Kết quả khớp Zernike', 'Position', [100, 100, 1500, 450]);

% Biểu đồ 1: Bề mặt gốc
subplot(1, 3, 1);
imagesc(z_map);
axis square;
colorbar;
title('Bề mặt gốc (Original Surface)');
xlabel('X'); ylabel('Y');

% Biểu đồ 2: Bề mặt được tái tạo từ các hệ số
subplot(1, 3, 2);
imagesc(z_recon_map);
axis square;
colorbar;
title(['Bề mặt tái tạo (Reconstructed Surface) j_{max} = ' num2str(coeff_max)]);
xlabel('X'); ylabel('Y');

% Biểu đồ 3: Sai số (Sự khác biệt giữa gốc và tái tạo)
residual_error = z_map - z_recon_map;
subplot(1, 3, 3);
imagesc(residual_error);
axis square;
colorbar;
title('Sai số còn lại (Residual Error)');
xlabel('X'); ylabel('Y');

sgtitle('So sánh kết quả khớp bề mặt Zernike', 'FontSize', 16, 'FontWeight', 'bold');