clc; clear; close all;
% Kích thước ảnh
N = 128;
[X, Y] = meshgrid(1:N, 1:N);

% Tạo bề mặt bậc step: một nửa có giá trị 0, một nửa có giá trị pi
Z = zeros(N);
Z(:, 1:N/2) = 0;
Z(:, N/2+1:end) = 3*pi;  % Bước nhảy đột ngột tại giữa ảnh

% Mô phỏng pha bị wrap (gói) về [-pi, pi]
wrapped_phase = angle(exp(1i * Z));

% Unwrap pha bằng hàm unwrap 2D (dùng unwrap theo từng hàng và cột)
unwrapped_phase = unwrap(unwrap(wrapped_phase, [], 1), [], 2);

% Hiển thị kết quả
figure;
surf(Z);
colorbar; title('Pha gốc (step surface)');

figure;surf(wrapped_phase);
colorbar; title('Wrapped phase');

figure; surf(unwrapped_phase);
colorbar; title('Unwrapped phase');
