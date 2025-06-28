clc; clear; close all;

% --- 1. Tạo mặt pha giả có nghiêng ---
N = 512;
[X, Y] = meshgrid(1:N, 1:N);

% Pha nền nghiêng + đỉnh Gaussian
% phi_true = 0.01*X + 0.02*Y + 5 * exp(-((X - N/2).^2 + (Y - N/2).^2)/(2*50^2));
phi_true = 0.002*X.^2 - 0.003*Y.^2 + 3 * exp(-((X - N/2).^2 + (Y - N/2).^2)/(2*40^2));

% --- 2. Áp dụng Fourier + loại nghiêng ---
phi_corrected = remove_tilt_fourier(phi_true, 0.5);  % giữ 10% tần số thấp

%%
% Tính mặt nghiêng đã loại bỏ
phi_plane = phi_true - phi_corrected;

% Tính sai số
mae = mean(abs(phi_plane(:)));
rmse = sqrt(mean((phi_plane(:)).^2));

% Hiển thị sai số
fprintf('>> Sai số nghiêng đã loại bỏ:\n');
fprintf('   MAE  = %.4f\n', mae);
fprintf('   RMSE = %.4f\n', rmse);


% --- 3. Hiển thị ---
figure;
subplot(1,3,1);
surf(phi_true); shading interp; title('Pha gốc có nghiêng');
view(45, 30); colormap turbo;

subplot(1,3,2);
surf(phi_corrected); shading interp; title('Pha đã loại bỏ nghiêng');
view(45, 30); colormap turbo;

subplot(1,3,3);
surf(phi_true - phi_corrected); shading interp; title('Phần nghiêng đã loại');
view(45, 30); colormap turbo;


% --- 4. Hàm loại bỏ nghiêng ---
function phi_corrected = remove_tilt_fourier(phi, keep_ratio)
    if nargin < 2
        keep_ratio = 0.1;
    end

    [N, M] = size(phi);
    F = fftshift(fft2(phi));

    % Tạo mặt nạ tần số thấp
    cx = floor(M/2)+1;
    cy = floor(N/2)+1;
    rx = floor(M * keep_ratio / 2);
    ry = floor(N * keep_ratio / 2);
    mask = zeros(N, M);
    mask(cy-ry:cy+ry, cx-rx:cx+rx) = 1;

    % Biến đổi ngược sau khi lọc
    F_low = F .* mask;
    phi_low = real(ifft2(ifftshift(F_low)));

    % Fit mặt phẳng nghiêng bằng fit (thay polyfitn)
    [X, Y] = meshgrid(1:M, 1:N);
    tbl = table(X(:), Y(:), phi_low(:), 'VariableNames', {'x', 'y', 'z'});
    f = fit([tbl.x, tbl.y], tbl.z, 'poly11');
    phi_plane = reshape(f(X, Y), size(phi));

    % Trừ nghiêng
    phi_corrected = phi - phi_plane;
end

