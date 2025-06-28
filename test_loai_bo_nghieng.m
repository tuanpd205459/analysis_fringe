clc; clear; close all;

% --- 1. Tạo bề mặt pha có nghiêng + lồi ---
N = 512;
[X, Y] = meshgrid(1:N, 1:N);
phi = 0.01*X + 0.02*Y + 4 * exp(-((X - N/2).^2 + (Y - N/2).^2)/(2*50^2));

% --- 2. Hiển thị ảnh để bạn chọn điểm ---
figure;
imagesc(phi); axis image; colormap turbo;
title('Click các điểm trên vùng phẳng nghiêng (Ấn Enter khi xong)');

% Ginput: chọn điểm rồi Enter
[x_pts, y_pts] = ginput();  
z_pts = interp2(phi, x_pts, y_pts);

% --- 3. Sau khi chọn xong, hiển thị lại ảnh + điểm đã chọn ---
figure;
imagesc(phi); axis image; colormap turbo; hold on;
title('Pha gốc và các điểm đã chọn');
plot(x_pts, y_pts, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
for i = 1:length(x_pts)
    text(x_pts(i)+5, y_pts(i), sprintf('%d', i), ...
        'Color', 'w', 'FontSize', 10, 'FontWeight', 'bold');
end
hold off;

% --- 4. Fit mặt phẳng từ các điểm đã chọn ---
tbl = table(x_pts, y_pts, z_pts, 'VariableNames', {'x', 'y', 'z'});
f = fit([tbl.x, tbl.y], tbl.z, 'poly11');  % z = ax + by + c

% --- 5. Nội suy mặt phẳng toàn ảnh và trừ ---
phi_plane = f(X, Y);
phi_corrected = phi - phi_plane;

% --- 6. Hiển thị kết quả so sánh ---
figure;
subplot(1,3,1);
imagesc(phi); axis image; colormap turbo; title('Pha gốc');

subplot(1,3,2);
imagesc(phi_plane); axis image; colormap turbo; title('Mặt phẳng đã fit');

subplot(1,3,3);
imagesc(phi_corrected); axis image; colormap turbo; title('Pha đã loại nghiêng');
