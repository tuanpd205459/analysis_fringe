clc, clear,close all;
% Define the domain
% N = 200;
% [x, sigma] = meshgrid(linspace(0, 1, N), linspace(0, pi/5, N));
% % Define the constant 'a'
% a = N * pi / 2;
% 
% % Define the zero-mean Gaussian noise component
% % This is a random term for each point, with a standard deviation of sigma
% eta = randn(N, N) .* sigma;
% 
% % Calculate the unwrapped phase distribution
% phi = a * x.^2 + eta;
M  = 512;

N = M;
noise_level = 0.3;
[X, Y] = meshgrid(linspace(-1, 1, N), linspace(-1, 1, M));
phi_ground_truth = 2 * peaks(3*X, 3*Y) ;

[x, sigma] = meshgrid(linspace(0, 1, N), linspace(0, pi/5, N));

% Define the zero-mean Gaussian noise component
% This is a random term for each point, with a standard deviation of sigma
sigma = pi/5;
eta = randn(N, N) .* sigma;

phi_ground_truth = phi_ground_truth + eta;

phi = phi_ground_truth;

% Display the simulated unwrapped phase
figure;
surf(phi,"EdgeColor","none");
title('Simulated Unwrapped Phase Distribution');
xlabel('x');
ylabel('σ (noise standard deviation)');
zlabel('Unwrapped Phase \phi');
colorbar;

% You can also generate the wrapped phase map for visualization
wrapped_phi = mod(phi, 2*pi);
figure;
surf(wrapped_phi,"EdgeColor","none");
title('Simulated Wrapped Phase Map');
xlabel('x');
ylabel('σ (noise standard deviation)');
colorbar;
axis xy; % Corrects the y-axis direction

wrappedPhase = wrapped_phi;
unwrapped_Phase_LS_DCT = unwrapping.unwrapPhase(wrappedPhase, 'ls', 'dct'); % LS với DCT
unwrapped_Phase_TIE_FFT = unwrapping.unwrapPhase(wrappedPhase, 'tie', 'fft'); % TIE với FFT
unwrapped_Phase_noncontinue = unwrapping.unwrapPhase(wrappedPhase, 'linh'); % Phương pháp của a Linh
unwrapped_Phase_2dweight = unwrapping.unwrapPhase(wrappedPhase, '2dweight'); % 2D weighted phase unwrapping

% --- Tính sai số so với groundtruth (phi) ---

% Hàm tính RMSE và MAE
calc_rmse = @(est, gt) sqrt(mean((est(:)-gt(:)).^2));
calc_mae  = @(est, gt) mean(abs(est(:)-gt(:)));

% (Nếu cần căn chỉnh offset hoặc modulo 2pi để so sánh - tùy phương pháp)
% Ở đây giả định đã căn chỉnh rồi, hoặc phase unwrapped nằm cùng hệ với phi.

methods = {'LS DCT', 'TIE FFT', 'Following Non-Cont Path', '2D-Weight'};
phases = {unwrapped_Phase_LS_DCT, unwrapped_Phase_TIE_FFT, unwrapped_Phase_noncontinue, unwrapped_Phase_2dweight};

for i = 1:length(methods)
    err_map = phases{i} - phi;
    rmse = calc_rmse(phases{i}, phi);
    mae  = calc_mae(phases{i}, phi);

    figure('Name', ['Sai số: ' methods{i}]);
    surf(err_map, 'EdgeColor', 'none');
    title({['Bản đồ sai số: ' methods{i}], ...
        ['RMSE = ' num2str(rmse, '%.4f') ', MAE = ' num2str(mae, '%.4f')]});
    xlabel('x'); ylabel('y'); zlabel('Sai số (rad)');
    colormap; colorbar;
end

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