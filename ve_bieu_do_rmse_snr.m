% Dữ liệu SNR
snr = 35:-2:13;

% Dữ liệu RMSE từ kết quả mô phỏng
rmse_tie = [1.7745e-02, 2.8838e-02, 4.6249e-02, 7.6708e-02, 1.2350e-01, 1.7845e-01, 2.8248e-01, 4.4859e-01, 6.7055e-01, 1.0528e+00, 1.5909e+00, 2.4067e+00];
rmse_reliability = [2.3168e-16, 2.3258e-16, 2.3410e-16, 2.3254e-16, 2.3152e-16, 2.3003e-16, 2.2911e-16, 2.2899e-16, 2.2754e-16, 1.2467e-02, 1.8364e-01, 2.8134e+00];
rmse_2dwls = [5.2403e-12, 4.8041e-12, 5.4058e-12, 4.6842e-12, 4.9124e-12, 4.7683e-12, 5.0833e-12, 4.9121e-12, 1.0168e-02, 4.2563e-02, 2.9795e-01, 9.1409e-01];
rmse_goldstein = [2.6945e-16, 2.7036e-16, 2.7083e-16, 2.6742e-16, 2.6616e-16, 2.6336e-16, 2.6103e-16, 2.5922e-16, 1.2467e-02, 4.4949e-02, 2.0861e-01, 6.1870e-01];
rmse_proposed = [2.3168e-16, 2.3258e-16, 2.3410e-16, 2.3254e-16, 2.3152e-16, 2.3003e-16, 2.2911e-16, 2.2898e-16, 2.2750e-16, 2.2543e-16, 2.4933e-02, 1.5519e-01];

% --- Bắt đầu vẽ biểu đồ ---
figure;

% Vẽ "Proposed" đầu tiên để nó nhận màu mặc định số 1 (xanh dương)
semilogy(snr, rmse_proposed, '-*', 'LineWidth', 1.5, 'MarkerSize', 6);
hold on; % Giữ nguyên biểu đồ để vẽ các đường tiếp theo

% Vẽ các đường còn lại
semilogy(snr, rmse_tie, '-o', 'LineWidth', 1.5, 'MarkerSize', 4);
semilogy(snr, rmse_reliability, '-s', 'LineWidth', 1.5, 'MarkerSize', 4);
semilogy(snr, rmse_2dwls, '-^', 'LineWidth', 1.5, 'MarkerSize', 4);
semilogy(snr, rmse_goldstein, '-d', 'LineWidth', 1.5, 'MarkerSize', 4);
hold off; % Kết thúc việc vẽ chồng lên nhau

% --- Thiết lập các thuộc tính cho biểu đồ ---
title('So sánh RMSE của các phương pháp theo SNR');
xlabel('SNR (dB)');
ylabel('RMSE (Log Scale)');
% Cập nhật lại thứ tự trong legend cho khớp với thứ tự vẽ
legend('Proposed', 'TIE (FFT-based)', 'Reliability-based', '2D-WLS', 'Goldstein', 'Location', 'northwest');
grid on;
set(gca, 'XDir','reverse');
set(gca, 'FontSize', 12);