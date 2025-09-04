%% compare_noise_models.m
clc; clear; close all;

% --- params ---
M = 256; N = 256;
[X,Y] = meshgrid(1:N,1:M);
sigma_gt = 60;
gt_phase = 1.0 * exp(-((X-N/2).^2 + (Y-M/2).^2)/(2*sigma_gt^2)); % ground truth phase (radians)

% interferogram params
a = 1; b = 0.8;
fx = 0.06; fy = 0.04;
carrier = 2*pi*(fx*X + fy*Y);
hologram_clean = a + b*cos(carrier + gt_phase);

% --- 1) add AWGN directly to phase (phase noise) ---
snr_phase = 20; % not in dB for phase - here we use awgn treating phase as 'signal'
% MATLAB awgn expects signal amplitude; using awgn on phase array:
phase_noisy = awgn(gt_phase, snr_phase, 'measured'); % produces noisy phase array
hologram_from_phaseNoise = a + b*cos(carrier + phase_noisy);
% reconstruct wrapped phase (simple analytic: angle of complex field)
E1 = hologram_from_phaseNoise; % here we use intensity directly, but better to use complex field model
% For fairness later we'll compute wrapped phase via atan2 of R/I after Hilbert-like model:
% simpler: form complex field ~ A*exp(i*(carrier+phase_noisy)) then take angle
complex1 = exp(1i*(carrier + phase_noisy));
wrapped1 = angle(complex1); % wrapped phase from phase-noise model

% --- 2) add AWGN to hologram intensity (sensor noise) ---
snr_holo_db = 20; % dB SNR for intensity
hologram_noisy = awgn(hologram_clean, snr_holo_db, 'measured');
% reconstruct wrapped phase: simulate Fourier filtering + angle extraction simplifed here
% We'll use analytic complex field reconstruction: assume carrier known -> demodulate via multiplication
complex_demod = hologram_noisy .* exp(-1i*carrier); % heterodyne demodulation (simple)
wrapped2 = angle(complex_demod);

% --- 3) add AWGN on complex field (realistic complex noise on quadratures) ---
complex_clean = exp(1i*(carrier + gt_phase));
sigma_q = 0.1; % std of AWGN on real/imag
complex_noisy_q = (real(complex_clean) + sigma_q*randn(size(complex_clean))) ...
                + 1i*(imag(complex_clean) + sigma_q*randn(size(complex_clean)));
wrapped3 = angle(complex_noisy_q);

% --- metrics: difference to ground truth wrapped (wrapToPi(gt_phase)) ---
gt_wrapped = wrapToPi(gt_phase);

err1 = wrapToPi(wrapped1 - gt_wrapped);
err2 = wrapToPi(wrapped2 - gt_wrapped);
err3 = wrapToPi(wrapped3 - gt_wrapped);

rmse1 = sqrt(mean(err1(:).^2));
rmse2 = sqrt(mean(err2(:).^2));
rmse3 = sqrt(mean(err3(:).^2));

fprintf('RMSE (phase-noise) = %.4f rad\n', rmse1);
fprintf('RMSE (hologram-noise) = %.4f rad\n', rmse2);
fprintf('RMSE (complex-quadrature-noise) = %.4f rad\n', rmse3);

% --- show images ---
figure;
subplot(2,3,1), imagesc(gt_phase), title('GT phase'), axis image, colorbar;
subplot(2,3,2), imagesc(wrapped1), title('wrapped (phase noise)'), axis image, colorbar;
subplot(2,3,3), imagesc(err1), title('err (phase noise)'), axis image, colorbar;
subplot(2,3,4), imagesc(wrapped2), title('wrapped (holo noise)'), axis image, colorbar;
subplot(2,3,5), imagesc(err2), title('err (holo noise)'), axis image, colorbar;
subplot(2,3,6), imagesc(err3), title('err (complex quad noise)'), axis image, colorbar;
colormap turbo;
