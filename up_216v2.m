%% 2D Phase Unwrapping Simulation with Surface Type Selection
% This script simulates wrapped phase and performs unwrapping using a noisy estimate.

clear; close all; clc;

%% 1. Parameters and Grid Initialization
fprintf('--- Initializing parameters and grid ---\n');
gridSize = 100;
[xGrid, yGrid] = meshgrid(linspace(-3*pi, 3*pi, gridSize));

% Choose surface type
surface_type = 'sinus'; 
% Options: 'paraboloid', 'sinus', 'cone', 'saddle', 'step', 'spiral', 'mixed', 'zernike'

%% 2. Generate True Phase Surface (Ground Truth)
fprintf('Generating phase surface: %s...\n', surface_type);
truePhase = generatePhaseSurface(xGrid, yGrid, surface_type);
truePhase = truePhase - mean(truePhase(:)); % Normalize mean

%% 3. Create Wrapped Phase with Optional Noise
fprintf('Creating wrapped phase...\n');
noiseLevel = 0;
measuredWrappedPhase = wrapToPi(truePhase + noiseLevel * randn(gridSize));

%% 4. Generate Noisy Phase Estimate
fprintf('Generating noisy phase estimate with sinusoidal error...\n');

% --- Định nghĩa các tham số cho sai lệch dạng sin ---
errorAmplitude = 2.0;   % Biên độ của sai lệch (rad). Giá trị lớn hơn sẽ làm estimate kém chính xác hơn.
errorFreqX = 0.7;       % Tần số không gian của sai lệch theo trục X.
errorFreqY = 1.1;       % Tần số không gian của sai lệch theo trục Y.
randomNoiseLevel = 0;   % Mức nhiễu ngẫu nhiên tần số cao thêm vào estimate.

% --- Tạo sai lệch ---
% Sai lệch chính có dạng sóng sin, mô phỏng lỗi mô hình tần số thấp.
sinusoidalError = errorAmplitude * (sin(errorFreqX * xGrid) + cos(errorFreqY * yGrid));
% Thêm một chút nhiễu ngẫu nhiên để estimate không quá "hoàn hảo".
randomError = (2 * rand(gridSize) - 1) * randomNoiseLevel;
% Tổng hợp sai lệch
totalError = sinusoidalError + randomError;

% Tạo pha ước lượng bằng cách cộng sai lệch vào pha gốc
estimatedPhase = truePhase + totalError;

%% 5. Unwrapping using Estimate
fprintf('Unwrapping using estimate...\n');
[unwrappedPhaseEstimateMethod, kMap] = unwrapUsingEstimate(estimatedPhase, measuredWrappedPhase);

%% 6. MATLAB 2D Unwrap (Sequential per axis)
matlabUnwrappedPhase = unwrap(unwrap(measuredWrappedPhase, [], 1), [], 2);

%% 7. Error Analysis
fprintf('--- Error analysis ---\n');
errorEstimateMethod = truePhase - unwrappedPhaseEstimateMethod;
errorMatlabMethod = truePhase - matlabUnwrappedPhase;

rmsErrorEstimate = sqrt(mean(errorEstimateMethod(:).^2));
rmsErrorMatlab = sqrt(mean(errorMatlabMethod(:).^2));

% Estimate Error
estimateError = estimatedPhase - truePhase;
rmsEstimateError = sqrt(mean(estimateError(:).^2));
fprintf('RMS error (estimate vs ground truth): %.4f rad\n', rmsEstimateError);


fprintf('RMS error (estimate method): %.4f rad\n', rmsErrorEstimate);
fprintf('RMS error (MATLAB unwrap): %.4f rad\n', rmsErrorMatlab);

if rmsErrorMatlab > 0
    improvementPercent = (rmsErrorMatlab - rmsErrorEstimate) / rmsErrorMatlab * 100;
    fprintf('Improvement over MATLAB: %.2f%%\n', improvementPercent);
end

%% 8. Visualization
fprintf('Visualizing results...\n');
figure('Position', [50, 50, 1600, 800]);
t = tiledlayout(2, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
title(t, sprintf('2D Phase Unwrapping - Surface: %s', surface_type), ...
    'FontSize', 16, 'FontWeight', 'bold');

% Plot 1: True Phase
nexttile;
surf(xGrid, yGrid, truePhase); shading interp; colorbar;
title('1. True Phase'); zlabel('Phase (rad)');

% Plot 2: Wrapped Phase
nexttile;
surf(xGrid, yGrid, measuredWrappedPhase); shading interp; colorbar;
title('2. Wrapped Phase'); zlabel('Phase (rad)');

% Plot 3: Estimated Phase
nexttile;
surf(xGrid, yGrid, estimatedPhase); shading interp; colorbar;
title('3. Estimated Phase'); zlabel('Phase (rad)');

% Plot 4: Unwrapped via Estimate
nexttile;
surf(xGrid, yGrid, unwrappedPhaseEstimateMethod); shading interp; colorbar;
title(sprintf('4. Unwrapped (Estimate)\nRMS = %.4f', rmsErrorEstimate));
zlabel('Phase (rad)');

% Plot 5: MATLAB Unwrap
nexttile;
surf(xGrid, yGrid, matlabUnwrappedPhase); shading interp; colorbar;
title(sprintf('5. MATLAB Unwrap\nRMS = %.4f', rmsErrorMatlab));
zlabel('Phase (rad)');

% Plot 6: Error Map
nexttile;
surf(xGrid, yGrid, errorEstimateMethod); shading interp; colorbar;
title('6. Error Map (Estimate Method)'); zlabel('Error (rad)');
figure('Position', [50, 50, 1600, 800]);
t = tiledlayout(2, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
title(t, sprintf('2D Phase Unwrapping - Surface: %s', surface_type), ...
    'FontSize', 16, 'FontWeight', 'bold');

% Plot 7: Estimate Error (3D)
nexttile;
surf(xGrid, yGrid, estimateError); shading interp; colorbar;
title(sprintf('7. Estimate Error (3D)\nRMS = %.4f', rmsEstimateError));
zlabel('Error (rad)');

% Plot 8: Estimate Error (2D Map)
nexttile;
imagesc(estimateError); axis image; colorbar;
title(sprintf('Estimate Error Map (2D View) - RMS = %.4f rad', rmsEstimateError));
xlabel('X'); ylabel('Y');
colormap jet;

nexttile;
imagesc(kMap); axis image; colorbar;
title(sprintf('K map'));
xlabel('X'); ylabel('Y');


%% --- Local Functions ---

function [unwrappedPhase, kMap, wrappedEstimate] = unwrapUsingEstimate(estimatedPhase, wrappedPhase)
    wrappedEstimate = wrapToPi(estimatedPhase);
    kMap = round((estimatedPhase - wrappedEstimate) / (2*pi));
    unwrappedPhase = wrappedPhase + 2*pi * kMap;
end

function wrapped = wrapToPi(phase)
    wrapped = mod(phase + pi, 2*pi) - pi;
end

function phase = generatePhaseSurface(X, Y, type)
    R = sqrt(X.^2 + Y.^2);
    theta = atan2(Y, X);
    switch lower(type)
        case 'paraboloid'
            phase = 0.3 * (X.^2 + Y.^2);
        case 'sinus'
            phase = 6*sin(X) + 6*cos(Y);
        case 'cone'
            phase = R;
        case 'saddle'
            phase = X.^2 - Y.^2;
        case 'step'
            phase = double(X > 0);  % Discontinuous step
        case 'spiral'
            phase = R + theta;
        case 'mixed'
            phase = 0.1*(X.^2 + Y.^2) + sin(3*X).*cos(3*Y);
        case 'zernike'
            Rn = R / max(X(:)); % Normalize to unit circle
            phase = Rn.^2 .* cos(theta); % simple Zernike mode
        otherwise
            error('Unknown phase surface type: %s', type);
    end
end
