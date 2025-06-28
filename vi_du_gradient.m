%% So sánh PhaseGradient với gradient() của MATLAB
clear; clc; close all;

%% 1. Tạo dữ liệu test
% Test case 1: Phase data liên tục
[X, Y] = meshgrid(1:20, 1:20);
phase_smooth = 0.5 * X + 0.3 * Y + 0.1 * sin(X/2) .* cos(Y/3);

% Test case 2: Phase data có discontinuity
phase_with_jumps = phase_smooth;
phase_with_jumps(:, 10:end) = phase_with_jumps(:, 10:end) + 2*pi; % Tạo jump

% Test case 3: Phase data thực tế (ví dụ interferometry)
phase_realistic = 2*pi * (sin(X/5) .* cos(Y/5)) + pi * rand(20,20) * 0.1;
phase_realistic = wrapToPi(phase_realistic); % Wrap về [-pi, pi]

%% 2. Tính gradient bằng các phương pháp khác nhau

test_cases = {phase_smooth, phase_with_jumps, phase_realistic};
case_names = {'Smooth Phase', 'Phase with Jumps', 'Realistic Phase'};

for test_idx = 1:length(test_cases)
    current_phase = test_cases{test_idx};
    
    fprintf('\n=== %s ===\n', case_names{test_idx});
    
    % Method 1: MATLAB gradient() - không xử lý phase unwrapping
    [gy_matlab, gx_matlab] = gradient(current_phase);
    
    % Method 2: PhaseGradient - có xử lý phase unwrapping
    [gx_phase, gy_phase] = PhaseGradient(current_phase);
    
    % Method 3: MATLAB gradient() với unwrap trước
    phase_unwrapped = unwrap(unwrap(current_phase, [], 1), [], 2); % Unwrap theo cả 2 chiều
    [gy_unwrapped, gx_unwrapped] = gradient(phase_unwrapped);
    
    % Method 4: Manual central difference (reference)
    [gx_manual, gy_manual] = manual_gradient(current_phase);
    
    %% 3. So sánh kết quả
    fprintf('Gradient X comparison:\n');
    fprintf('Max difference (PhaseGradient vs MATLAB): %.6f\n', ...
        max(abs(gx_phase(:) - gx_matlab(:))));
    fprintf('Max difference (PhaseGradient vs Unwrapped): %.6f\n', ...
        max(abs(gx_phase(:) - gx_unwrapped(:))));
    fprintf('Max difference (MATLAB vs Manual): %.6f\n', ...
        max(abs(gx_matlab(:) - gx_manual(:))));
    
    fprintf('Gradient Y comparison:\n');
    fprintf('Max difference (PhaseGradient vs MATLAB): %.6f\n', ...
        max(abs(gy_phase(:) - gy_matlab(:))));
    fprintf('Max difference (PhaseGradient vs Unwrapped): %.6f\n', ...
        max(abs(gy_phase(:) - gy_unwrapped(:))));
    fprintf('Max difference (MATLAB vs Manual): %.6f\n', ...
        max(abs(gy_matlab(:) - gy_manual(:))));
    
    %% 4. Visualize kết quả
    figure('Position', [100, 100, 1400, 800]);
    sgtitle(sprintf('Gradient Comparison: %s', case_names{test_idx}));
    
    % Original phase
    subplot(3,4,1);
    imagesc(current_phase); colorbar; title('Original Phase');
    colormap(gca, 'hsv');
    
    % Gradient X comparisons
    subplot(3,4,2);
    imagesc(gx_matlab); colorbar; title('MATLAB gradient() - X');
    
    subplot(3,4,3);
    imagesc(gx_phase); colorbar; title('PhaseGradient - X');
    
    subplot(3,4,4);
    imagesc(gx_unwrapped); colorbar; title('MATLAB (unwrapped) - X');
    
    % Gradient Y comparisons
    subplot(3,4,6);
    imagesc(gy_matlab); colorbar; title('MATLAB gradient() - Y');
    
    subplot(3,4,7);
    imagesc(gy_phase); colorbar; title('PhaseGradient - Y');
    
    subplot(3,4,8);
    imagesc(gy_unwrapped); colorbar; title('MATLAB (unwrapped) - Y');
    
    % Difference maps
    subplot(3,4,10);
    imagesc(abs(gx_phase - gx_matlab)); colorbar; 
    title('|PhaseGrad - MATLAB| X');
    
    subplot(3,4,11);
    imagesc(abs(gy_phase - gy_matlab)); colorbar; 
    title('|PhaseGrad - MATLAB| Y');
    
    subplot(3,4,12);
    imagesc(abs(gx_phase - gx_unwrapped)); colorbar; 
    title('|PhaseGrad - Unwrapped| X');
    
    %% 5. Thống kê chi tiết
    fprintf('\nStatistical Analysis:\n');
    fprintf('Gradient X - Mean absolute difference:\n');
    fprintf('  PhaseGradient vs MATLAB: %.6f\n', mean(abs(gx_phase(:) - gx_matlab(:))));
    fprintf('  PhaseGradient vs Unwrapped: %.6f\n', mean(abs(gx_phase(:) - gx_unwrapped(:))));
    
    fprintf('Gradient Y - Mean absolute difference:\n');
    fprintf('  PhaseGradient vs MATLAB: %.6f\n', mean(abs(gy_phase(:) - gy_matlab(:))));
    fprintf('  PhaseGradient vs Unwrapped: %.6f\n', mean(abs(gy_phase(:) - gy_unwrapped(:))));
    
    % Correlation analysis
    corrx_pm = corrcoef(gx_phase(:), gx_matlab(:));
    corrx_pu = corrcoef(gx_phase(:), gx_unwrapped(:));
    corry_pm = corrcoef(gy_phase(:), gy_matlab(:));
    corry_pu = corrcoef(gy_phase(:), gy_unwrapped(:));
    
    fprintf('Correlation coefficients:\n');
    fprintf('  Gradient X (PhaseGrad vs MATLAB): %.4f\n', corrx_pm(1,2));
    fprintf('  Gradient X (PhaseGrad vs Unwrapped): %.4f\n', corrx_pu(1,2));
    fprintf('  Gradient Y (PhaseGrad vs MATLAB): %.4f\n', corry_pm(1,2));
    fprintf('  Gradient Y (PhaseGrad vs Unwrapped): %.4f\n', corry_pu(1,2));
end

%% 6. Test với mask
fprintf('\n=== Testing with Mask ===\n');
test_phase = phase_realistic;
mask = zeros(size(test_phase));
mask(5:15, 5:15) = 1; % ROI mask

% With mask
[gx_masked, gy_masked] = PhaseGradient(test_phase, mask);

% Without mask  
[gx_full, gy_full] = PhaseGradient(test_phase);

fprintf('Mask effect - values inside mask region:\n');
roi_indices = find(mask);
fprintf('Mean difference in ROI (X): %.6f\n', ...
    mean(abs(gx_masked(roi_indices) - gx_full(roi_indices))));
fprintf('Mean difference in ROI (Y): %.6f\n', ...
    mean(abs(gy_masked(roi_indices) - gy_full(roi_indices))));

%% Helper Functions

function [gx, gy] = manual_gradient(img)
    % Manual implementation của gradient sử dụng central difference
    [m, n] = size(img);
    gx = zeros(m, n);
    gy = zeros(m, n);
    
    % Gradient X
    gx(:, 1) = img(:, 2) - img(:, 1);           % Forward difference
    gx(:, n) = img(:, n) - img(:, n-1);         % Backward difference  
    gx(:, 2:n-1) = (img(:, 3:n) - img(:, 1:n-2)) / 2; % Central difference
    
    % Gradient Y
    gy(1, :) = img(2, :) - img(1, :);           % Forward difference
    gy(m, :) = img(m, :) - img(m-1, :);         % Backward difference
    gy(2:m-1, :) = (img(3:m, :) - img(1:m-2, :)) / 2; % Central difference
end

function [gradient_x, gradient_y] = PhaseGradient(IM_phase, varargin)
    % Phiên bản đơn giản của PhaseGradient function để test
    [r_dim, c_dim] = size(IM_phase);
    
    % Xử lý mask (simplified)
    if nargin >= 2
        IM_mask = varargin{1};
        [maskrows, maskcols] = find(IM_mask);
        minrow = max(1, min(maskrows) - 1);
        maxrow = min(r_dim, max(maskrows) + 1);
        mincol = max(1, min(maskcols) - 1);
        maxcol = min(c_dim, max(maskcols) + 1);
        
        IM_phase = IM_phase(minrow:maxrow, mincol:maxcol);
    end
    
    [dimx, dimy] = size(IM_phase);
    
    % Gradient X
    gradient_x = zeros(dimx, dimy);
    
    % First column
    p = unwrap([IM_phase(:,1) IM_phase(:,2)], [], 2);
    gradient_x(:,1) = (p(:,2) - IM_phase(:,1)) / 2;
    
    % Last column  
    p = unwrap([IM_phase(:,dimy-1) IM_phase(:,dimy)], [], 2);
    gradient_x(:,dimy) = (p(:,2) - IM_phase(:,dimy-1)) / 2;
    
    % Middle columns
    for i = 2:dimy-1
        p = unwrap([IM_phase(:,i-1) IM_phase(:,i+1)], [], 2);
        gradient_x(:,i) = (p(:,2) - IM_phase(:,i-1)) / 2; % Fixed: /2 instead of /3
    end
    
    % Gradient Y
    gradient_y = zeros(dimx, dimy);
    
    % First row
    q = unwrap([IM_phase(1,:)' IM_phase(2,:)'], [], 2);
    gradient_y(1,:) = (q(:,2)' - IM_phase(1,:)) / 2;
    
    % Last row
    p = unwrap([IM_phase(dimx-1,:)' IM_phase(dimx,:)'], [], 2);
    gradient_y(dimx,:) = (p(:,2)' - IM_phase(dimx-1,:)) / 2;
    
    % Middle rows
    for i = 2:dimx-1
        q = unwrap([IM_phase(i-1,:)' IM_phase(i+1,:)'], [], 2);
        gradient_y(i,:) = (q(:,2)' - IM_phase(i-1,:)) / 2; % Fixed: /2 instead of /3
    end
    
    % Pad back if mask was used
    if nargin >= 2
        temp_gx = zeros(r_dim, c_dim);
        temp_gy = zeros(r_dim, c_dim);
        temp_gx(minrow:maxrow, mincol:maxcol) = gradient_x;
        temp_gy(minrow:maxrow, mincol:maxcol) = gradient_y;
        gradient_x = temp_gx;
        gradient_y = temp_gy;
    end
end