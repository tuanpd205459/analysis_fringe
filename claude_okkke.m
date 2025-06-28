
% Fast 2D Phase Unwrapping Algorithm Based on Reliability-Guided Sorting   
% https://claude.ai/public/artifacts/1c40bf8a-3472-4097-9889-b811035b79b6
%%
clc, clear, close all;
%%
%% Phase Unwrapping cho bề mặt 3D
% Mô phỏng unwrapping trên dữ liệu 2D (như ảnh interferometry)
clear; clc; close all;

%% 1. Tạo dữ liệu mẫu 3D
fprintf('=== TẠO DỮ LIỆU SURFACE 3D ===\n');

% Kích thước surface
nx = 50; ny = 40;
[X, Y] = meshgrid(1:nx, 1:ny);

% Tạo true phase surface phức tạp
fprintf('Tạo true phase surface...\n');
true_phase = create_complex_phase_surface(X, Y);

% Tạo wrapped phase
wrapped_phase = angle(exp(1i * true_phase));

% Tạo estimated phase (có nhiễu và một số vùng sai lệch)
estimated_phase = create_estimated_phase(true_phase, X, Y);

% Thêm nhiễu
noise_level = 0.2;
estimated_phase = estimated_phase + noise_level * randn(size(estimated_phase));

fprintf('Kích thước surface: %dx%d\n', size(true_phase));
fprintf('Range true phase: [%.2f, %.2f]\n', min(true_phase(:)), max(true_phase(:)));

%% 2. Thực hiện 3D Phase Unwrapping
fprintf('\n=== THỰC HIỆN 3D PHASE UNWRAPPING ===\n');

% Cấu hình tham số
options.confidence_weight = 0.8;
options.gradient_threshold = pi * 0.3;
options.smooth_kernel_size = 3;
options.max_iterations = 50;

% Unwrap surface
[unwrapped_surface, quality_map, process_info] = fast_phase_unwrapping_2d(...
    wrapped_phase, estimated_phase);

%% 3. Visualize kết quả
fprintf('\n=== VISUALIZATION ===\n');
visualize_results(wrapped_phase, estimated_phase, ...
                       unwrapped_surface, quality_map, process_info);

%% 4. Đánh giá chất lượng
fprintf('\n=== ĐÁNH GIÁ CHẤT LƯỢNG ===\n');
evaluate_3d_results(true_phase, estimated_phase, unwrapped_surface);


function [unwrapped_phase, reliability_map, processing_order] = fast_phase_unwrapping_2d(wrapped_phase, estimated_phase, varargin)
%FAST_PHASE_UNWRAPPING_2D Fast 2D phase unwrapping based on reliability-guided sorting
%
% Syntax:
%   [unwrapped_phase, reliability_map, processing_order] = ...
%       fast_phase_unwrapping_2d(wrapped_phase, estimated_phase)
%   [...] = fast_phase_unwrapping_2d(..., 'PropertyName', PropertyValue)
%
% Input:
%   wrapped_phase    - 2D matrix of wrapped phase values [-π, π]
%   estimated_phase  - 2D matrix of estimated phase values (smooth approximation)
%
% Optional Parameters:
%   'WindowSize'           - Window size for reliability calculation (default: 5)
%   'ReliabilityThreshold' - Threshold for reliability-based processing (default: 0.5)
%   'WeightGradient'       - Weight for gradient consistency (default: 0.3)
%   'WeightVariance'       - Weight for variance reliability (default: 0.2)
%   'WeightCoherence'      - Weight for phase coherence (default: 0.3)
%   'WeightAgreement'      - Weight for agreement reliability (default: 0.2)
%   'PostProcessPasses'    - Number of post-processing passes (default: 3)
%   'ShowProgress'         - Display progress bar (default: true)
%   'Visualization'        - Show visualization (default: false)
%
% Output:
%   unwrapped_phase    - 2D matrix of unwrapped phase values
%   reliability_map    - 2D matrix of reliability values [0, 1]
%   processing_order   - 2D matrix showing processing order
%
% Example:
%   % Generate test data
%   [X, Y] = meshgrid(1:64, 1:64);
%   true_phase = 2*pi*(sin(X/10) + cos(Y/10));
%   wrapped_phase = angle(exp(1i*true_phase));
%   estimated_phase = imgaussfilt(true_phase, 2) + 0.5*randn(size(true_phase));
%   
%   % Run phase unwrapping
%   [unwrapped, reliability] = fast_phase_unwrapping_2d(wrapped_phase, estimated_phase, ...
%       'Visualization', true, 'ShowProgress', true);
%
% Author: Advanced Phase Processing Team
% Date: 2025

%% Input validation and parameter parsing
if nargin < 2
    error('fast_phase_unwrapping_2d:NotEnoughInputs', ...
        'At least two input arguments required: wrapped_phase and estimated_phase');
end

% Validate input dimensions
if ~isequal(size(wrapped_phase), size(estimated_phase))
    error('fast_phase_unwrapping_2d:DimensionMismatch', ...
        'wrapped_phase and estimated_phase must have the same dimensions');
end

[M, N] = size(wrapped_phase);

% Parse optional parameters
p = inputParser;
addParameter(p, 'WindowSize', 5, @(x) isscalar(x) && x > 0 && mod(x,2) == 1);
addParameter(p, 'ReliabilityThreshold', 0.5, @(x) isscalar(x) && x >= 0 && x <= 1);
addParameter(p, 'WeightGradient', 0.3, @(x) isscalar(x) && x >= 0 && x <= 1);
addParameter(p, 'WeightVariance', 0.2, @(x) isscalar(x) && x >= 0 && x <= 1);
addParameter(p, 'WeightCoherence', 0.3, @(x) isscalar(x) && x >= 0 && x <= 1);
addParameter(p, 'WeightAgreement', 0.2, @(x) isscalar(x) && x >= 0 && x <= 1);
addParameter(p, 'PostProcessPasses', 3, @(x) isscalar(x) && x >= 0);
addParameter(p, 'ShowProgress', true, @islogical);
addParameter(p, 'Visualization', false, @islogical);

parse(p, varargin{:});
params = p.Results;

% Validate weights sum to 1
weight_sum = params.WeightGradient + params.WeightVariance + ...
             params.WeightCoherence + params.WeightAgreement;
if abs(weight_sum - 1) > 1e-6
    warning('fast_phase_unwrapping_2d:WeightSum', ...
        'Reliability weights do not sum to 1. Normalizing weights.');
    params.WeightGradient = params.WeightGradient / weight_sum;
    params.WeightVariance = params.WeightVariance / weight_sum;
    params.WeightCoherence = params.WeightCoherence / weight_sum;
    params.WeightAgreement = params.WeightAgreement / weight_sum;
end

%% Initialize progress tracking
if params.ShowProgress
    fprintf('Fast 2D Phase Unwrapping Algorithm\n');
    fprintf('==================================\n');
    fprintf('Image size: %d x %d\n', M, N);
    fprintf('Total pixels: %d\n', M*N);
    fprintf('\n');
end

tic; % Start timing

%% Step 1: Calculate Reliability Map
if params.ShowProgress
    fprintf('Step 1/5: Calculating reliability map...\n');
end

reliability_map = calculate_reliability_map(wrapped_phase, estimated_phase, params);

if params.ShowProgress
    avg_reliability = mean(reliability_map(:));
    fprintf('          Average reliability: %.3f\n', avg_reliability);
end

%% Step 2: Create Priority Queue
if params.ShowProgress
    fprintf('Step 2/5: Creating priority queue...\n');
end

[pixel_queue, ~] = create_priority_queue(reliability_map, M, N);

if params.ShowProgress
    fprintf('          Pixels sorted by priority\n');
end

%% Step 3: Initialize Processing Arrays
if params.ShowProgress
    fprintf('Step 3/5: Initializing processing arrays...\n');
end

unwrapped_phase = zeros(M, N);
processed = false(M, N);
quality_score = zeros(M, N);
processing_order = zeros(M, N);

%% Step 4: Reliability-guided Unwrapping
if params.ShowProgress
    fprintf('Step 4/5: Performing reliability-guided unwrapping...\n');
    progress_bar = waitbar(0, 'Processing pixels...');
end

total_pixels = length(pixel_queue);
processed_count = 0;

for idx = 1:total_pixels
    i = pixel_queue(idx, 1);
    j = pixel_queue(idx, 2);
    
    if processed(i, j)
        continue;
    end
    
    % Find processed neighbors as references
    [references, ref_weights] = find_processed_neighbors(i, j, processed, ...
        unwrapped_phase, quality_score, M, N);
    
    if ~isempty(references)
        % Multi-candidate unwrapping with weighted references
        [best_unwrapped, pixel_quality] = unwrap_with_multiple_references(...
            wrapped_phase(i,j), estimated_phase(i,j), references, ref_weights, ...
            reliability_map(i,j));
    else
        % Seed pixel - use estimated phase
        best_unwrapped = estimated_phase(i, j);
        pixel_quality = reliability_map(i, j);
    end
    
    % Store results
    unwrapped_phase(i, j) = best_unwrapped;
    quality_score(i, j) = pixel_quality;
    processed(i, j) = true;
    processing_order(i, j) = idx / total_pixels;
    processed_count = processed_count + 1;
    
    % Update progress
    if params.ShowProgress && mod(idx, 100) == 0
        waitbar(idx/total_pixels, progress_bar, ...
            sprintf('Processing pixels... %d/%d (%.1f%%)', ...
            processed_count, total_pixels, 100*idx/total_pixels));
    end
end

if params.ShowProgress
    close(progress_bar);
    fprintf('          Processed %d pixels\n', processed_count);
end

%% Step 5: Post-processing Optimization
if params.ShowProgress && params.PostProcessPasses > 0
    fprintf('Step 5/5: Post-processing optimization...\n');
end

unwrapped_phase = post_process_optimization(unwrapped_phase, quality_score, ...
    params.ReliabilityThreshold, params.PostProcessPasses, params.ShowProgress);

%% Display Results
total_time = toc;
if params.ShowProgress
    fprintf('\nProcessing completed successfully!\n');
    fprintf('Total processing time: %.2f seconds\n', total_time);
    fprintf('Average processing rate: %.0f pixels/sec\n', total_pixels/total_time);
end

%% Visualization
if params.Visualization
    visualize_results(wrapped_phase, estimated_phase, unwrapped_phase, ...
        reliability_map, processing_order);
end

end

%% Helper Functions

function reliability_map = calculate_reliability_map(wrapped_phase, estimated_phase, params)
%CALCULATE_RELIABILITY_MAP Calculate multi-criteria reliability map

[M, N] = size(wrapped_phase);
window_size = params.WindowSize;
half_window = floor(window_size / 2);

% Initialize reliability components
grad_reliability = zeros(M, N);
var_reliability = zeros(M, N);
coherence_reliability = zeros(M, N);
agreement_reliability = zeros(M, N);

% Calculate gradients
[grad_wrap_x, grad_wrap_y] = gradient(wrapped_phase);
[grad_est_x, grad_est_y] = gradient(estimated_phase);

for i = 1:M
    for j = 1:N
        % 1. Gradient Consistency Reliability
        grad_diff = sqrt((grad_wrap_x(i,j) - grad_est_x(i,j))^2 + ...
                         (grad_wrap_y(i,j) - grad_est_y(i,j))^2);
        grad_reliability(i,j) = exp(-grad_diff * 5);
        
        % 2. Variance-based Reliability
        var_reliability(i,j) = calculate_variance_reliability(wrapped_phase, i, j, M, N);
        
        % 3. Phase Coherence Reliability
        coherence_reliability(i,j) = calculate_phase_coherence(wrapped_phase, ...
            estimated_phase, i, j, half_window, M, N);
        
        % 4. Agreement Reliability
        agreement_reliability(i,j) = calculate_agreement_reliability(...
            wrapped_phase(i,j), estimated_phase(i,j));
    end
end

% Combine reliabilities with weights
reliability_map = params.WeightGradient * grad_reliability + ...
                  params.WeightVariance * var_reliability + ...
                  params.WeightCoherence * coherence_reliability + ...
                  params.WeightAgreement * agreement_reliability;

% Ensure values are in [0, 1]
reliability_map = max(0, min(1, reliability_map));

end

function var_reliability = calculate_variance_reliability(wrapped_phase, i, j, M, N)
%CALCULATE_VARIANCE_RELIABILITY Calculate variance-based reliability

% Define 3x3 neighborhood
local_values = [];
for di = -1:1
    for dj = -1:1
        ni = i + di;
        nj = j + dj;
        if ni >= 1 && ni <= M && nj >= 1 && nj <= N
            local_values = [local_values, wrapped_phase(ni, nj)];
        end
    end
end

if length(local_values) > 1
    local_var = var(local_values);
    var_reliability = exp(-local_var * 3);
else
    var_reliability = 0.5;
end

end

function coherence_reliability = calculate_phase_coherence(wrapped_phase, estimated_phase, i, j, half_window, M, N)
%CALCULATE_PHASE_COHERENCE Calculate phase coherence reliability

coherence_sum = 0;
count = 0;

for di = -half_window:half_window
    for dj = -half_window:half_window
        ni = i + di;
        nj = j + dj;
        
        if ni >= 1 && ni <= M && nj >= 1 && nj <= N && (di ~= 0 || dj ~= 0)
            distance = sqrt(di^2 + dj^2);
            if distance <= half_window
                phase_diff = abs(wrapped_phase(ni,nj) - wrapped_phase(i,j));
                expected_diff = abs(estimated_phase(ni,nj) - estimated_phase(i,j));
                
                agreement = exp(-abs(phase_diff - expected_diff));
                coherence_sum = coherence_sum + agreement / distance;
                count = count + 1;
            end
        end
    end
end

if count > 0
    coherence_reliability = coherence_sum / count;
else
    coherence_reliability = 0.5;
end

end

function agreement_reliability = calculate_agreement_reliability(wrapped_val, estimated_val)
%CALCULATE_AGREEMENT_RELIABILITY Calculate agreement between wrapped and estimated

% Normalize both to [0, 2π]
wrapped_norm = mod(wrapped_val + pi, 2*pi);
estimated_norm = mod(estimated_val + pi, 2*pi);

% Calculate circular difference
diff = min(abs(wrapped_norm - estimated_norm), ...
           2*pi - abs(wrapped_norm - estimated_norm));

agreement_reliability = exp(-diff);

end

function [pixel_queue, priority_scores] = create_priority_queue(reliability_map, M, N)
%CREATE_PRIORITY_QUEUE Create priority queue based on reliability and position

pixel_list = zeros(M*N, 2);
priority_scores = zeros(M*N, 1);
idx = 1;

center_i = M / 2;
center_j = N / 2;
max_dist = sqrt(M^2 + N^2);

for i = 1:M
    for j = 1:N
        % Distance from center (normalized)
        dist_from_center = sqrt((i - center_i)^2 + (j - center_j)^2) / max_dist;
        
        % Priority combines reliability with slight center bias
        priority = reliability_map(i,j) * (1 - 0.1 * dist_from_center);
        
        pixel_list(idx,:) = [i, j];
        priority_scores(idx) = priority;
        idx = idx + 1;
    end
end

% Sort by priority (descending)
[~, sort_idx] = sort(priority_scores, 'descend');
pixel_queue = pixel_list(sort_idx, :);
priority_scores = priority_scores(sort_idx);

end

function [references, weights] = find_processed_neighbors(i, j, processed, unwrapped_phase, quality_score, M, N)
%FIND_PROCESSED_NEIGHBORS Find processed neighbors for reference

references = [];
weights = [];

% Search in expanding neighborhood (up to 2 pixels away)
for search_radius = 1:2
    for di = -search_radius:search_radius
        for dj = -search_radius:search_radius
            if di == 0 && dj == 0, continue; end % Skip the center pixel
            
            ni = i + di;
            nj = j + dj;
            
            if ni >= 1 && ni <= M && nj >= 1 && nj <= N && processed(ni, nj)
                
                distance = sqrt(di^2 + dj^2);
                weight = quality_score(ni, nj) / distance;
                
                references = [references; unwrapped_phase(ni, nj)];
                weights = [weights; weight];
            end
        end
    end
    
    % If we found enough references, stop searching
    if length(references) >= 3
        break;
    end
end

% Normalize weights
if ~isempty(weights)
    sum_weights = sum(weights);
    if sum_weights > 0
        weights = weights / sum_weights;
    else
        % Avoid division by zero if all weights are somehow zero
        weights = ones(size(weights)) / length(weights);
    end
end

end



function [best_unwrapped, pixel_quality] = unwrap_with_multiple_references(wrapped_val, estimated_val, references, ref_weights, reliability)
%UNWRAP_WITH_MULTIPLE_REFERENCES Unwrap using multiple reference points

if isempty(references)
    best_unwrapped = estimated_val;
    pixel_quality = reliability;
    return;
end

% Calculate weighted reference
weighted_ref = sum(references .* ref_weights);

% Generate unwrapping candidates
candidates = zeros(5,1);
scores = zeros(5,1);

for k_idx = 1:5
    k = k_idx - 3; % k ranges from -2 to 2
    candidate = wrapped_val + k * 2 * pi;
    
    % Score based on distance from reference and estimated
    ref_diff = abs(candidate - weighted_ref);
    est_diff = abs(candidate - estimated_val);
    
    score = 1 / (1 + ref_diff + 0.5 * est_diff);
    
    candidates(k_idx) = candidate;
    scores(k_idx) = score;
end

% Select best candidate
[~, best_idx] = max(scores);
best_candidate = candidates(best_idx);

% Weighted combination of best candidate and estimated
alpha = reliability;
best_unwrapped = alpha * best_candidate + (1 - alpha) * estimated_val;

% Calculate pixel quality
if ~isempty(ref_weights)
    max_ref_quality = max(ref_weights) * length(ref_weights);
else
    max_ref_quality = 0;
end
pixel_quality = reliability * min(1, max_ref_quality);

end

function optimized_phase = post_process_optimization(unwrapped_phase, quality_score, threshold, num_passes, show_progress)
%POST_PROCESS_OPTIMIZATION Multi-pass optimization of unwrapped phase

optimized_phase = unwrapped_phase;
[M, N] = size(unwrapped_phase);

if show_progress && num_passes > 0
    fprintf('          Performing %d optimization passes...\n', num_passes);
end

for pass = 1:num_passes
    if show_progress
        fprintf('          Pass %d/%d\n', pass, num_passes);
    end
    
    for i = 2:M-1
        for j = 2:N-1
            if quality_score(i,j) < threshold
                % Find high-quality neighbors
                neighbor_values = [];
                neighbor_weights = [];
                
                for di = -1:1
                    for dj = -1:1
                        ni = i + di;
                        nj = j + dj;
                        
                        if quality_score(ni,nj) > threshold
                            neighbor_values = [neighbor_values; optimized_phase(ni,nj)];
                            neighbor_weights = [neighbor_weights; quality_score(ni,nj)];
                        end
                    end
                end
                
                if ~isempty(neighbor_values)
                    % Weighted average of high-quality neighbors
                    total_weight = sum(neighbor_weights);
                    if total_weight > 0
                        weighted_avg = sum(neighbor_values .* neighbor_weights) / total_weight;
                        
                        % Soft update to avoid artifacts
                        alpha = 0.3;
                        optimized_phase(i,j) = alpha * weighted_avg + (1-alpha) * optimized_phase(i,j);
                        
                        % Slightly improve quality
                        quality_score(i,j) = min(threshold, quality_score(i,j) * 1.1);
                    end
                end
            end
        end
    end
end

end

function visualize_results(wrapped_phase, estimated_phase, unwrapped_phase, reliability_map, processing_order)
%VISUALIZE_RESULTS Display comprehensive visualization of results

figure('Name', 'Fast 2D Phase Unwrapping Results', 'Position', [100, 100, 1200, 800], 'Color', 'w');
sgtitle('2D Phase Unwrapping Algorithm Results', 'FontSize', 16, 'FontWeight', 'bold');

% 1. Wrapped Phase
subplot(2, 3, 1);
imagesc(wrapped_phase);
axis image;
colorbar;
title('1. Wrapped Phase (Input)');
xlabel('X-axis');
ylabel('Y-axis');
colormap(gca, hsv);

% 2. Estimated Phase
subplot(2, 3, 2);
imagesc(estimated_phase);
axis image;
colorbar;
title('2. Estimated Phase (Input)');
xlabel('X-axis');
ylabel('Y-axis');
colormap(gca, parula);

% 3. Unwrapped Phase (2D)
subplot(2, 3, 3);
imagesc(unwrapped_phase);
axis image;
colorbar;
title('3. Unwrapped Phase (Result)');
xlabel('X-axis');
ylabel('Y-axis');
colormap(gca, parula);

% 4. Reliability Map
subplot(2, 3, 4);
imagesc(reliability_map, [0 1]);
axis image;
colorbar;
title('4. Reliability Map');
xlabel('X-axis');
ylabel('Y-axis');
colormap(gca, hot);

% 5. Processing Order
subplot(2, 3, 5);
imagesc(processing_order);
axis image;
colorbar;
title('5. Processing Order (High-reliability first)');
xlabel('X-axis');
ylabel('Y-axis');
colormap(gca, jet);

% 6. Unwrapped Phase (3D Surface)
subplot(2, 3, 6);
surf(unwrapped_phase, 'EdgeColor', 'none', 'FaceAlpha', 0.8);
axis tight;
colorbar;
title('6. Unwrapped Phase (3D Surface)');
xlabel('X');
ylabel('Y');
zlabel('Unwrapped Phase');
grid on;
view(30, 45); % Isometric view
colormap(gca, parula);

end



%%


function true_phase = create_complex_phase_surface(X, Y)
    % Tạo surface pha phức tạp với nhiều đặc trưng
    
    [ny, nx] = size(X);
    
    % Component 1: Ramped surface (xu hướng chính)
    ramp_x = 0.3 * X;
    ramp_y = 0.2 * Y;
    
    % Component 2: Gaussian peaks (các đỉnh)
%     peak1 = 8 * exp(-((X-15).^2 + (Y-10).^2)/20);
%     peak2 = 6 * exp(-((X-35).^2 + (Y-25).^2)/30);
%     peak3 = -4 * exp(-((X-25).^2 + (Y-35).^2)/25);
    
    % Component 3: Sinusoidal waves (sóng)
    wave1 = 3 * sin(0.4 * X) .* cos(0.3 * Y);
    wave2 = 2 * sin(0.2 * (X + Y));
    wave2 = 3 * sin(0.2* Y) .* cos(0.2* X);
    
%     Component 4: Sharp discontinuity (bất liên tục)
    discontinuity = zeros(size(X));
    discontinuity(Y > 20 & X < 30) = 4;
    
    % Kết hợp tất cả
    true_phase =   wave2 + discontinuity;
    
    fprintf('   - Created surface with ramps, peaks, waves, and discontinuities\n');
end

function estimated_phase = create_estimated_phase(true_phase, X, Y)
    % Tạo estimated phase có một số vùng sai lệch
    
    estimated_phase = true_phase;
    
    % Thêm sai lệch có hệ thống ở một số vùng
%     error_region1 = (X > 20 & X < 35 & Y > 15 & Y < 25);
%     estimated_phase(error_region1) = estimated_phase(error_region1) + 3;
    
%     error_region2 = (X > 40 & Y < 15);
%     estimated_phase(error_region2) = estimated_phase(error_region2) - 2;
%     
    % Thêm smooth error
    [ny, nx] = size(X);
    smooth_error = 2 * sin(0.1 * X) .* sin(0.15 * Y);
    estimated_phase = estimated_phase + smooth_error;

    
    fprintf('   - Added systematic errors in specific regions\n');
end

function [unwrapped_surface, quality_map, process_info] = unwrap_3d_surface(...
    wrapped_phase, estimated_phase, options)
    
    fprintf('Bắt đầu 3D surface unwrapping...\n');
    
    [ny, nx] = size(wrapped_phase);
    
    % Khởi tạo
    unwrapped_surface = zeros(size(wrapped_phase));
    quality_map = zeros(size(wrapped_phase));
    confidence_map = zeros(size(wrapped_phase));
    
    % Tính gradient trong cả hai hướng
    [grad_x_wrapped, grad_y_wrapped] = gradient(wrapped_phase);
    [grad_x_estimated, grad_y_estimated] = gradient(estimated_phase);
    
    % Phát hiện và sửa chữa phase jumps
    fprintf('Phát hiện phase jumps trong 2D...\n');
    [grad_x_corrected, grad_y_corrected, jump_info] = fix_2d_phase_jumps(...
        grad_x_wrapped, grad_y_wrapped, grad_x_estimated, grad_y_estimated, options);
    
    % Tính confidence map
    fprintf('Tính confidence map...\n');
    confidence_map = calculate_2d_confidence(grad_x_corrected, grad_y_corrected, ...
                                           grad_x_estimated, grad_y_estimated);
    
    % Iterative unwrapping với path-following
    fprintf('Thực hiện iterative unwrapping...\n');
    unwrapped_surface = perform_2d_unwrapping(wrapped_phase, ...
        grad_x_corrected, grad_y_corrected, estimated_phase, confidence_map, options);
    
    % Tính quality map
    quality_map = calculate_quality_map(unwrapped_surface, estimated_phase, confidence_map);
    
    % Thông tin quá trình
    process_info.num_jumps_x = jump_info.num_jumps_x;
    process_info.num_jumps_y = jump_info.num_jumps_y;
    process_info.avg_confidence = mean(confidence_map(:));
    process_info.convergence = 'completed';
    
    fprintf('   - Phát hiện %d jumps theo X, %d jumps theo Y\n', ...
            jump_info.num_jumps_x, jump_info.num_jumps_y);
    fprintf('   - Confidence trung bình: %.3f\n', process_info.avg_confidence);
end


function evaluate_3d_results(true_phase, estimated_phase, unwrapped_surface) 
    %EVALUATE_3D_RESULTS Tính toán, hiển thị chỉ số và trực quan hóa 3D.
    %   Hàm này thực hiện hai nhiệm vụ:
    %   1. In ra các chỉ số đánh giá (RMSE, Max Error) cho kết quả mở pha.
    %   2. Tạo một cửa sổ hình ảnh mới để hiển thị 4 bề mặt 3D so sánh.

    %% --- Phần 1: Tính toán và In chỉ số đánh giá ---
    
    fprintf('--- Đánh giá pha ước tính ban đầu (Estimated Phase) ---\n');
    % Tính toán sai số cho pha ước tính
    error_estimated_raw = estimated_phase - true_phase;
    piston_estimated = mean(error_estimated_raw(:));
    error_estimated_corrected = error_estimated_raw - piston_estimated;
    rmse_estimated = sqrt(mean(error_estimated_corrected(:).^2));
    
    fprintf('   - Sai số RMSE (hiệu chỉnh piston): %.4f rad\n', rmse_estimated);

    fprintf('\n--- Đánh giá kết quả cuối cùng (Unwrapped Surface) ---\n');
    % Tính toán sai số cho bề mặt đã mở pha
    error_unwrapped_raw = unwrapped_surface - true_phase;
    piston_unwrapped = mean(error_unwrapped_raw(:));
    error_unwrapped_corrected = error_unwrapped_raw - piston_unwrapped;
    rmse_unwrapped = sqrt(mean(error_unwrapped_corrected(:).^2));
    
    fprintf('   - Sai số RMSE (hiệu chỉnh piston): %.4f rad\n', rmse_unwrapped);
    
    fprintf('\n--- Tóm tắt ---\n');
    improvement = ((rmse_estimated - rmse_unwrapped) / rmse_estimated) * 100;
    fprintf('   => Thuật toán đã cải thiện RMSE %.2f%% so với pha ước tính ban đầu.\n', improvement);
    
    if rmse_unwrapped < 0.1
        fprintf('   => Đánh giá: Kết quả rất tốt! 👍\n');
    elseif rmse_unwrapped < 0.5
        fprintf('   => Đánh giá: Kết quả khá tốt. ✅\n');
    else
        fprintf('   => Đánh giá: Kết quả có sai số đáng kể. ⚠️\n');
    end

    %% --- Phần 2: Trực quan hóa các bề mặt 3D ---
    
    figure('Name', 'So sánh các bề mặt 3D', 'Position', [100, 100, 1000, 800], 'Color', 'w');
    sgtitle('So sánh các bề mặt pha ở dạng 3D', 'FontSize', 16, 'FontWeight', 'bold');

    % 1. Bề mặt pha gốc (True Phase)
    subplot(2, 2, 1);
    surf( true_phase, 'EdgeColor', 'none');
    title('1. Bề mặt pha gốc (True)');
    xlabel('X'); ylabel('Y'); zlabel('Phase (rad)');
    axis tight; colormap(gca, 'parula'); colorbar;
    % 2. Bề mặt sai lệch giữa estimate và unwrapped
    subplot(2, 2, 2);
    surf( estimated_phase - unwrapped_surface, 'EdgeColor', 'none');
    title('4. Bề mặt sai lệch (Estimated - Unwrapped)');
    xlabel('X'); ylabel('Y'); zlabel('Error (rad)');
    axis tight; colormap(gca, 'parula'); colorbar;
    % 3. Bề mặt đã mở pha (Unwrapped Surface)
    subplot(2, 2, 3);
    surf( unwrapped_surface, 'EdgeColor', 'none');
    title('3. Bề mặt đã mở pha (Unwrapped)');
    xlabel('X'); ylabel('Y'); zlabel('Phase (rad)');
    axis tight; colormap(gca, 'parula'); colorbar;

    % 4. Bề mặt sai lệch (Error Surface)
    subplot(2, 2, 4);
    surf( error_unwrapped_corrected, 'EdgeColor', 'none');
    title('4. Bề mặt sai lệch (True - Unwrapped)');
    xlabel('X'); ylabel('Y'); zlabel('Error (rad)');
    axis tight; colormap(gca, 'parula'); colorbar;
end