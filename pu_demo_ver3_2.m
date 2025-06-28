
% PHASE UNWRAPPING HOLOGRAPHY DEMO
% Thuật toán kết hợp phase unwrapping cho giao thoa holography
% Tác giả: Demo Code
% Ngày: 2025
% Phiên bản: Cập nhật với bề mặt test chuẩn và sai lệch lớn.

clc; clear; close all;
%% Phần 1: Mô phỏng dữ liệu (Phiên bản cập nhật)
fprintf('=== MÔ PHỎNG DỮ LIỆU HOLOGRAPHIC INTERFEROMETRY (NÂNG CAO) ===\n');
% Tham số mô phỏng
N = 256;                    % Kích thước ảnh
[X, Y] = meshgrid(1:N, 1:N);

% Tạo pha thực (true phase) - dạng bề mặt phức tạp chuẩn
true_phase = create_complex_phase_surface(X, Y, N);

% Mô phỏng pha wrapped từ FFT hologram
phase_wrapped = wrap_phase(true_phase);

% THÊM BƯỚC LỌC PHA WRAPPED
fprintf('Thực hiện lọc pha wrapped...\n');
phase_wrapped_filtered = filter_wrapped_phase(phase_wrapped);
phase_wrapped = phase_wrapped_filtered; 
fprintf('Lọc pha wrapped hoàn tất.\n');

% Mô phỏng pha ước lượng từ vân giao thoa (có sai lệch lớn 2pi-4pi)
phase_estimate = simulate_fringe_phase_estimate(true_phase, X, Y);

% Hiển thị dữ liệu mô phỏng
display_simulation_data(true_phase, phase_wrapped, phase_estimate);

%% Phần 2: Thuật toán Phase Unwrapping kết hợp
fprintf('\n=== THỰC HIỆN PHASE UNWRAPPING KẾT HỢP ===\n');
% Bước 1: Phân tích chất lượng
reliability_map = calculate_reliability_map(phase_wrapped, phase_estimate);
% Bước 2: Ước lượng số lần nhảy 2π
jump_count = estimate_jump_count(phase_wrapped, phase_estimate);
% Bước 3: Phase unwrapping kết hợp
phase_unwrapped = hybrid_phase_unwrapping(phase_wrapped, phase_estimate, ...
                                         reliability_map, jump_count);
% Bước 4: Tinh chỉnh toàn cục
phase_refined_global = global_refinement(phase_unwrapped, reliability_map);
% THÊM BƯỚC XỬ LÝ ĐIỂM NHIỄU SPARSE SAU UNWRAPPING
fprintf('\n=== XỬ LÝ ĐIỂM NHIỄU SPARSE TRONG PHA UNWRAPPED ===\n');
phase_final = correct_sparse_artifacts(phase_refined_global);
fprintf('Xử lý điểm nhiễu sparse hoàn tất.\n');


%%




%% Phần 3: Đánh giá kết quả
fprintf('\n=== ĐÁNH GIÁ KẾT QUẢ ===\n');
evaluate_results(true_phase, phase_wrapped, phase_estimate, phase_final);
% Hiển thị kết quả cuối cùng
display_final_results(true_phase, phase_wrapped, phase_estimate, ...
    phase_final, reliability_map);
% Hiển thị kết quả dưới dạng 3D
fprintf('\n=== HIỂN THỊ KẾT QUẢ 3D ===\n');
display_3d_views(true_phase, phase_wrapped, phase_final);

%% FUNCTIONS

% =========================================================================
% CÁC HÀM MÔ PHỎNG ĐÃ ĐƯỢC CẬP NHẬT
% =========================================================================

function true_phase = create_complex_phase_surface(X, Y, N)
    % CẬP NHẬT: Tạo bề mặt pha phức tạp dựa trên hàm 'peaks' của MATLAB,
    % một hàm kiểm tra (benchmark) tiêu chuẩn trong các bài báo khoa học.
    
    % Tạo ma trận tọa độ chuẩn hóa từ -1 đến 1 cho hàm peaks
    [X_norm, Y_norm] = meshgrid(linspace(-3, 3, N), linspace(-3, 3, N));
    
    % Thành phần chính - hàm peaks, được nhân hệ số để tăng độ phức tạp
    component_peaks = 2 * peaks(X_norm, Y_norm);
    
    % Thành phần nghiêng (tilt) để mô phỏng vật thể không song song hoàn hảo
    component_tilt = 0.08 * (X - N/2) + 0.05 * (Y - N/2);
    
    true_phase = component_peaks + component_tilt;
    
    fprintf('Tạo pha thực (dựa trên hàm peaks) với range: %.2f đến %.2f rad\n', ...
            min(true_phase(:)), max(true_phase(:)));
end

function phase_estimate = simulate_fringe_phase_estimate(true_phase, X, Y)
    % CẬP NHẬT: Mô phỏng pha ước lượng với sai lệch hệ thống LỚN (2pi - 4pi).
    
    % Làm mờ để mô phỏng độ phân giải thấp hơn
    h = fspecial('gaussian', [5 5], 1.5);
    phase_smoothed = imfilter(true_phase, h, 'replicate');
    
    % Thêm nhiễu Gaussian
    noise_level = 0.1;
    gaussian_noise = noise_level * randn(size(true_phase));
    
    % Thêm sai lệch hệ thống lớn (systematic error) trong khoảng 0 đến ~4pi
    % Đây là một thách thức lớn cho thuật toán
    systematic_error = 3*pi*(X/size(X,1)) - 2*pi*sin(2*pi*Y/(4*size(Y,1)));

    % Kết hợp các yếu tố
    phase_estimate = phase_smoothed + gaussian_noise + systematic_error ;

    %     Thêm một số điểm outlier để tăng độ khó
    %     outlier_mask = rand(size(true_phase)) < 0.02;
    %     phase_estimate(outlier_mask) = phase_estimate(outlier_mask) + ...
    %                                    5 * randn(sum(outlier_mask(:)), 1);
    %
    % Tính sai số thực tế so với pha thực
    actual_deviation = phase_estimate - true_phase;
    fprintf('Tạo pha ước lượng với RMSE: %.3f rad\n', ...
        sqrt(mean((phase_estimate(:) - true_phase(:)).^2)));
    fprintf('   -> Dải sai lệch của pha ước lượng: %.2f đến %.2f rad\n', ...
        min(actual_deviation(:)), max(actual_deviation(:)));
end

% =========================================================================
% CÁC HÀM CÒN LẠI (GIỮ NGUYÊN)
% =========================================================================

function phase_wrapped = wrap_phase(phase)
    phase_wrapped = angle(exp(1i * phase));
end

function phase_wrapped_filtered = filter_wrapped_phase(phase_wrapped_input)
    filter_size = [3 3];
    phase_wrapped_filtered = medfilt2(phase_wrapped_input, filter_size);
end

function reliability_map = calculate_reliability_map(phase_wrapped, phase_estimate)
    %     [gx_w, gy_w] = gradient(phase_wrapped);
    %     [gx_e, gy_e] = gradient(phase_estimate);
    [gx_w, gy_w] = PhaseGradientXY(phase_wrapped);
    [gx_e, gy_e] = PhaseGradientXY(phase_estimate);
    grad_consistency = exp(-abs(gx_w - gx_e) - abs(gy_w - gy_e));
    
    h = ones(5,5)/25;
    %     local_var_wrapped = imfilter(phase_wrapped.^2, h) - (imfilter(phase_wrapped, h)).^2;
    %     var_reliability = exp(-local_var_wrapped);
    var_reliability = exp(-PhaseDerivativeVariance(phase_wrapped));
    he_so = 0.5 ;
    reliability_map = he_so * grad_consistency + (1-he_so) * var_reliability;
    
    h_smooth = fspecial('gaussian', [3 3], 0.8);
    reliability_map = imfilter(reliability_map, h_smooth, 'replicate');
    
    fprintf('Tính toán bản đồ độ tin cậy hoàn tất\n');
end

function jump_count = estimate_jump_count(phase_wrapped, phase_estimate)
phase_diff = phase_estimate - phase_wrapped;
jump_count = round(phase_diff / (2*pi));

jump_count = medfilt2(jump_count, [5 5]); % Tăng kích thước lọc để ổn định hơn

fprintf('Ước lượng jump count: min=%d, max=%d\n', ...
    min(jump_count(:)), max(jump_count(:)));
end


function phase_refined = process_boundary_regions(phase_unwrapped, reliability_map)
reliable_mask = reliability_map > 0.3;
phase_refined = phase_unwrapped;
unreliable_regions = ~reliable_mask;

if sum(unreliable_regions(:)) > 0
    [Y, X] = find(reliable_mask);
    [Yi, Xi] = find(unreliable_regions);

    if ~isempty(Y) && ~isempty(Yi)
        reliable_values = phase_unwrapped(reliable_mask);
        F = scatteredInterpolant(X, Y, reliable_values, 'natural', 'nearest');
        interpolated_values = F(Xi, Yi);
            phase_refined(unreliable_regions) = interpolated_values;
        end
    end
end

function phase_final = global_refinement(phase_unwrapped, reliability_map)
    h = fspecial('gaussian', [3 3], 0.8);
    phase_smoothed = imfilter(phase_unwrapped, h, 'replicate');
    
    alpha = 0.15;
    weight_smooth = alpha * (1 - reliability_map);
    weight_original = 1 - weight_smooth;
    
    phase_final = weight_original .* phase_unwrapped + weight_smooth .* phase_smoothed;
    
    phase_final = fix_remaining_discontinuities(phase_final);
    
    fprintf('Tinh chỉnh toàn cục hoàn tất\n');
end

function phase_corrected = fix_remaining_discontinuities(phase)
    [gx, gy] = gradient(phase);
    grad_magnitude = sqrt(gx.^2 + gy.^2);
    
    threshold = 3 * std(grad_magnitude(:));
    discontinuity_mask = grad_magnitude > threshold;
    
    phase_corrected = phase;
    
    if sum(discontinuity_mask(:)) > 0
        se = strel('disk', 1);
        discontinuity_mask = imdilate(discontinuity_mask, se);
        phase_corrected = regionfill(phase, discontinuity_mask);
    end
end

function corrected_unwrapped_phase = correct_sparse_artifacts(unwrapped_phase_input)
    filter_size = [7 7]; % Sử dụng bộ lọc lớn hơn để xử lý nhiễu hiệu quả hơn
    filtered_unwrapped_phase = medfilt2(unwrapped_phase_input, filter_size, 'symmetric');
    
    delta_k = round((filtered_unwrapped_phase - unwrapped_phase_input) / (2*pi));
    
    corrected_unwrapped_phase = unwrapped_phase_input + delta_k * (2*pi);
end

function display_simulation_data(true_phase, phase_wrapped, phase_estimate)
    figure('Name', 'Dữ liệu mô phỏng (Nâng cao)', 'Position', [100 100 1200 400]);
    
    subplot(1,3,1);
    imagesc(true_phase); colorbar; title('Pha thực (Peaks)'); axis equal tight;
    
    subplot(1,3,2);
    imagesc(phase_wrapped); colorbar; title('Pha Wrapped (đã lọc)'); axis equal tight;
    
    subplot(1,3,3);
    imagesc(phase_estimate); colorbar; title('Pha ước lượng (sai lệch lớn)'); axis equal tight;
    
    colormap jet;
end

function evaluate_results(true_phase, phase_wrapped, phase_estimate, phase_final)
    rmse_wrapped = sqrt(mean((wrap_phase(true_phase(:) - phase_wrapped(:))).^2));
    rmse_estimate = sqrt(mean((true_phase(:) - phase_estimate(:)).^2));
    rmse_final = sqrt(mean((true_phase(:) - phase_final(:)).^2));
    
    % Hiệu chỉnh độ lệch DC trung bình trước khi tính correlation
    corr_estimate = corr(true_phase(:), phase_estimate(:) - mean(phase_estimate(:)));
    corr_final = corr(true_phase(:), phase_final(:) - mean(phase_final(:)));
    corr_wrapped = corr(true_phase(:), phase_wrapped(:));
    
    fprintf('RMSE Comparison:\n');
    fprintf('  Phase Wrapped (đã lọc):  %.4f rad\n', rmse_wrapped); 
    fprintf('  Phase Estimate: %.4f rad\n', rmse_estimate);
    fprintf('  Phase Final:    %.4f rad\n', rmse_final);
    fprintf('\nCorrelation with True Phase:\n');
    fprintf('  Phase Wrapped (đã lọc):  %.4f\n', corr_wrapped); 
    fprintf('  Phase Estimate: %.4f\n', corr_estimate);
    fprintf('  Phase Final:    %.4f\n', corr_final);
    
    improvement_vs_estimate = (rmse_estimate - rmse_final) / rmse_estimate * 100;
    
    fprintf('\nImprovement:\n');
    fprintf('  vs Estimate: %.1f%%\n', improvement_vs_estimate);
end

function display_final_results(true_phase, phase_wrapped, phase_estimate, ...
                              phase_final, reliability_map)
    figure('Name', 'Kết quả Phase Unwrapping', 'Position', [100 500 1500 800]);
    
    subplot(2,3,1); imagesc(true_phase); colorbar; title('Pha thực'); axis equal tight;
    subplot(2,3,2); imagesc(phase_wrapped); colorbar; title('Pha Wrapped (đã lọc)'); axis equal tight;
    subplot(2,3,3); imagesc(phase_estimate); colorbar; title('Pha ước lượng'); axis equal tight;
    subplot(2,3,4); imagesc(phase_final); colorbar; title('Pha sau unwrapping và xử lý nhiễu'); axis equal tight;
    subplot(2,3,5); imagesc(reliability_map); colorbar; title('Bản đồ độ tin cậy'); axis equal tight;
    subplot(2,3,6); error_map = abs(true_phase - phase_final); imagesc(error_map); colorbar; title('Sai số tuyệt đối'); axis equal tight;
    
    colormap jet;
    
    figure('Name', 'So sánh Profile', 'Position', [200 100 800 600]);
    center_row = round(size(true_phase, 1) / 2);
    profile_true = true_phase(center_row, :);
    profile_wrapped = phase_wrapped(center_row, :);
    profile_estimate = phase_estimate(center_row, :);
    profile_final = phase_final(center_row, :);
    
    subplot(2,1,1);
    plot(profile_true, 'k-', 'LineWidth', 2); hold on;
    plot(profile_wrapped, 'r--', 'LineWidth', 1.5);
    plot(profile_final, 'g-', 'LineWidth', 2);
    plot(profile_estimate, 'b:', 'LineWidth', 1.5);
    legend('True', 'Wrapped (đã lọc)', 'Final', 'Estimate', 'Location', 'best');
    title('So sánh Profile tại hàng giữa'); xlabel('Pixel'); ylabel('Phase (rad)'); grid on;
    
    subplot(2,1,2);
    plot(abs(profile_true - profile_final), 'g-', 'LineWidth', 2); hold on;
    plot(abs(true_phase(center_row,:) - profile_estimate(:)'), 'b:', 'LineWidth', 1.5);
    legend('Sai số Final', 'Sai số Estimate', 'Location', 'best');
    title('Sai số tuyệt đối'); xlabel('Pixel'); ylabel('Absolute Error (rad)'); grid on; ylim([0 max(abs(profile_true-profile_estimate))*1.1]);
end

function display_3d_views(true_phase, phase_wrapped, phase_final)
    figure('Name', 'Hiển thị 3D', 'Position', [300 300 1200 800]);
    colormap jet;
    
    view_angle = [-35, 45];
    
    subplot(2, 2, 1); surf(true_phase); shading interp; colorbar; title('Pha thực (3D)');
    xlabel('X'); ylabel('Y'); zlabel('Phase (rad)'); axis tight; view(view_angle);
    
    subplot(2, 2, 2); surf(phase_wrapped); shading interp; colorbar; title('Pha Wrapped (3D - đã lọc)');
    xlabel('X'); ylabel('Y'); zlabel('Phase (rad)'); axis tight; view(view_angle);
    
    subplot(2, 2, 3); surf(phase_final); shading interp; colorbar; title('Pha sau unwrapping (3D)');
    xlabel('X'); ylabel('Y'); zlabel('Phase (rad)'); axis tight; view(view_angle);
    
    subplot(2, 2, 4); error_map_3d = abs(true_phase - phase_final); surf(error_map_3d);
    shading interp; colorbar; title('Sai số tuyệt đối (3D)');
    xlabel('X'); ylabel('Y'); zlabel('Error (rad)'); axis tight; view(view_angle);
    
    fprintf('Hiển thị 3D hoàn tất.\n');
end

function phase_unwrapped = hybrid_phase_unwrapping(phase_wrapped, phase_estimate, reliability_map, jump_count)
    % HYBRID_PHASE_UNWRAPPING - Kết hợp quality-guided và estimate-guided unwrapping
    %
    % Inputs:
    %   phase_wrapped - Wrapped phase data
    %   phase_estimate - Phase estimate from external source
    %   reliability_map - Quality/reliability map
    %   jump_count - Initial estimate of 2π jumps
    %
    % Output:
    %   phase_unwrapped - Unwrapped phase result
    
    [rows, cols] = size(phase_wrapped);
    phase_unwrapped = zeros(rows, cols);
    unwrapped_mask = false(rows, cols);
    
    % Tính combined quality metric
    residual = abs(phase_estimate - phase_wrapped - 2*pi*jump_count);
    residual(residual > pi) = 2*pi - residual(residual > pi); % Wrap residual
    combined_quality = reliability_map .* exp(-residual);
    
    % Tìm pixel có quality cao nhất để bắt đầu
    [~, start_idx] = max(combined_quality(:));
    [start_row, start_col] = ind2sub([rows, cols], start_idx);
    
    % Khởi tạo pixel đầu tiên
    phase_unwrapped(start_row, start_col) = phase_wrapped(start_row, start_col) + ...
                                           2*pi * jump_count(start_row, start_col);
    unwrapped_mask(start_row, start_col) = true;
    
    % Tạo priority queue sử dụng cell array
    queue_data = [];
    queue_priority = [];
    
    % Thêm neighbors của pixel đầu tiên vào queue
    neighbors = get_neighbors(start_row, start_col, rows, cols);
    for k = 1:size(neighbors, 1)
        nr = neighbors(k, 1);
        nc = neighbors(k, 2);
        if ~unwrapped_mask(nr, nc)
            queue_data = [queue_data; nr, nc];
            queue_priority = [queue_priority; combined_quality(nr, nc)];
        end
    end
    
    % Main unwrapping loop
    while ~isempty(queue_data)
        % Lấy pixel có priority cao nhất
        [~, max_idx] = max(queue_priority);
        current_row = queue_data(max_idx, 1);
        current_col = queue_data(max_idx, 2);
        
        % Xóa khỏi queue
        queue_data(max_idx, :) = [];
        queue_priority(max_idx) = [];
        
        % Skip nếu pixel đã được unwrap
        if unwrapped_mask(current_row, current_col)
            continue;
        end
        
        % Hybrid decision making
        k_estimate = jump_count(current_row, current_col);
        k_neighbors = estimate_k_from_neighbors(current_row, current_col, ...
                                              phase_wrapped, phase_unwrapped, ...
                                              unwrapped_mask);
        
        % Tính weights
        w_estimate = combined_quality(current_row, current_col);
        w_neighbors = calculate_neighbor_weight(current_row, current_col, ...
                                              reliability_map, unwrapped_mask);
        
        % Quyết định k final
        if w_neighbors == 0
            % Không có neighbors đã unwrap, dùng estimate
            k_final = k_estimate;
        else
            % Kết hợp estimate và neighbors
            k_combined = (w_estimate * k_estimate + w_neighbors * k_neighbors) / ...
                        (w_estimate + w_neighbors);
            k_final = round(k_combined);
        end
        
        % Unwrap pixel
        phase_unwrapped(current_row, current_col) = ...
            phase_wrapped(current_row, current_col) + 2*pi * k_final;
        unwrapped_mask(current_row, current_col) = true;
        
        % Thêm neighbors mới vào queue
        neighbors = get_neighbors(current_row, current_col, rows, cols);
        for k = 1:size(neighbors, 1)
            nr = neighbors(k, 1);
            nc = neighbors(k, 2);
            if ~unwrapped_mask(nr, nc)
                % Kiểm tra xem pixel đã có trong queue chưa
                existing_idx = find(queue_data(:,1) == nr & queue_data(:,2) == nc);
                if isempty(existing_idx)
                    queue_data = [queue_data; nr, nc];
                    queue_priority = [queue_priority; combined_quality(nr, nc)];
                end
            end
        end
    end
end
function neighbors = get_neighbors(row, col, rows, cols)
    % Lấy 4-connected neighbors
    neighbors = [];
    directions = [-1, 0; 1, 0; 0, -1; 0, 1]; % up, down, left, right
    
    for i = 1:size(directions, 1)
        nr = row + directions(i, 1);
        nc = col + directions(i, 2);
        if nr >= 1 && nr <= rows && nc >= 1 && nc <= cols
            neighbors = [neighbors; nr, nc];
        end
    end
end

function k_neighbors = estimate_k_from_neighbors(row, col, phase_wrapped, ...
                                               phase_unwrapped, unwrapped_mask)
    % Ước lượng k từ các neighbors đã được unwrap
    neighbors = get_neighbors(row, col, size(phase_wrapped, 1), size(phase_wrapped, 2));
    
    k_estimates = [];
    for i = 1:size(neighbors, 1)
        nr = neighbors(i, 1);
        nc = neighbors(i, 2);
        if unwrapped_mask(nr, nc)
            % Tính k từ neighbor
            phase_diff = phase_unwrapped(nr, nc) - phase_wrapped(row, col);
            k_from_neighbor = round(phase_diff / (2*pi));
            k_estimates = [k_estimates; k_from_neighbor];
        end
    end
    
    if isempty(k_estimates)
        k_neighbors = 0;
    else
        % Lấy median để robust với outliers
        k_neighbors = median(k_estimates);
    end
end

function weight = calculate_neighbor_weight(row, col, reliability_map, unwrapped_mask)
    % Tính weight dựa trên reliability của neighbors đã unwrap
    neighbors = get_neighbors(row, col, size(reliability_map, 1), size(reliability_map, 2));
    
    total_weight = 0;
    count = 0;
    for i = 1:size(neighbors, 1)
        nr = neighbors(i, 1);
        nc = neighbors(i, 2);
        if unwrapped_mask(nr, nc)
            total_weight = total_weight + reliability_map(nr, nc);
            count = count + 1;
        end
    end
    
    if count == 0
        weight = 0;
    else
        weight = total_weight / count;
    end
end

function derivative_variance = PhaseDerivativeVariance(IM_phase, varargin)
[r_dim,c_dim]=size(IM_phase);
if nargin>=2                                    %Has a mask been included? If so crop the image to the mask borders to save computational time
    IM_mask=varargin{1};
    [maskrows,maskcols]=find(IM_mask);          %Identify coordinates of the mask
    minrow=min(maskrows)-1;                     %Identify the limits of the mask 
    maxrow=max(maskrows)+1;
    mincol=min(maskcols)-1;
    maxcol=max(maskcols)+1;
    width=maxcol-mincol;                        %Now ensure that the cropped area is square
    height=maxrow-minrow;
    if height>width
        maxcol=maxcol + floor((height-width)/2) + mod(height-width,2);
        mincol=mincol - floor((height-width)/2);
    elseif width>height
        maxrow=maxrow + floor((width-height)/2) + mod(width-height,2);
        minrow=minrow - floor((width-height)/2);
    end
    if minrow<1 minrow=1; end
    if maxrow>r_dim maxrow=r_dim; end
    if mincol<1 mincol=1; end
    if maxcol>c_dim maxcol=c_dim; end
    IM_phase=IM_phase(minrow:maxrow, mincol:maxcol);        %Crop the original image to save computation time
end
    
[dimx, dimy]=size(IM_phase);
dx=zeros(dimx,dimy);
p = unwrap([IM_phase(:,1) IM_phase(:,2)],[],2);
dx(:,1)=(p(:,2) - IM_phase(:,1))./2;                    %Take the partial derivative of the unwrapped phase in the x-direction for the first column
p = unwrap([IM_phase(:,dimy-1) IM_phase(:,dimy)],[],2);
dx(:,dimy)=(p(:,2) - IM_phase(:,dimy-1))./2;            %Take the partial derivative of the unwrapped phase in the x-direction for the last column
for i=2:dimy-1
    p = unwrap([IM_phase(:,i-1) IM_phase(:,i+1)],[],2);
    dx(:,i)=(p(:,2) - IM_phase(:,i-1))./3;              %Take partial derivative of the unwrapped phase in the x-direction for the remaining columns
end
dy = zeros(dimx,dimy);
q = unwrap([IM_phase(1,:)' IM_phase(2,:)'],[],2);
dy(1,:)=(q(:,2)' - IM_phase(1,:))./2;                   %Take the partial derivative of the unwrapped phase in the y-direction for the first row
p = unwrap([IM_phase(dimx-1,:)' IM_phase(dimx,:)'],[],2);
dy(dimx,:)=(q(:,2)' - IM_phase(dimx-1,:))./2;           %Take the partial derivative of the unwrapped phase in the y-direction for the last row
for i=2:dimx-1
    q=unwrap([IM_phase(i-1,:)' IM_phase(i+1,:)'],[],2);
    dy(i,:)=(q(:,2)' - IM_phase(i-1,:))./3;             %Take the partial derivative of the unwrapped phase in the y-direction for the remaining rows
end
dx_centre=dx(2:dimx-1, 2:dimy-1);
dx_left=dx(2:dimx-1,1:dimy-2); 
dx_right=dx(2:dimx-1,3:dimy);
dx_above=dx(1:dimx-2,2:dimy-1);
dx_below=dx(3:dimx,2:dimy-1);
mean_dx=(dx_centre+dx_left+dx_right+dx_above+dx_below)./5;
dy_centre=dy(2:dimx-1, 2:dimy-1);
dy_left=dy(2:dimx-1,1:dimy-2); 
dy_right=dy(2:dimx-1,3:dimy);
dy_above=dy(1:dimx-2,2:dimy-1);
dy_below=dy(3:dimx,2:dimy-1);
mean_dy=(dy_centre+dy_left+dy_right+dy_above+dy_below)./5;
stdvarx=sqrt( (dx_left - mean_dx).^2 + (dx_right - mean_dx).^2 + ...
              (dx_above - mean_dx).^2 + (dx_below - mean_dx).^2 + (dx_centre - mean_dx).^2 ); 
stdvary=sqrt( (dy_left - mean_dy).^2 + (dy_right - mean_dy).^2 + ...
              (dy_above - mean_dy).^2 + (dy_below - mean_dy).^2 + (dy_centre - mean_dy).^2 ); 
derivative_variance=100*ones(dimx, dimy);                         %Ensure that the border pixels have high derivative variance values
derivative_variance(2:dimx-1, 2:dimy-1)=stdvarx + stdvary;
if nargin>=2                                                      %Does the image have to be padded back to the original size?
    [orig_rows, orig_cols]=size(IM_mask);
    temp=100*ones(orig_rows, orig_cols);
    temp(minrow:maxrow, mincol:maxcol)=derivative_variance;       %Pad the remaining pixels with poor phase quality values
    derivative_variance=temp;
end
end

function [gradient_x, gradient_y] = PhaseGradientXY(IM_phase, varargin)
    % PhaseGradient - Tính gradient của ảnh phase theo hướng x và y
    %
    % Input:
    %   IM_phase - Ma trận ảnh phase
    %   varargin{1} - (Tùy chọn) Mask để crop ảnh và tiết kiệm thời gian tính toán
    %
    % Output:
    %   gradient_x - Gradient theo hướng x (đạo hàm riêng theo cột)
    %   gradient_y - Gradient theo hướng y (đạo hàm riêng theo hàng)
    
    [r_dim, c_dim] = size(IM_phase);
    
    % Xử lý mask nếu có
    if nargin >= 2
        IM_mask = varargin{1};
        [maskrows, maskcols] = find(IM_mask);
    
        % Xác định giới hạn của mask
        minrow = min(maskrows) - 1;
        maxrow = max(maskrows) + 1;
        mincol = min(maskcols) - 1;
        maxcol = max(maskcols) + 1;
    
        % Đảm bảo vùng crop là hình vuông
        width = maxcol - mincol;
        height = maxrow - minrow;
    
        if height > width
            maxcol = maxcol + floor((height-width)/2) + mod(height-width, 2);
            mincol = mincol - floor((height-width)/2);
        elseif width > height
            maxrow = maxrow + floor((width-height)/2) + mod(width-height, 2);
            minrow = minrow - floor((width-height)/2);
        end
    
        % Đảm bảo không vượt quá biên ảnh
        if minrow < 1, minrow = 1; end
        if maxrow > r_dim, maxrow = r_dim; end
        if mincol < 1, mincol = 1; end
        if maxcol > c_dim, maxcol = c_dim; end
    
        % Crop ảnh để tiết kiệm thời gian tính toán
        IM_phase = IM_phase(minrow:maxrow, mincol:maxcol);
    end
    
    [dimx, dimy] = size(IM_phase);
    
    %% Tính gradient theo hướng x (đạo hàm riêng theo cột)
    gradient_x = zeros(dimx, dimy);
    
    % Cột đầu tiên - sử dụng forward difference với unwrap
    p = unwrap([IM_phase(:,1) IM_phase(:,2)], [], 2);
    gradient_x(:,1) = (p(:,2) - IM_phase(:,1)) / 2;
    
    % Cột cuối cùng - sử dụng backward difference với unwrap
    p = unwrap([IM_phase(:,dimy-1) IM_phase(:,dimy)], [], 2);
    gradient_x(:,dimy) = (p(:,2) - IM_phase(:,dimy-1)) / 2;
    
    % Các cột ở giữa - sử dụng central difference với unwrap
    for i = 2:dimy-1
        p = unwrap([IM_phase(:,i-1) IM_phase(:,i+1)], [], 2);
        gradient_x(:,i) = (p(:,2) - IM_phase(:,i-1)) / 3;
    end
    
    %% Tính gradient theo hướng y (đạo hàm riêng theo hàng)
    gradient_y = zeros(dimx, dimy);
    
    % Hàng đầu tiên - sử dụng forward difference với unwrap
    q = unwrap([IM_phase(1,:)' IM_phase(2,:)'], [], 2);
    gradient_y(1,:) = (q(:,2)' - IM_phase(1,:)) / 2;
    
    % Hàng cuối cùng - sử dụng backward difference với unwrap
    p = unwrap([IM_phase(dimx-1,:)' IM_phase(dimx,:)'], [], 2);
    gradient_y(dimx,:) = (p(:,2)' - IM_phase(dimx-1,:)) / 2;
    
    % Các hàng ở giữa - sử dụng central difference với unwrap
    for i = 2:dimx-1
        q = unwrap([IM_phase(i-1,:)' IM_phase(i+1,:)'], [], 2);
        gradient_y(i,:) = (q(:,2)' - IM_phase(i-1,:)) / 3;
    end
    
    %% Nếu có mask, pad kết quả về kích thước ban đầu
    if nargin >= 2
        [orig_rows, orig_cols] = size(IM_mask);
    
        % Pad gradient_x
        temp_gx = zeros(orig_rows, orig_cols);
        temp_gx(minrow:maxrow, mincol:maxcol) = gradient_x;
        gradient_x = temp_gx;
    
        % Pad gradient_y
        temp_gy = zeros(orig_rows, orig_cols);
        temp_gy(minrow:maxrow, mincol:maxcol) = gradient_y;
        gradient_y = temp_gy;
    end

end