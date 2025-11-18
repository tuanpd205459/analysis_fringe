close all;
% chỉ có thuật toán unwrapping
load("after_estimate.mat");
% Thêm độ nghiêng vào pha đối tượng
slope_x = 0.2;  % Độ nghiêng theo hướng X
slope_y = 0.3;  % Độ nghiêng theo hướng Y
bias = 0.1;     % Độ chênh lệch


skeleton_image = BW;
object_phase = object_phase + slope_x * x + slope_y * y;

wrapped_phase = wrapToPi(object_phase) ;
figure;
surf(wrapped_phase,"EdgeColor","none");
colorbar;
title('wrapped phase: ');

%% 7. ƯỚC LƯỢNG PHA BẰNG PHƯƠNG PHÁP PHÂN TÍCH VÂN
fprintf('--> Bước 3: Ước lượng pha thô bằng phân tích vân...\n');
% Làm mảnh và gán bậc vân

[~, labels, img] = assign_fringe_order(skeleton_image, true);

% Tái tạo bề mặt từ vân
[phi_est, ~] = reconSurface_linearPushed(img, labels, 632.8e-9, 'None', false);

% Sau khi có phi_est
% systematic_error = (pi/5) * (2*rand(size(phi_est))-1); % random ±λ/20
% phi_est = phi_est + systematic_error;

phi_est = phi_est - min(phi_est(:));

[X, Y] = meshgrid(1:N, 1:M);
plane_phase = 2*pi*(fx*X + fy*Y);

plane_phase = plane_phase - min(plane_phase(:));
[phi_est, plane_phase, object_phase_without_noise,...
    wrapped_phase] = crop_multiple_to_smallest(phi_est, plane_phase,...
    object_phase_without_noise, wrapped_phase);
phi_est = phi_est - plane_phase - (max(phi_est(:)- max(plane_phase(:))))/2;

figure;
surf(phi_est,"EdgeColor","none");
title("Anh pha phi estimate co nhieu");

figure;
imagesc(phi_est - object_phase_without_noise);
title("Anh sai lech giua phi est va ground truth");
colorbar;

%% 8. GIẢI BỌC PHA VÀ TINH CHỈNH
fprintf('--> Bước 4: Giải bọc pha và tinh chỉnh kết quả...\n');
% --- Giải bọc pha sử dụng pha ước lượng ---
% [est_phase_flat, wrapped_phase, object_phase] = crop_multiple_to_smallest(est_phase_flat, wrapped_phase, object_phase);

[finalUnwrappedPhase, kMap] = unwrapUsingEstimate(phi_est, wrapped_phase);
% [finalUnwrappedPhase, kMap] = unwrapUsingEstimate2(phi_est, wrapped_phase);

% fprintf("chay k map");
% kMap = kMap - min(kMap(:));
% fprintf("ket thuc kmap");
% figure();
% surf(kMap, 'EdgeColor', 'none');
% title("kMap");
% xlabel('x'); ylabel('y'); zlabel('(rad)');
% colormap; colorbar; 
% off_set = 2;
% % finalUnwrappedPhase = finalUnwrappedPhase(off_set:end-off_set,off_set:end-off_set );
% figure("Name","Kết quả");
% surf(finalUnwrappedPhase, 'EdgeColor', 'none');
% title("Kết quả finalUnwrappedPhase");
% xlabel('x'); ylabel('y'); zlabel('(rad)');
% colormap; colorbar; 


%% 10. Refine artifacts points

% [finalUnwrappedPhase, ~, ~] = correct_sparse_artifacts_iterative(finalUnwrappedPhase, ...
%     'BoundaryCondition', 'symmetric', 'BoundaryWidth', 2, 'MaxIterations', 150);
% 
% figure("Name","Kết quả sau refine");
% surf(finalUnwrappedPhase, 'EdgeColor', 'none');
% title("Kết quả finalUnwrappedPhase sau khi refine");
% xlabel('x'); ylabel('y'); zlabel('(rad)');
% colormap; colorbar; 

% Cắt biên để hiển thị tốt hơn
offset = 10;
finalUnwrappedPhase = finalUnwrappedPhase(offset+1:end-offset, offset+1:end-offset);
%% 11. CÁC THUẬT TOÁN UNWRAPPING KHÁC
unwrapped_Phase_LS_DCT = unwrapping.unwrapPhase(wrapped_phase, 'ls', 'dct'); % LS với DCT
unwrapped_Phase_TIE_FFT = unwrapping.unwrapPhase(wrapped_phase, 'tie', 'fft'); % TIE với FFT
unwrapped_Phase_noncontinue = unwrapping.unwrapPhase(wrapped_phase, 'linh'); % Phương pháp của a Linh
unwrapped_Phase_2dweight = unwrapping.unwrapPhase(wrapped_phase, '2dweight'); % 2D weighted phase unwrapping
unwrapped_Phase_goldstein = goldstein_unwrap(wrapped_phase);
% proposal 
unwrapped_Phase_proposal = finalUnwrappedPhase;
[object_phase, unwrapped_Phase_LS_DCT, unwrapped_Phase_TIE_FFT, unwrapped_Phase_noncontinue,...
    unwrapped_Phase_2dweight, unwrapped_Phase_goldstein, unwrapped_Phase_proposal]...
    = crop_multiple_to_smallest(object_phase, unwrapped_Phase_LS_DCT, unwrapped_Phase_TIE_FFT, unwrapped_Phase_noncontinue,...
    unwrapped_Phase_2dweight, unwrapped_Phase_goldstein, unwrapped_Phase_proposal);
[M,N] = size(object_phase);
% [unwrapped_Phase_LS_DCT, unwrapped_Phase_TIE_FFT, ...
%  unwrapped_Phase_noncontinue, unwrapped_Phase_2dweight, ...
%  unwrapped_Phase_goldstein, unwrapped_Phase_proposal] = ...
%     process_phases(fx, fy, M, N, unwrapped_Phase_LS_DCT, ...
%     unwrapped_Phase_TIE_FFT, unwrapped_Phase_noncontinue, ...
%     unwrapped_Phase_2dweight, unwrapped_Phase_goldstein, ...
%     unwrapped_Phase_proposal);
% Giả sử đã có:
% object_phase, unwrapped_Phase_LS_DCT, unwrapped_Phase_TIE_FFT,
% unwrapped_Phase_noncontinue, unwrapped_Phase_2dweight,
% unwrapped_Phase_goldstein, unwrapped_Phase_proposal

% phase_unwrapped: ma trận pha đã unwrap

% --- Unwrap theo chiều X (cột)
% for j = 2:size(unwrapped_Phase_proposal, 2)
%     delta = unwrapped_Phase_proposal(:, j) - unwrapped_Phase_proposal(:, j-1);
%     unwrapped_Phase_proposal(:, j) = unwrapped_Phase_proposal(:, j) - 2*pi*round(delta/(2*pi));
% end
% 
% --- Unwrap theo chiều Y (hàng)
% for i = 2:size(unwrapped_Phase_proposal, 1)
%     delta = unwrapped_Phase_proposal(i, :) - unwrapped_Phase_proposal(i-1, :);
%     unwrapped_Phase_proposal(i, :) = unwrapped_Phase_proposal(i, :) - 2*pi*round(delta/(2*pi));
% end


figure;
titles = {'Object phase (GT)', 'LS+DCT', 'TIE+FFT', ...
          'Noncontinue', '2D Weighted', 'Goldstein', 'Proposal'};

phases = {object_phase, unwrapped_Phase_LS_DCT, unwrapped_Phase_TIE_FFT, ...
          unwrapped_Phase_noncontinue, unwrapped_Phase_2dweight, ...
          unwrapped_Phase_goldstein, unwrapped_Phase_proposal};

% Tạo lưới tọa độ
[M, N] = size(object_phase);
[X, Y] = meshgrid(1:N, 1:M);

for k = 1:length(phases)
    subplot(2,4,k);
    surf(X, Y, phases{k}, 'EdgeColor','none');  % hiển thị 3D
    colormap jet; 
    colorbar;
    title(titles{k});
    view(45,45);  % góc nhìn đẹp
    axis tight; shading interp;
end





%% 11. Plot mặt cắt ngang
fprintf('--> Bước 5: plot MCN...\n');
% Chọn 1 đường cắt ngang (ví dụ: đường giữa ảnh theo trục y)
row = round(size(object_phase,1)/2);

% Lấy profile theo hàng đó
x = 1:size(object_phase,2);
profile_true        = object_phase(row,:);
profile_LS_DCT      = unwrapped_Phase_LS_DCT(row,:);
profile_TIE_FFT     = unwrapped_Phase_TIE_FFT(row,:);
profile_noncontinue = unwrapped_Phase_noncontinue(row,:);
profile_2dweight    = unwrapped_Phase_2dweight(row,:);
profile_goldstein   = unwrapped_Phase_goldstein(row,:);
profile_proposal    = unwrapped_Phase_proposal(row,:);

% Vẽ tất cả trên 1 plot
figure;
plot(x, profile_true,        'k-',  'LineWidth',1.5); hold on;
plot(x, profile_LS_DCT,      'r--', 'LineWidth',1.2);
plot(x, profile_TIE_FFT,     'b-.', 'LineWidth',1.2);
plot(x, profile_noncontinue, 'g:',  'LineWidth',1.5);
plot(x, profile_2dweight,    'm-',  'LineWidth',1.2);
plot(x, profile_goldstein,   'c--', 'LineWidth',1.2);
plot(x, profile_proposal,    'y-',  'LineWidth',1.5);

grid on;
xlabel('Pixel index');
ylabel('Phase (rad)');
title('So sánh mặt cắt ngang (row giữa ảnh)');
legend('Ground Truth', 'LS+DCT', 'TIE+FFT', 'Non-continue (Linh)', ...
       '2D Weighted', 'Goldstein', 'Proposal');



%% 6. PHÂN TÍCH SAI SỐ (TIẾP THEO)
% --- Tính toán sai số cho các thuật toán khác ---
error_LS_DCT = unwrapped_Phase_LS_DCT - object_phase;
error_TIE_FFT = unwrapped_Phase_TIE_FFT - object_phase;
error_noncontinue = unwrapped_Phase_noncontinue - object_phase;
error_2dweight = unwrapped_Phase_2dweight - object_phase;
error_goldstein = unwrapped_Phase_goldstein - object_phase;
error_proposal = unwrapped_Phase_proposal - object_phase;

% Chuẩn hoá lỗi

%%
% --- Tính sai số giữa object_phase và các bề mặt khác ---
phases = { unwrapped_Phase_LS_DCT, unwrapped_Phase_TIE_FFT, ...
          unwrapped_Phase_noncontinue, unwrapped_Phase_2dweight, ...
          unwrapped_Phase_goldstein, unwrapped_Phase_proposal};

phase_names = {'LS-DCT', 'TIE-FFT', ...
               'Non-continue', '2D-weight', ...
               'Goldstein', 'Proposed'};

nPhase = numel(phases);
errors = struct();

for k = 1:nPhase
    phase = phases{k};

    % --- B1: Loại bỏ offset (so sánh công bằng)
    % Tạo mặt nạ logic cho các điểm ảnh hợp lệ trên cả hai ảnh
    valid_mask = ~isnan(phase) & ~isnan(object_phase);

    % Tính offset chỉ trên những điểm hợp lệ
    offset = median(phase(valid_mask)) - median(object_phase(valid_mask));

    % Điều chỉnh phase
    phase_adj = phase - offset;

    % --- B2: Tính sai số
    % Chỉ tính toán trên các điểm ảnh hợp lệ
    diff = phase_adj(valid_mask) - object_phase(valid_mask);

    rmse = sqrt(mean(diff.^2));              % Không cần 'omitnan' vì đã loại NaN
    mae  = mean(abs(diff));                  % Không cần 'omitnan'
    maxe = max(abs(diff));                   % Không cần 'omitnan'
    % --- B3: Lưu kết quả
    errors(k).Name = phase_names{k};
    errors(k).RMSE = rmse;
    errors(k).MAE  = mae;
    errors(k).MAX  = maxe;
end

% --- Hiển thị kết quả ---
disp('Sai số so với object_phase:');
for k = 1:nPhase
    fprintf('%-12s | RMSE = %.4f | MAE = %.4f | MAX = %.4f\n', ...
        errors(k).Name, errors(k).RMSE, errors(k).MAE, errors(k).MAX);
end

% --- Hiển thị bản đồ sai số 2D ---
% --- Hiển thị bản đồ sai số 2D + giá trị RMSE ---
figure;
for k = 1:nPhase
    phase = phases{k};
    
    % B1: Loại bỏ offset
    offset = median(phase(:)) - median(object_phase(:));
    phase_adj = phase - offset;
    
    % B2: Sai số tuyệt đối từng điểm
    error_map = abs(phase_adj - object_phase);

    % B3: Vẽ
    subplot(2, ceil(nPhase/2), k);
    imagesc(error_map);
    axis image; colorbar;
    title(sprintf('%s\nRMSE = %g', errors(k).Name, errors(k).RMSE));
end
sgtitle('Bản đồ sai số tuyệt đối + RMSE');


%% 8. HIỂN THỊ MẶT CẮT NGANG SAI SỐ
fprintf('\nQuy trình đã hoàn thành!\n');

%% ========================================================================

% -------------------------------------------------------------------------
function [unwrappedPhase, kMap] = unwrapUsingEstimate(estimatedPhase, wrappedPhase)
    % Giải Wrapped pha `wrappedPhase` dựa trên pha ước lượng `estimatedPhase`.
%     wrappedEstimate = wrapToPi(estimatedPhase);
    kMap = round((estimatedPhase - wrappedPhase) / (2*pi));
%     unwrappedPhase = wrappedPhase + 2*pi * kMap;
    unwrappedPhase = estimatedPhase + angle(estimatedPhase - wrappedPhase);
end

% -------------------------------------------------------------------------
function [unwrappedPhase, kMap] = unwrapUsingEstimate2(estimatedPhase, wrappedPhase)
    % Giải Wrapped pha `wrappedPhase` dựa trên pha ước lượng `estimatedPhase`.
    wrappedEstimate = wrapToPi(estimatedPhase);
    kMap = round((estimatedPhase - wrappedEstimate) / (2*pi));
    unwrappedPhase = wrappedPhase + 2*pi * kMap;
end


function [corrected_unwrapped_phase, num_iterations, convergence_history] = correct_sparse_artifacts_iterative(unwrapped_phase_input, varargin)
% Hàm cải tiến: Xử lý các điểm nhiễu sparse với thuật toán lặp và ràng buộc biên
% Dựa trên phương pháp lọc trung vị để xác định và hiệu chỉnh các điểm lỗi.
% Lặp đến khi hội tụ (không còn thay đổi k hoặc thay đổi < epsilon)
%
% Inputs:
%   unwrapped_phase_input - Ma trận pha unwrapped đầu vào
%   varargin - Các tham số tùy chọn:
%       'FilterSize' - Kích thước bộ lọc [default: [15 15]]
%       'Epsilon' - Ngưỡng hội tụ [default: 1e-6]
%       'MaxIterations' - Số lần lặp tối đa [default: 50]
%       'Verbose' - Hiển thị thông tin debug [default: false]
%       'BoundaryCondition' - Điều kiện biên ['zero'|'symmetric'|'replicate'|'circular'] [default: 'symmetric']
%       'BoundaryWidth' - Độ rộng vùng biên không được hiệu chỉnh [default: 0]
%       'PreserveBoundary' - Giữ nguyên giá trị biên [default: true]
%       'MaxDeltaK' - Giới hạn tối đa cho |delta_k| [default: 10]
%       'MaskInvalid' - Mask cho các pixel không hợp lệ [default: []]
%
% Outputs:
%   corrected_unwrapped_phase - Pha đã được hiệu chỉnh
%   num_iterations - Số lần lặp thực tế
%   convergence_history - Lịch sử hội tụ (RMS của delta_k)

    % Xử lý tham số đầu vào
    p = inputParser;
    addParameter(p, 'FilterSize', [5 5], @(x) isnumeric(x) && length(x) == 2);
    addParameter(p, 'Epsilon', 1e-6, @(x) isnumeric(x) && x > 0);
    addParameter(p, 'MaxIterations', 100, @(x) isnumeric(x) && x > 0);
    addParameter(p, 'Verbose', false, @islogical);
    addParameter(p, 'BoundaryCondition', 'symmetric', @(x) ischar(x) && ismember(x, {'zero', 'symmetric', 'replicate', 'circular'}));
    addParameter(p, 'BoundaryWidth', 5, @(x) isnumeric(x) && x >= 0);
    addParameter(p, 'PreserveBoundary', true, @islogical);
    addParameter(p, 'MaxDeltaK', 2, @(x) isnumeric(x) && x > 0);
    addParameter(p, 'MaskInvalid', [], @(x) isempty(x) || islogical(x));
    parse(p, varargin{:});
    
    filter_size = p.Results.FilterSize;
    epsilon = p.Results.Epsilon;
    max_iterations = p.Results.MaxIterations;
    verbose = p.Results.Verbose;
    boundary_condition = p.Results.BoundaryCondition;
    boundary_width = p.Results.BoundaryWidth;
    preserve_boundary = p.Results.PreserveBoundary;
    max_delta_k = p.Results.MaxDeltaK;
    mask_invalid = p.Results.MaskInvalid;
    
    % Khởi tạo
    [rows, cols] = size(unwrapped_phase_input);
    current_phase = unwrapped_phase_input;
    original_phase = unwrapped_phase_input; % Lưu pha gốc để tham chiếu biên
    convergence_history = [];
    num_iterations = 0;
    previous_delta_k = [];
    
    % Tạo mask cho vùng biên nếu cần
    if preserve_boundary && boundary_width > 0
        boundary_mask = create_boundary_mask(rows, cols, boundary_width);
    else
        boundary_mask = false(rows, cols);
    end

% Hàm hỗ trợ: Tạo mask cho vùng biên
function boundary_mask = create_boundary_mask(rows, cols, width)
    boundary_mask = false(rows, cols);
    if width > 0
        boundary_mask(1:width, :) = true;           % Biên trên
        boundary_mask(end-width+1:end, :) = true;   % Biên dưới
        boundary_mask(:, 1:width) = true;           % Biên trái
        boundary_mask(:, end-width+1:end) = true;   % Biên phải
    end
end

% Hàm hỗ trợ: Áp dụng điều kiện biên
function phase_with_boundary = apply_boundary_condition(phase, condition, filter_size)
    [rows, cols] = size(phase);
    pad_rows = floor(filter_size(1)/2);
    pad_cols = floor(filter_size(2)/2);
    
    switch lower(condition)
        case 'zero'
            phase_with_boundary = padarray(phase, [pad_rows, pad_cols], 0, 'both');
        case 'symmetric'
            phase_with_boundary = padarray(phase, [pad_rows, pad_cols], 'symmetric', 'both');
        case 'replicate'
            phase_with_boundary = padarray(phase, [pad_rows, pad_cols], 'replicate', 'both');
        case 'circular'
            phase_with_boundary = padarray(phase, [pad_rows, pad_cols], 'circular', 'both');
        otherwise
            phase_with_boundary = padarray(phase, [pad_rows, pad_cols], 'symmetric', 'both');
    end
end

% Hàm hỗ trợ: Ràng buộc tính liên tục không gian
function delta_k_constrained = apply_spatial_continuity_constraint(delta_k, current_phase)
    % Kiểm tra gradient địa phương để tránh các thay đổi đột ngột
    [rows, cols] = size(delta_k);
    delta_k_constrained = delta_k;
    
    % Tính gradient của pha hiện tại
    [grad_x, grad_y] = gradient(current_phase);
    grad_magnitude = sqrt(grad_x.^2 + grad_y.^2);
    
    % Định nghĩa ngưỡng gradient (vùng có gradient cao được phép thay đổi nhiều hơn)
    grad_threshold = prctile(grad_magnitude(:), 75); % 75th percentile
    
    % Áp dụng ràng buộc dựa trên gradient
    for i = 2:rows-1
        for j = 2:cols-1
            if abs(delta_k(i,j)) > 1 && grad_magnitude(i,j) < grad_threshold
                % Nếu thay đổi lớn nhưng gradient thấp, hạn chế thay đổi
                neighbors = delta_k(i-1:i+1, j-1:j+1);
                median_neighbor = median(neighbors(:));
                
                % Chỉ cho phép thay đổi không quá 1 bước so với median của lân cận
                if abs(delta_k(i,j) - median_neighbor) > 1
                    delta_k_constrained(i,j) = median_neighbor + sign(delta_k(i,j) - median_neighbor);
                end
            end
        end
    end
end
    
    % Xử lý mask invalid
    if isempty(mask_invalid)
        mask_invalid = false(rows, cols);
    else
        if ~isequal(size(mask_invalid), [rows, cols])
            error('MaskInvalid phải có cùng kích thước với unwrapped_phase_input');
        end
    end
    
    % Mask tổng hợp (vùng không được hiệu chỉnh)
    protection_mask = boundary_mask | mask_invalid;
    
    if verbose
        fprintf('Bắt đầu quá trình hiệu chỉnh lặp với ràng buộc biên...\n');
        fprintf('Image size: %dx%d\n', rows, cols);
        fprintf('Filter size: [%d %d], Epsilon: %.2e, Max iterations: %d\n', ...
                filter_size(1), filter_size(2), epsilon, max_iterations);
        fprintf('Boundary condition: %s, Boundary width: %d\n', boundary_condition, boundary_width);
        fprintf('Protected pixels: %d (%.2f%%)\n', sum(protection_mask(:)), 100*sum(protection_mask(:))/(rows*cols));
    end
    
    % Vòng lặp chính
    for iter = 1:max_iterations
        % Bước 1: Xử lý điều kiện biên trước khi lọc
        phase_with_boundary = apply_boundary_condition(current_phase, boundary_condition, filter_size);
        
        % Bước 2: Áp dụng bộ lọc trung vị với xử lý biên
        filtered_phase = medfilt2(phase_with_boundary, filter_size, 'symmetric');
        
        % Cắt về kích thước ban đầu nếu cần
        if ~isequal(size(filtered_phase), [rows, cols])
            filtered_phase = filtered_phase(1:rows, 1:cols);
        end
        
        % Bước 3: Tính toán sự khác biệt về "thứ tự vân" 
        % delta_k = Round[(Phi_filtered - Phi_current) / 2π]
        delta_k = round((filtered_phase - current_phase) / (2*pi));
        
        % Bước 4: Áp dụng các ràng buộc
        % Giới hạn |delta_k|
        delta_k = sign(delta_k) .* min(abs(delta_k), max_delta_k);
        
        % Bảo vệ vùng biên và các pixel không hợp lệ
        delta_k(protection_mask) = 0;
        
        % Bước 5: Kiểm tra tính liên tục không gian (spatial continuity constraint)
        delta_k = apply_spatial_continuity_constraint(delta_k, current_phase);
        
        % Tính toán metric hội tụ (RMS của delta_k chỉ trên vùng được phép thay đổi)
        active_pixels = ~protection_mask;
        if sum(active_pixels(:)) > 0
            rms_delta_k = sqrt(mean((delta_k(active_pixels)).^2));
        else
            rms_delta_k = 0;
        end
        
        convergence_history(end+1) = rms_delta_k;
        num_iterations = iter;
        
        if verbose
            num_corrections = sum(delta_k(:) ~= 0);
            fprintf('Iteration %d: RMS(delta_k) = %.6f, Corrections: %d, Unique values: %d\n', ...
                    iter, rms_delta_k, num_corrections, length(unique(delta_k(:))));
        end
        
        % Kiểm tra điều kiện hội tụ
        if iter > 1
            % Kiểm tra xem delta_k có thay đổi không
            if isequal(delta_k, previous_delta_k)
                if verbose
                    fprintf('Hội tụ đạt được: delta_k không thay đổi (iteration %d)\n', iter);
                end
                break;
            end
            
            % Kiểm tra xem thay đổi có nhỏ hơn epsilon không
            if rms_delta_k < epsilon
                if verbose
                    fprintf('Hội tụ đạt được: RMS(delta_k) < epsilon (iteration %d)\n', iter);
                end
                break;
            end
            
            % Kiểm tra thay đổi tương đối giữa các lần lặp
            relative_change = abs(convergence_history(end) - convergence_history(end-1)) / ...
                             (convergence_history(end-1) + eps);
            if relative_change < epsilon
                if verbose
                    fprintf('Hội tụ đạt được: Thay đổi tương đối < epsilon (iteration %d)\n', iter);
                end
                break;
            end
        end
        
        % Bước 3: Hiệu chỉnh pha với ràng buộc biên
        % Phi_corrected = Phi_current + delta_k * 2π
        current_phase = current_phase + delta_k * (2*pi);
        
        % Khôi phục giá trị biên gốc nếu cần
        if preserve_boundary
            current_phase(protection_mask) = original_phase(protection_mask);
        end
        
        % Lưu delta_k hiện tại để so sánh ở lần lặp tiếp theo
        previous_delta_k = delta_k;
        
        % Kiểm tra nếu đạt số lần lặp tối đa
        if iter == max_iterations
            if verbose
                fprintf('Cảnh báo: Đạt số lần lặp tối đa (%d) mà chưa hội tụ hoàn toàn\n', max_iterations);
            end
        end
    end
    
    corrected_unwrapped_phase = current_phase;
    
    if verbose
        fprintf('Hoàn thành sau %d lần lặp\n', num_iterations);
        fprintf('RMS cuối cùng của delta_k: %.6f\n', convergence_history(end));
    end
end

function [fringe_order, fringe_labels, processed_image] = assign_fringe_order(input_image, display_result)
% ASSIGN_FRINGE_ORDER Gán bậc vân cho ảnh hologram đã được skeletonize
%
% Hàm này thực hiện gán nhãn bậc vân dựa trên vị trí tương đối so với tâm ảnh.
% Vân gần tâm nhất được gán bậc 0, các vân phía trên có bậc dương tăng dần,
% các vân phía dưới có bậc âm giảm dần.
%
% INPUT:
%   input_image    - Ảnh binary đã được skeletonize
%   display_result - (Optional) true/false để hiển thị kết quả (default: true)
%
% OUTPUT:
%   fringe_order     - Số lượng vân được phát hiện
%   fringe_labels    - Vector chứa nhãn bậc vân của từng vùng liên thông
%   processed_image  - Ảnh đã được cắt biên và xử lý
%
% EXAMPLE:
%   [order, labels, img] = assign_fringe_order(skeleton_image);
%   [order, labels, img] = assign_fringe_order(skeleton_image, false); % Không hiển thị

% --- Xử lý tham số đầu vào ---
if nargin < 1
    error('Thiếu tham số đầu vào: input_image');
end

if nargin < 2
    display_result = true; % Mặc định hiển thị kết quả
end

% --- Kiểm tra đầu vào ---
if isempty(input_image)
    error('Ảnh đầu vào không được để trống');
end

if ~islogical(input_image) && ~(isnumeric(input_image) && all(input_image(:) == 0 | input_image(:) == 1))
    error('Ảnh đầu vào phải là ảnh binary (logical hoặc 0/1)');
end

% Chuyển đổi sang logical nếu cần
if ~islogical(input_image)
    input_image = logical(input_image);
end

try
    % --- Bước 1: Cắt biên ảnh để tránh ảnh hưởng vùng biên ---
    offset = 0;
    [orig_H, orig_W] = size(input_image);

    % Kiểm tra kích thước ảnh
    if orig_H <= 2*offset || orig_W <= 2*offset
        warning('Ảnh quá nhỏ để cắt biên. Sử dụng ảnh gốc.');
        bw_crop = input_image;
        offset = 0;
    else
        bw_crop = input_image(offset+1:end-offset, offset+1:end-offset);
    end

    [H, W] = size(bw_crop);

    % --- Bước 2: Tìm các vùng liên thông (vân) ---

    cc = bwconncomp(bw_crop);

    if cc.NumObjects == 0
        warning('Không tìm thấy vân nào trong ảnh');
        fringe_order = 0;
        fringe_labels = [];
        processed_image = bw_crop;
        return;
    end

    labeled_matrix = labelmatrix(cc);
    stats = regionprops(cc, 'Centroid', 'BoundingBox');

    % --- Bước 3: Tìm nhóm gần tâm nhất làm gốc ---
    centroids = cat(1, stats.Centroid);
    image_center = [W/2, H/2];
    dist = vecnorm(centroids - image_center, 2, 2);
    [~, idx_center] = min(dist);

    % --- Bước 4: Khởi tạo và gán nhãn ---
    labels = nan(cc.NumObjects, 1);
    labels(idx_center) = 0; % Nhóm gốc đặt nhãn 0

    queue = idx_center; % Hàng đợi để duyệt lan truyền nhãn
    processed_groups = false(cc.NumObjects, 1);
    processed_groups(idx_center) = true;

    % --- Bước 5: Lan truyền nhãn ---
    while ~isempty(queue)
        current_group = queue(1);
        queue(1) = [];

        current_label = labels(current_group);
        pixels = cc.PixelIdxList{current_group};
        [rows, cols] = ind2sub([H, W], pixels);

        visited_gid = []; % Tránh xét lại nhóm cùng vòng lặp

        for i = 1:length(rows)
            r = rows(i);
            c = cols(i);

            % Lan truyền lên trên theo cột
            for y = r-1:-1:1
                gid = labeled_matrix(y, c);
                if gid > 0 && ~processed_groups(gid) && ~ismember(gid, visited_gid)
                    labels(gid) = current_label + 1; % Nhãn tăng dần lên trên
                    queue(end+1) = gid;
                    processed_groups(gid) = true;
                    visited_gid(end+1) = gid;
                    break;
                elseif gid > 0 && processed_groups(gid)
                    break;
                end
            end

            % Lan truyền xuống dưới theo cột
            for y = r+1:H
                gid = labeled_matrix(y, c);
                if gid > 0 && ~processed_groups(gid) && ~ismember(gid, visited_gid)
                    labels(gid) = current_label - 1; % Nhãn giảm dần xuống dưới
                    queue(end+1) = gid;
                    processed_groups(gid) = true;
                    visited_gid(end+1) = gid;
                    break;
                elseif gid > 0 && processed_groups(gid)
                    break;
                end
            end
        end
    end

    % --- Bước 6: Chuẩn hóa nhãn thành số nguyên dương bắt đầu từ 1 ---
    valid_labels = labels(~isnan(labels));

    if isempty(valid_labels)
        warning('Không thể gán nhãn cho bất kỳ vân nào');
        fringe_order = 0;
        fringe_labels = [];
        processed_image = bw_crop;
        return;
    end

    unique_labels = unique(valid_labels);
    labels_new = nan(size(labels));
    for i = 1:length(unique_labels)
        labels_new(labels == unique_labels(i)) = i;
    end
    labels = labels_new;

    % --- Bước 7: Hiển thị kết quả (nếu được yêu cầu) ---
    if display_result
        figure('Name', 'Kết quả gán bậc vân', 'NumberTitle', 'off');
        imshow(bw_crop);
        hold on;

        for k = 1:cc.NumObjects
            if ~isnan(labels(k))
                pixels = cc.PixelIdxList{k};
                [rows, cols] = ind2sub([H, W], pixels);
                coords = [cols, rows]; % [x, y]

                % Tính khoảng cách từ tâm ảnh để đặt nhãn ở vị trí gần tâm nhất
                dists = sqrt((coords(:,1) - image_center(1)).^2 + (coords(:,2) - image_center(2)).^2);
                [~, min_idx] = min(dists);
                label_pos = coords(min_idx, :);

                text(label_pos(1), label_pos(2), num2str(labels(k)), ...
                    'Color', 'r', 'FontSize', 11, 'FontWeight', 'bold', ...
                    'HorizontalAlignment', 'center');
            end
        end

        title('Gán bậc vân', 'FontSize', 12);
        hold off;
    end

    % --- Bước 8: Trả về kết quả ---
    fringe_order = cc.NumObjects;
    fringe_labels = labels;
    processed_image = bw_crop;

    % Hiển thị thống kê
    fprintf('Đã phát hiện %d vân\n', fringe_order);
    fprintf('Số vân được gán nhãn: %d\n', sum(~isnan(labels)));
    if ~isempty(valid_labels)
        fprintf('Phạm vi bậc vân: %d đến %d\n', min(unique_labels), max(unique_labels));
    end

catch ME
    % Xử lý lỗi
    error_msg = sprintf('Lỗi trong quá trình gán bậc vân:\n%s', ME.message);
    error(error_msg);
end

end



%% thêm 9-7-25
function varargout = crop_multiple_to_smallest(varargin)
    % Giả định tất cả các biến là 2D ma trận
    n = nargin;
    sizes = cellfun(@size, varargin, 'UniformOutput', false);

    % Tìm kích thước nhỏ nhất theo từng chiều
    min_rows = min(cellfun(@(s) s(1), sizes));
    min_cols = min(cellfun(@(s) s(2), sizes));

    varargout = cell(1, n);
    for i = 1:n
        mat = varargin{i};
        [m, n_] = size(mat);
        
        % Tính chỉ số cắt đều 4 phía
        row_start = floor((m - min_rows)/2) + 1;
        col_start = floor((n_ - min_cols)/2) + 1;
        row_end = row_start + min_rows - 1;
        col_end = col_start + min_cols - 1;
        
        varargout{i} = mat(row_start:row_end, col_start:col_end);
    end
end

%% them ngay 14/8/2025
function im_unwrapped = goldstein_unwrap(phase_wrapped)
    % GOLDSTEIN_UNWRAP - Phase unwrapping theo phương pháp Goldstein
    % Input:
    %   IM  - ảnh phức (complex image), IM = mag .* exp(1i * wrapped_phase)
    % Output:
    %   im_unwrapped - ảnh pha đã unwrap

    % 1. Khởi tạo
    % Biên độ (magnitude) = 1 
    mag = ones(size(phase_wrapped));
    IM = mag .* exp(1i * phase_wrapped);   

    im_mag   = abs(IM);       % Magnitude
    im_phase = angle(IM);     % Wrapped phase
    im_mask  = ones(size(IM));

    % 2. Tính residues
    residue_charge = PhaseResidues_r1(im_phase, im_mask);

    % 3. Tạo branch cuts
    max_box_radius = 4;
    branch_cuts = BranchCuts_r1(residue_charge, max_box_radius, im_mask);

    % 4. Loại branch cuts khỏi mask
    im_mask(branch_cuts) = 0;
    im_mag1 = im_mag .* im_mask;

    % 5. Chọn điểm tham chiếu (tự động chọn magnitude lớn nhất)
    [r_dim, c_dim] = size(im_phase);
    im_mag1([1 r_dim], :) = 0;
    im_mag1(:, [1 c_dim]) = 0;
    [~, idx_max] = max(im_mag1(:));
    [rowref, colref] = ind2sub(size(im_mag1), idx_max);

    % 6. Unwrap
    im_unwrapped = FloodFill_r1(im_phase, im_mag, branch_cuts, im_mask, colref, rowref);
end

function [recons_surface, figure_handle] = reconSurface_linearPushed(BW, fringe_labels, lambda, tilt_option, show_figure)
% RECONSURFACE_LINEARPUSHED Tái tạo bề mặt 3D từ ảnh vân giao thoa
%
% Cú pháp:
%   [recons_surface, figure_handle] = reconSurface_linearPushed(BW, fringe_labels, lambda, tilt_option, show_figure)
%
% Tham số đầu vào:
%   BW            - Ảnh nhị phân đã cắt biên (logical matrix)
%   fringe_labels - Vector chứa nhãn của các vân (double array)
%   lambda        - Bước sóng ánh sáng (double)
%   tilt_option   - Tùy chọn xử lý ('None', 'Remove tilt', 'Invert', 'Remove + Invert')
%   show_figure   - Có hiển thị figure hay không (logical, optional, default: true)
%
% Tham số đầu ra:
%   recons_surface - Ma trận bề mặt 3D đã tái tạo
%   figure_handle  - Handle của figure (nếu show_figure = true)
%
% Ví dụ:
%   [surface, fig] = reconSurface_linearPushed(BW, [1,2,3,4,5], 632.8e-9, 'Remove tilt');

% Xử lý tham số đầu vào
if nargin < 5
    show_figure = true;
end

% Kiểm tra tham số đầu vào
if isempty(fringe_labels)
    error('Bạn cần gán nhãn vân trước khi nội suy.');
end

if ~islogical(BW)
    error('BW phải là ảnh nhị phân (logical matrix).');
end

% Thiết lập khoảng cách giữa các vân
khoang_cach_van = lambda/2;

% Tìm các thành phần liên thông
cc = bwconncomp(BW);
L = labelmatrix(cc);

% Khởi tạo các mảng điểm 3D
num_labels = max(L(:));
X = []; Y = []; Z = [];

for i = 1:num_labels
    % Tìm các điểm thuộc vân có nhãn i
    [y, x] = find(L == i);

    if i <= length(fringe_labels)
        % Tính độ cao z dựa trên nhãn vân
        z = ones(size(x)) * (fringe_labels(i)) * khoang_cach_van;
        X = [X; x];
        Y = [Y; y];
        Z = [Z; z];
    end
end

% Kiểm tra xem có dữ liệu để nội suy không
if isempty(X)
    error('Không có dữ liệu để nội suy. Kiểm tra lại fringe_labels và BW.');
end

% Nội suy để tạo mặt 3D mượt
[xq, yq] = meshgrid(1:size(BW,2), 1:size(BW,1));
% F = scatteredInterpolant(X, Y, Z, 'natural', 'nearest');
% Zq = F(xq, yq);

F = scatteredInterpolant(X,Y,Z,'natural','nearest');
F.ExtrapolationMethod = 'nearest';
Zq = F(xq,yq);

Zq(~isfinite(Zq)) = 0;

% %
% Z_grid_cubic = griddata(X, Y, Z, xq, yq, 'cubic');
% Z_grid_cubic(~isfinite(Z_grid_cubic)) = 0;
% 
% % 6. Làm mượt hậu xử lý cho cubic
% Z_cubic_smooth = imgaussfilt(Z_grid_cubic, 2);
% Zq = Z_cubic_smooth;
% %

% Chuyển từ mét sang radian
phi_rad = (4 * pi / lambda) * Zq;
Zq = phi_rad;

% Cắt biên để hiển thị tốt hơn

Z_crop = Zq;


[M, N] = size(Z_crop);
[xGrid, yGrid] = meshgrid(1:N, 1:M);
x = xGrid(:);
y = yGrid(:);
z = Z_crop(:);

% Xử lý theo lựa chọn của người dùng
switch tilt_option
    case 'None'
        Z_processed = Z_crop;

    case 'Remove tilt'
        good = ~isnan(z);
        if sum(good) < 3
            warning('Không đủ điểm hợp lệ để loại bỏ độ nghiêng.');
            Z_processed = Z_crop;
        else
            A = [x, y, ones(size(x))];
            coeff = A(good,:) \ z(good);
            Z_fit = reshape(A * coeff, size(Z_crop));
            Z_processed = Z_crop - Z_fit;
        end

    case 'Invert'
        Z_processed = max(Z_crop(:)) - Z_crop;

    case 'Remove + Invert'
        good = ~isnan(z);
        if sum(good) < 3
            warning('Không đủ điểm hợp lệ để loại bỏ độ nghiêng.');
            Z_leveled = Z_crop;
        else
            A = [x, y, ones(size(x))];
            coeff = A(good,:) \ z(good);
            Z_fit = reshape(A * coeff, size(Z_crop));
            Z_leveled = Z_crop - Z_fit;
        end
        Z_processed = max(Z_leveled(:)) - Z_leveled;

    otherwise
        warning('Tùy chọn không hợp lệ. Sử dụng "None".');
        Z_processed = Z_crop;
end

% Chuẩn hóa bắt đầu từ 0
Z_offset = Z_processed - min(Z_processed(:));

% Gán kết quả đầu ra
recons_surface = Z_offset;

% Hiển thị bề mặt 3D nếu được yêu cầu
if show_figure
    figure_handle = figure;
    surf(xGrid, yGrid, Z_offset);
    shading interp;
    xlabel('X (px)');
    ylabel('Y (px)');
    zlabel('rad');
    title(['3D Surface Linear (Option: ', tilt_option, ')']);
    colormap parula;
    colorbar;
else
    figure_handle = [];
end

end

