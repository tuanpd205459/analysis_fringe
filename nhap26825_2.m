%% ===================================================================
%  COMPLETE BRIDGE REMOVAL SYSTEM FOR SKELETON IMAGES
%  Author: tuanpd205459
%  Date: 2025-08-26
%% ===================================================================

clc; clear; close all;

%% 1. KHỞI TẠO VÀ MÔ PHỎNG HOLOGRAM
fprintf('=== BẮT ĐẦU QUY TRÌNH MÔ PHỎNG VÀ TÁI TẠO ===\n');

%% 2. MÔ PHỎNG HOLOGRAM
fprintf('--> Bước 1: Mô phỏng Hologram...\n');
% Thiết lập thông số
M = 512; N = 512; snr = 15;
sigma = pi/5; noise_level = 0;
noise = noise_level * randn(N, N) .* sigma;

[X, Y] = meshgrid(linspace(-1,1,N), linspace(-1,1,M));
object_phase_without_noise = 2 * peaks(3*X, 3*Y);

% Thêm nhiễu vào pha đối tượng
object_phase = awgn(object_phase_without_noise, snr, 'measured', 'db');

% Hiển thị pha gốc
figure('Name', 'Đối tượng pha gốc');
surf(object_phase_without_noise, "EdgeColor", "none");
colorbar; title('Đối tượng pha (không nhiễu)');

%% 3. TẠO HOLOGRAM
fprintf('--> Bước 2: Tạo Hologram...\n');
fx = 40 / N; fy = -60 / M;
[X, Y] = meshgrid(1:N, 1:M);
a = 1.0; b = 0.8; % Background intensity và modulation depth
carrier = 2 * pi * (fx * X + fy * Y);
hologram = a + b .* cos(carrier + object_phase);

figure('Name', 'Hologram gốc');
imshow(hologram, []); title('Ảnh Hologram (Giao thoa) có nhiễu');

%% 4. XỬ LÝ HOLOGRAM
hologram = mat2gray(hologram);
imwrite(hologram, 'hologram.bmp');

% Noise removal
hologram = imgaussfilt(hologram, 1);
hologram = medfilt2(hologram, [3 3]);
hologram = wiener2(hologram, [5 5]);

% Histogram equalization
hologram = adapthisteq(hologram);
input_image = hologram;

%% 5. SKELETONIZATION
fprintf('--> Bước 3: Skeletonization...\n');
skeleton_result = performSkeletonization(input_image);
skeleton = skeleton_result.skeleton;

%% 6. BRIDGE REMOVAL
fprintf('--> Bước 4: Bridge Removal...\n');
% Với tham số mặc định
result_default = removeBridgesWithOrientationField(skeleton);

% Với tham số tùy chỉnh
result_custom = removeBridgesWithOrientationField(skeleton, ...
    'blockSize', 16, ...
    'angleThreshold', pi/4, ...
    'inconsistencyThreshold', 0.6, ...
    'minLength', 10);

%% 7. HIỂN THỊ KẾT QUẢ
fprintf('--> Bước 5: Hiển thị kết quả...\n');
visualizeCompleteResults(input_image, skeleton, result_custom);

% Đánh giá hiệu suất
metrics = evaluateBridgeRemoval(result_custom.originalSkeleton, result_custom.cleanedSkeleton);
displayMetrics(metrics);

fprintf('=== HOÀN THÀNH QUY TRÌNH ===\n');

%% ===================================================================
%  MAIN FUNCTIONS
%% ===================================================================

function skeleton_result = performSkeletonization(input_image)
    % Thực hiện skeletonization bằng thuật toán Zhang-Suen
    
    fprintf('Bắt đầu quá trình skeletonization...\n');
    
    % Chuyển đổi sang ảnh xám nếu cần
    if size(input_image, 3) == 3
        input_image = rgb2gray(input_image);
        fprintf('Đã chuyển đổi ảnh RGB sang grayscale\n');
    end
    
    % Bước 1: Nhị phân hóa ảnh bằng Otsu
    fprintf('Bước 1/3: Nhị phân hóa ảnh bằng phương pháp Otsu...\n');
    thresh = graythresh(input_image);
    BW_Original = imbinarize(input_image, thresh);
    fprintf('Ngưỡng Otsu: %.4f\n', thresh);
    
    % Bước 2: Skeletonize bằng Zhang-Suen
    fprintf('Bước 2/3: Áp dụng thuật toán Zhang-Suen...\n');
    BW_Thinned = zhangSuenThinning(BW_Original);
    
    % Bước 3: Hiển thị kết quả
    fprintf('Bước 3/3: Hiển thị kết quả skeletonization...\n');
    displaySkeletonizationResults(input_image, BW_Original, BW_Thinned);
    
    % Trả về kết quả
    skeleton_result.original = input_image;
    skeleton_result.binary = BW_Original;
    skeleton_result.skeleton = BW_Thinned;
    skeleton_result.threshold = thresh;
end

function skeleton = zhangSuenThinning(binaryImage)
    % Thuật toán Zhang-Suen thinning
    
    BW_Thinned = binaryImage;
    [rows, cols] = size(BW_Thinned);
    changing = true;
    iteration = 0;
    
    while changing
        iteration = iteration + 1;
        changing = false;
        BW_Del = true(rows, cols);
        
        % Step 1 của Zhang-Suen
        for i = 2:rows-1
            for j = 2:cols-1
                P = BW_Thinned(i-1:i+1, j-1:j+1);
                P = P(:)';
                P = [P(5), P(2), P(3), P(6), P(9), P(8), P(7), P(4), P(1), P(2)];
                
                if P(1) == 1
                    neighbors = sum(P(2:9));
                    transitions = sum(P(2:9) == 0 & P(3:10) == 1);
                    
                    if neighbors >= 2 && neighbors <= 6 && transitions == 1 ...
                            && P(2)*P(4)*P(6) == 0 && P(4)*P(6)*P(8) == 0
                        BW_Del(i,j) = false;
                        changing = true;
                    end
                end
            end
        end
        BW_Thinned = BW_Thinned & BW_Del;
        
        % Step 2 của Zhang-Suen
        BW_Del = true(rows, cols);
        for i = 2:rows-1
            for j = 2:cols-1
                P = BW_Thinned(i-1:i+1, j-1:j+1);
                P = P(:)';
                P = [P(5), P(2), P(3), P(6), P(9), P(8), P(7), P(4), P(1), P(2)];
                
                if P(1) == 1
                    neighbors = sum(P(2:9));
                    transitions = sum(P(2:9) == 0 & P(3:10) == 1);
                    
                    if neighbors >= 2 && neighbors <= 6 && transitions == 1 ...
                            && P(2)*P(4)*P(8) == 0 && P(2)*P(6)*P(8) == 0
                        BW_Del(i,j) = false;
                        changing = true;
                    end
                end
            end
        end
        BW_Thinned = BW_Thinned & BW_Del;
        
        % Hiển thị tiến trình
        if mod(iteration, 10) == 0
            fprintf('  Iteration %d: %d pixels còn lại\n', iteration, sum(BW_Thinned(:)));
        end
        
        % Tránh vòng lặp vô hạn
        if iteration > 1000
            warning('Đã đạt giới hạn iteration (1000). Dừng thuật toán.');
            break;
        end
    end
    
    fprintf('Hoàn thành sau %d iterations\n', iteration);
    fprintf('Số pixel skeleton: %d\n', sum(BW_Thinned(:)));
    skeleton = BW_Thinned;
end

function result = removeBridgesWithOrientationField(skeletonImage, varargin)
    % Hàm chính để loại bỏ spurious bridges sử dụng orientation field
    
    % Parse input parameters
    p = inputParser;
    addRequired(p, 'skeletonImage');
    addParameter(p, 'blockSize', 16);
    addParameter(p, 'angleThreshold', pi/4);
    addParameter(p, 'inconsistencyThreshold', 0.6);
    addParameter(p, 'minLength', 10);
    addParameter(p, 'sigma', 2.0);
    parse(p, skeletonImage, varargin{:});
    
    params = p.Results;
    
    fprintf('Starting bridge removal pipeline...\n');
    
    % Step 1: Validate và prepare skeleton image
    fprintf('Step 1: Validating skeleton image...\n');
    if ~islogical(skeletonImage)
        skeleton = skeletonImage > 0;
    else
        skeleton = skeletonImage;
    end
    skeleton = bwmorph(skeleton, 'skel', Inf);
    
    % Step 2: Tính orientation field từ skeleton
    fprintf('Step 2: Computing orientation field from skeleton...\n');
    orientationField = computeOrientationFieldFromSkeleton(skeleton, params.blockSize);
    orientationField = smoothOrientationField(orientationField, params.sigma);
    
    % Step 3: Extract skeleton segments
    fprintf('Step 3: Extracting skeleton segments...\n');
    segments = extractSkeletonSegments(skeleton);
    
    % Step 4: Phân tích và loại bỏ bridges
    fprintf('Step 4: Analyzing segments and removing bridges...\n');
    [cleanedSkeleton, removedSegments] = removeSpuriousBridges(...
        skeleton, segments, orientationField, params);
    
    % Step 5: Post-processing
    fprintf('Step 5: Post-processing...\n');
    finalSkeleton = postProcessSkeleton(cleanedSkeleton);
    
    % Chuẩn bị kết quả
    result = struct();
    result.originalSkeleton = skeleton;
    result.cleanedSkeleton = finalSkeleton;
    result.orientationField = orientationField;
    result.removedSegments = removedSegments;
    result.segments = segments;
    result.numOriginalPoints = sum(skeleton(:));
    result.numCleanedPoints = sum(finalSkeleton(:));
    result.numRemovedPoints = result.numOriginalPoints - result.numCleanedPoints;
    result.params = params;
    
    fprintf('Pipeline completed. Removed %d points from skeleton.\n', ...
            result.numRemovedPoints);
end

%% ===================================================================
%  ORIENTATION FIELD COMPUTATION
%% ===================================================================

function orientationField = computeOrientationFieldFromSkeleton(skeleton, blockSize)
    % Tính orientation field từ skeleton image
    
    [rows, cols] = size(skeleton);
    orientationField = zeros(ceil(rows/blockSize), ceil(cols/blockSize));
    
    for i = 1:blockSize:rows
        for j = 1:blockSize:cols
            rowEnd = min(i + blockSize - 1, rows);
            colEnd = min(j + blockSize - 1, cols);
            block = skeleton(i:rowEnd, j:colEnd);
            
            if sum(block(:)) > 2
                orientation = computeLocalOrientation(block);
                orientationField(ceil(i/blockSize), ceil(j/blockSize)) = orientation;
            end
        end
    end
end

function orientation = computeLocalOrientation(block)
    % Tính local orientation của skeleton pixels trong một block
    
    [y, x] = find(block);
    
    if length(x) < 2
        orientation = 0;
        return;
    end
    
    % Tính principal direction bằng PCA
    coords = [x - mean(x), y - mean(y)];
    
    if size(coords, 1) < 2
        orientation = 0;
        return;
    end
    
    covMatrix = cov(coords);
    [eigVec, eigVal] = eig(covMatrix);
    
    % Principal direction là eigenvector với eigenvalue lớn nhất
    [~, maxIdx] = max(diag(eigVal));
    principalDir = eigVec(:, maxIdx);
    orientation = atan2(principalDir(2), principalDir(1));
    
    % Normalize về [0, pi)
    if orientation < 0
        orientation = orientation + pi;
    end
end

function smoothedField = smoothOrientationField(orientationField, sigma)
    % Smooth orientation field sử dụng circular statistics
    
    % Chuyển đổi sang complex representation
    complexField = exp(2i * orientationField);
    
    % Tạo Gaussian kernel
    kernelSize = 2 * ceil(3 * sigma) + 1;
    kernel = fspecial('gaussian', kernelSize, sigma);
    
    % Smooth complex field
    smoothedComplex = imfilter(complexField, kernel, 'replicate');
    
    % Chuyển về angle
    smoothedField = 0.5 * angle(smoothedComplex);
end

%% ===================================================================
%  SKELETON SEGMENTATION
%% ===================================================================

function segments = extractSkeletonSegments(skeleton)
    % Extract các connected segments từ skeleton
    
    segments = {};
    visited = false(size(skeleton));
    
    % Tìm endpoints (pixels có 1 neighbor)
    endpoints = findEndpoints(skeleton);
    
    % Tìm branch points (pixels có 3+ neighbors)
    branchPoints = findBranchPoints(skeleton);
    
    % Đánh dấu branch points đã visited
    for i = 1:size(branchPoints, 1)
        visited(branchPoints(i, 1), branchPoints(i, 2)) = true;
    end
    
    % Trace từ mỗi endpoint
    for i = 1:size(endpoints, 1)
        startPoint = endpoints(i, :);
        if ~visited(startPoint(1), startPoint(2))
            segment = traceSkeletonPath(skeleton, startPoint, visited);
            if length(segment) > 5
                segments{end+1} = segment;
            end
        end
    end
    
    % Trace giữa các branch points
    for i = 1:size(branchPoints, 1)
        startPoint = branchPoints(i, :);
        neighbors = getSkeletonNeighbors(skeleton, startPoint);
        
        for j = 1:size(neighbors, 1)
            nextPoint = neighbors(j, :);
            if ~visited(nextPoint(1), nextPoint(2))
                segment = traceSkeletonPath(skeleton, nextPoint, visited);
                if length(segment) > 3
                    segments{end+1} = [startPoint; segment];
                end
            end
        end
    end
end

function endpoints = findEndpoints(skeleton)
    % Tìm endpoints của skeleton (pixels có đúng 1 neighbor)
    
    kernel = [1 1 1; 1 0 1; 1 1 1];
    neighborCount = conv2(double(skeleton), kernel, 'same');
    [rows, cols] = find(skeleton & neighborCount == 1);
    endpoints = [rows, cols];
end

function branchPoints = findBranchPoints(skeleton)
    % Tìm branch points (pixels có 3+ neighbors)
    
    kernel = [1 1 1; 1 0 1; 1 1 1];
    neighborCount = conv2(double(skeleton), kernel, 'same');
    [rows, cols] = find(skeleton & neighborCount >= 3);
    branchPoints = [rows, cols];
end

function segment = traceSkeletonPath(skeleton, startPoint, visited)
    % Trace một path dọc skeleton từ starting point
    
    segment = startPoint;
    visited(startPoint(1), startPoint(2)) = true;
    currentPoint = startPoint;
    
    while true
        neighbors = getSkeletonNeighbors(skeleton, currentPoint);
        unvisitedNeighbors = [];
        
        for i = 1:size(neighbors, 1)
            if ~visited(neighbors(i, 1), neighbors(i, 2))
                unvisitedNeighbors = [unvisitedNeighbors; neighbors(i, :)];
            end
        end
        
        if isempty(unvisitedNeighbors)
            break;
        end
        
        nextPoint = unvisitedNeighbors(1, :);
        segment = [segment; nextPoint];
        visited(nextPoint(1), nextPoint(2)) = true;
        currentPoint = nextPoint;
    end
end

function neighbors = getSkeletonNeighbors(skeleton, point)
    % Lấy 8-connected skeleton neighbors của một point
    
    [rows, cols] = size(skeleton);
    r = point(1); c = point(2);
    neighbors = [];
    
    for dr = -1:1
        for dc = -1:1
            if dr == 0 && dc == 0
                continue;
            end
            
            nr = r + dr; nc = c + dc;
            
            if nr >= 1 && nr <= rows && nc >= 1 && nc <= cols
                if skeleton(nr, nc)
                    neighbors = [neighbors; nr, nc];
                end
            end
        end
    end
end

%% ===================================================================
%  BRIDGE ANALYSIS AND REMOVAL
%% ===================================================================

function [cleanedSkeleton, removedSegments] = removeSpuriousBridges(...
    skeleton, segments, orientationField, params)
    % Loại bỏ spurious bridges dựa trên orientation field analysis
    
    cleanedSkeleton = skeleton;
    removedSegments = {};
    
    for i = 1:length(segments)
        segment = segments{i};
        
        if isSpuriousBridge(segment, orientationField, params)
            % Loại bỏ segment này từ skeleton
            for j = 1:size(segment, 1)
                cleanedSkeleton(segment(j, 1), segment(j, 2)) = false;
            end
            removedSegments{end+1} = segment;
        end
    end
    
    fprintf('Removed %d spurious bridge segments.\n', length(removedSegments));
end

function isBridge = isSpuriousBridge(segment, orientationField, params)
    % Xác định xem segment có phải spurious bridge không
    
    if size(segment, 1) < 2
        isBridge = false;
        return;
    end
    
    % Điều kiện 1: Kiểm tra độ dài segment
    if size(segment, 1) > params.minLength
        isBridge = false;
        return;
    end
    
    % Điều kiện 2: Kiểm tra orientation consistency
    inconsistencyRatio = analyzeOrientationConsistency(segment, orientationField, params);
    
    isBridge = inconsistencyRatio > params.inconsistencyThreshold;
end

function inconsistencyRatio = analyzeOrientationConsistency(segment, orientationField, params)
    % Phân tích mức độ segment tuân theo orientation field
    
    if size(segment, 1) < 2
        inconsistencyRatio = 0;
        return;
    end
    
    % Tính segment orientation
    segmentVector = segment(end, :) - segment(1, :);
    segmentAngle = atan2(segmentVector(1), segmentVector(2));
    
    % Kiểm tra consistency tại mỗi điểm dọc segment
    inconsistentCount = 0;
    totalPoints = 0;
    
    [fieldRows, fieldCols] = size(orientationField);
    blockSize = params.blockSize;
    
    for i = 1:size(segment, 1)
        point = segment(i, :);
        
        % Tính chỉ số trong orientation field
        fieldRow = min(ceil(point(1) / blockSize), fieldRows);
        fieldCol = min(ceil(point(2) / blockSize), fieldCols);
        
        localOrientation = orientationField(fieldRow, fieldCol);
        
        % Tính angular difference
        angleDiff = computeAngularDifference(segmentAngle, localOrientation);
        
        if angleDiff > params.angleThreshold
            inconsistentCount = inconsistentCount + 1;
        end
        totalPoints = totalPoints + 1;
    end
    
    inconsistencyRatio = inconsistentCount / totalPoints;
end

function angleDiff = computeAngularDifference(angle1, angle2)
    % Tính minimum angular difference giữa hai angles
    
    diff = abs(angle1 - angle2);
    angleDiff = min(diff, pi - diff);
end

%% ===================================================================
%  POST-PROCESSING
%% ===================================================================

function cleanedSkeleton = postProcessSkeleton(skeleton)
    % Post-process skeleton để loại bỏ isolated points và small artifacts
    
    % Loại bỏ isolated points (không có neighbors)
    kernel = [1 1 1; 1 0 1; 1 1 1];
    neighborCount = conv2(double(skeleton), kernel, 'same');
    cleanedSkeleton = skeleton & (neighborCount > 0);
    
    % Loại bỏ các connected components rất nhỏ
    cc = bwconncomp(cleanedSkeleton);
    for i = 1:cc.NumObjects
        if length(cc.PixelIdxList{i}) < 5
            cleanedSkeleton(cc.PixelIdxList{i}) = false;
        end
    end
    
    % Optional: morphological cleaning
    cleanedSkeleton = bwmorph(cleanedSkeleton, 'clean');
end

%% ===================================================================
%  EVALUATION AND VISUALIZATION
%% ===================================================================

function metrics = evaluateBridgeRemoval(originalSkeleton, cleanedSkeleton, groundTruth)
    % Đánh giá hiệu quả của bridge removal
    
    metrics = struct();
    
    % Thống kê cơ bản
    metrics.originalPoints = sum(originalSkeleton(:));
    metrics.cleanedPoints = sum(cleanedSkeleton(:));
    metrics.removedPoints = metrics.originalPoints - metrics.cleanedPoints;
    metrics.removalRatio = metrics.removedPoints / metrics.originalPoints;
    
    % Nếu có ground truth
    if nargin > 2 && ~isempty(groundTruth)
        tp = sum(cleanedSkeleton(:) & groundTruth(:));
        fp = sum(cleanedSkeleton(:) & ~groundTruth(:));
        fn = sum(~cleanedSkeleton(:) & groundTruth(:));
        
        metrics.precision = tp / (tp + fp);
        metrics.recall = tp / (tp + fn);
        metrics.f1Score = 2 * metrics.precision * metrics.recall / ...
                         (metrics.precision + metrics.recall);
    end
end

function displayMetrics(metrics)
    % Hiển thị các metrics đánh giá
    
    fprintf('\n=== EVALUATION METRICS ===\n');
    fprintf('Original points: %d\n', metrics.originalPoints);
    fprintf('Cleaned points: %d\n', metrics.cleanedPoints);
    fprintf('Removed points: %d (%.2f%%)\n', metrics.removedPoints, ...
            metrics.removalRatio * 100);
    
    if isfield(metrics, 'f1Score')
        fprintf('Precision: %.3f\n', metrics.precision);
        fprintf('Recall: %.3f\n', metrics.recall);
        fprintf('F1-Score: %.3f\n', metrics.f1Score);
    end
    fprintf('===========================\n\n');
end

function displaySkeletonizationResults(original, binary, skeleton)
    % Hiển thị kết quả skeletonization
    
    figure('Name', 'Kết quả Skeletonization', 'Position', [100, 100, 1200, 400]);
    
    subplot(1, 3, 1);
    imshow(original);
    title('Ảnh gốc', 'FontSize', 12);
    
    subplot(1, 3, 2);
    imshow(binary);
    title('Ảnh nhị phân (Otsu)', 'FontSize', 12);
    
    subplot(1, 3, 3);
    imshow(skeleton);
    title('Skeleton (Zhang-Suen)', 'FontSize', 12);
    
    sgtitle('Quá trình Skeletonization', 'FontSize', 14, 'FontWeight', 'bold');
end

function visualizeCompleteResults(originalImage, skeleton, result)
    % Hiển thị toàn bộ kết quả của hệ thống
    
    figure('Name', 'Kết quả Bridge Removal System', 'Position', [50, 50, 1400, 900]);
    
    % Original image
    subplot(2, 4, 1);
    imshow(originalImage, []);
    title('Ảnh gốc', 'FontSize', 10);
    
    % Original skeleton
    subplot(2, 4, 2);
    imshow(skeleton);
    title(sprintf('Skeleton gốc\n(%d points)', sum(skeleton(:))), 'FontSize', 10);
    
    % Orientation field
    subplot(2, 4, 3);
    visualizeOrientationField(originalImage, result.orientationField);
    title('Orientation Field', 'FontSize', 10);
    
    % Segments
    subplot(2, 4, 4);
    visualizeSegments(skeleton, result.segments);
    title(sprintf('Segments\n(%d segments)', length(result.segments)), 'FontSize', 10);
    
    % Cleaned skeleton
    subplot(2, 4, 5);
    imshow(result.cleanedSkeleton);
    title(sprintf('Skeleton đã làm sạch\n(%d points)', sum(result.cleanedSkeleton(:))), 'FontSize', 10);
    
    % Comparison overlay
    subplot(2, 4, 6);
    overlaySkeletons(originalImage, skeleton, result.cleanedSkeleton);
    title('So sánh\n(Đỏ: Gốc, Xanh: Sạch)', 'FontSize', 10);
    
    % Removed segments
    subplot(2, 4, 7);
    visualizeRemovedSegments(originalImage, result.removedSegments);
    title(sprintf('Bridges đã loại bỏ\n(%d segments)', length(result.removedSegments)), 'FontSize', 10);
    
    % Statistics
    subplot(2, 4, 8);
    displayStatistics(result);
    title('Thống kê', 'FontSize', 10);
    
    sgtitle('Bridge Removal System - Kết quả hoàn chỉnh', 'FontSize', 16, 'FontWeight', 'bold');
end

function visualizeOrientationField(image, orientationField, varargin)
    % Hiển thị orientation field dưới dạng arrows
    
    p = inputParser;
    addParameter(p, 'step', 8);
    addParameter(p, 'scale', 0.8);
    parse(p, varargin{:});
    
    step = p.Results.step;
    scale = p.Results.scale;
    
    imshow(image, []);
    hold on;
    
    [rows, cols] = size(orientationField);
    [X, Y] = meshgrid(1:step:cols, 1:step:rows);
    
    % Sample orientation field
    U = cos(orientationField(1:step:rows, 1:step:cols)) * scale * step;
    V = sin(orientationField(1:step:rows, 1:step:cols)) * scale * step;
    
    quiver(X, Y, U, V, 0, 'r', 'LineWidth', 1);
    hold off;
end

function visualizeSegments(skeleton, segments)
    % Hiển thị các segments với màu sắc khác nhau
    
    imshow(skeleton);
    hold on;
    
    colors = lines(min(length(segments), 10)); % Giới hạn số màu
    
    for i = 1:length(segments)
        segment = segments{i};
        colorIdx = mod(i-1, size(colors, 1)) + 1;
        plot(segment(:, 2), segment(:, 1), 'Color', colors(colorIdx, :), ...
             'LineWidth', 2, 'Marker', 'o', 'MarkerSize', 2);
    end
    
    hold off;
end

function overlaySkeletons(image, originalSkel, cleanedSkel)
    % Overlay original và cleaned skeletons để so sánh
    
    rgbImage = repmat(mat2gray(image), [1, 1, 3]);
    
    % Original skeleton màu đỏ
    rgbImage(:, :, 1) = rgbImage(:, :, 1) + 0.5 * double(originalSkel);
    
    % Cleaned skeleton màu xanh
    rgbImage(:, :, 2) = rgbImage(:, :, 2) + 0.5 * double(cleanedSkel);
    
    rgbImage = min(rgbImage, 1);
    imshow(rgbImage);
end

function visualizeRemovedSegments(image, removedSegments)
    % Hiển thị các removed bridge segments
    
    imshow(image, []);
    hold on;
    
    colors = lines(length(removedSegments));
    
    for i = 1:length(removedSegments)
        segment = removedSegments{i};
        plot(segment(:, 2), segment(:, 1), 'Color', colors(i, :), ...
             'LineWidth', 3, 'Marker', 'o', 'MarkerSize', 4);
    end
    
    hold off;
end

function displayStatistics(result)
    % Hiển thị thống kê dưới dạng text
    
    axis off;
    
    stats_text = {
        sprintf('Tham số:');
        sprintf('- Block Size: %d', result.params.blockSize);
        sprintf('- Angle Threshold: %.2f', result.params.angleThreshold);
        sprintf('- Inconsistency Th: %.2f', result.params.inconsistencyThreshold);
        sprintf('- Min Length: %d', result.params.minLength);
        sprintf('');
        sprintf('Kết quả:');
        sprintf('- Points gốc: %d', result.numOriginalPoints);
        sprintf('- Points sạch: %d', result.numCleanedPoints);
        sprintf('- Points loại bỏ: %d', result.numRemovedPoints);
        sprintf('- Tỷ lệ loại bỏ: %.1f%%', (result.numRemovedPoints/result.numOriginalPoints)*100);
        sprintf('- Segments gốc: %d', length(result.segments));
        sprintf('- Bridges loại bỏ: %d', length(result.removedSegments));
    };
    
    text(0.05, 0.95, stats_text, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
         'FontSize', 9, 'FontName', 'FixedWidth');
end
