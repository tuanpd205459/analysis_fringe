%% ===================================================================
%  LOẠI BỎ CẦU NỐI BẰNG ORIENTATION FIELD - MATLAB IMPLEMENTATION
%  ===================================================================

%% Main Pipeline Function
function result = removeBridgesWithOrientationField(image, varargin)
    % Main function to remove spurious bridges using orientation field
    % 
    % Inputs:
    %   image - Input grayscale image
    %   varargin - Optional parameters
    %
    % Outputs:
    %   result - Structure containing results
    
    % Parse input parameters
    p = inputParser;
    addRequired(p, 'image');
    addParameter(p, 'blockSize', 16);
    addParameter(p, 'angleThreshold', pi/4);
    addParameter(p, 'inconsistencyThreshold', 0.6);
    addParameter(p, 'minLength', 10);
    addParameter(p, 'sigma', 2.0);
    parse(p, image, varargin{:});
    
    params = p.Results;
    
    fprintf('Starting bridge removal pipeline...\n');
    
    % Step 1: Skeletonization
    fprintf('Step 1: Skeletonizing image...\n');
    binaryImage = image > 128;
    skeleton = bwmorph(binaryImage, 'skel', Inf);
    
    % Step 2: Compute Orientation Field
    fprintf('Step 2: Computing orientation field...\n');
    orientationField = computeOrientationField(image, params.blockSize);
    orientationField = smoothOrientationField(orientationField, params.sigma);
    
    % Step 3: Extract skeleton segments
    fprintf('Step 3: Extracting skeleton segments...\n');
    segments = extractSkeletonSegments(skeleton);
    
    % Step 4: Analyze and remove bridges
    fprintf('Step 4: Analyzing segments and removing bridges...\n');
    [cleanedSkeleton, removedSegments] = removeSpuriousBridges(...
        skeleton, segments, orientationField, params);
    
    % Step 5: Post-processing
    fprintf('Step 5: Post-processing...\n');
    finalSkeleton = postProcessSkeleton(cleanedSkeleton);
    
    % Prepare results
    result = struct();
    result.originalSkeleton = skeleton;
    result.cleanedSkeleton = finalSkeleton;
    result.orientationField = orientationField;
    result.removedSegments = removedSegments;
    result.numOriginalPoints = sum(skeleton(:));
    result.numCleanedPoints = sum(finalSkeleton(:));
    result.numRemovedPoints = result.numOriginalPoints - result.numCleanedPoints;
    
    fprintf('Pipeline completed. Removed %d points from skeleton.\n', ...
            result.numRemovedPoints);
end

%% ===================================================================
%  STEP 1: ORIENTATION FIELD COMPUTATION
%% ===================================================================

function orientationField = computeOrientationField(image, blockSize)
    % Compute orientation field using gradient-based method
    
    [rows, cols] = size(image);
    orientationField = zeros(rows, cols);
    
    % Convert to double for gradient computation
    image = double(image);
    
    % Compute gradients
    [Gx, Gy] = gradient(image);
    
    % Process in blocks
    for i = 1:blockSize:rows
        for j = 1:blockSize:cols
            % Define block boundaries
            rowEnd = min(i + blockSize - 1, rows);
            colEnd = min(j + blockSize - 1, cols);
            
            % Extract gradients for current block
            blockGx = Gx(i:rowEnd, j:colEnd);
            blockGy = Gy(i:rowEnd, j:colEnd);
            
            % Compute orientation using structure tensor approach
            Gxx = sum(blockGx(:).^2);
            Gyy = sum(blockGy(:).^2);
            Gxy = sum(blockGx(:) .* blockGy(:));
            
            % Orientation angle: theta = 0.5 * atan2(2*Gxy, Gxx - Gyy)
            if abs(Gxx - Gyy) < 1e-10 && abs(Gxy) < 1e-10
                theta = 0;  % No dominant direction
            else
                theta = 0.5 * atan2(2 * Gxy, Gxx - Gyy);
            end
            
            % Assign to all pixels in the block
            orientationField(i:rowEnd, j:colEnd) = theta;
        end
    end
end

function smoothedField = smoothOrientationField(orientationField, sigma)
    % Smooth orientation field using circular statistics
    
    % Convert to complex representation for circular smoothing
    complexField = exp(2i * orientationField);
    
    % Create Gaussian kernel
    kernelSize = 2 * ceil(3 * sigma) + 1;
    kernel = fspecial('gaussian', kernelSize, sigma);
    
    % Smooth the complex field
    smoothedComplex = imfilter(complexField, kernel, 'replicate');
    
    % Convert back to angle
    smoothedField = 0.5 * angle(smoothedComplex);
end

%% ===================================================================
%  STEP 2: SKELETON SEGMENTATION
%% ===================================================================

function segments = extractSkeletonSegments(skeleton)
    % Extract connected segments from skeleton
    
    segments = {};
    visited = false(size(skeleton));
    
    % Find endpoints (pixels with only 1 neighbor)
    endpoints = findEndpoints(skeleton);
    
    % Find branch points (pixels with 3+ neighbors)
    branchPoints = findBranchPoints(skeleton);
    
    % Mark branch points as visited to prevent crossing
    for i = 1:size(branchPoints, 1)
        visited(branchPoints(i, 1), branchPoints(i, 2)) = true;
    end
    
    % Trace from each endpoint
    for i = 1:size(endpoints, 1)
        startPoint = endpoints(i, :);
        if ~visited(startPoint(1), startPoint(2))
            segment = traceSkeletonPath(skeleton, startPoint, visited);
            if length(segment) > 5  % Only keep segments with minimum length
                segments{end+1} = segment;
            end
        end
    end
    
    % Also trace between branch points
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
    % Find endpoints of skeleton (pixels with exactly 1 neighbor)
    
    % 8-connectivity kernel
    kernel = [1 1 1; 1 0 1; 1 1 1];
    
    % Count neighbors for each skeleton pixel
    neighborCount = conv2(double(skeleton), kernel, 'same');
    
    % Endpoints have exactly 1 neighbor
    [rows, cols] = find(skeleton & neighborCount == 1);
    endpoints = [rows, cols];
end

function branchPoints = findBranchPoints(skeleton)
    % Find branch points (pixels with 3+ neighbors)
    
    % 8-connectivity kernel
    kernel = [1 1 1; 1 0 1; 1 1 1];
    
    % Count neighbors
    neighborCount = conv2(double(skeleton), kernel, 'same');
    
    % Branch points have 3+ neighbors
    [rows, cols] = find(skeleton & neighborCount >= 3);
    branchPoints = [rows, cols];
end

function segment = traceSkeletonPath(skeleton, startPoint, visited)
    % Trace a path along the skeleton from a starting point
    
    segment = startPoint;
    visited(startPoint(1), startPoint(2)) = true;
    currentPoint = startPoint;
    
    while true
        % Find unvisited neighbors
        neighbors = getSkeletonNeighbors(skeleton, currentPoint);
        unvisitedNeighbors = [];
        
        for i = 1:size(neighbors, 1)
            if ~visited(neighbors(i, 1), neighbors(i, 2))
                unvisitedNeighbors = [unvisitedNeighbors; neighbors(i, :)];
            end
        end
        
        if isempty(unvisitedNeighbors)
            break;  % No more unvisited neighbors
        end
        
        % Choose the next point (first unvisited neighbor)
        nextPoint = unvisitedNeighbors(1, :);
        segment = [segment; nextPoint];
        visited(nextPoint(1), nextPoint(2)) = true;
        currentPoint = nextPoint;
    end
end

function neighbors = getSkeletonNeighbors(skeleton, point)
    % Get 8-connected skeleton neighbors of a point
    
    [rows, cols] = size(skeleton);
    r = point(1);
    c = point(2);
    
    neighbors = [];
    
    % Check 8-connected neighbors
    for dr = -1:1
        for dc = -1:1
            if dr == 0 && dc == 0
                continue;  % Skip the center point
            end
            
            nr = r + dr;
            nc = c + dc;
            
            % Check bounds and skeleton membership
            if nr >= 1 && nr <= rows && nc >= 1 && nc <= cols
                if skeleton(nr, nc)
                    neighbors = [neighbors; nr, nc];
                end
            end
        end
    end
end

%% ===================================================================
%  STEP 3: BRIDGE ANALYSIS AND REMOVAL
%% ===================================================================

function [cleanedSkeleton, removedSegments] = removeSpuriousBridges(...
    skeleton, segments, orientationField, params)
    % Remove spurious bridges based on orientation field analysis
    
    cleanedSkeleton = skeleton;
    removedSegments = {};
    
    for i = 1:length(segments)
        segment = segments{i};
        
        if isSpuriousBridge(segment, orientationField, params)
            % Remove this segment from skeleton
            for j = 1:size(segment, 1)
                cleanedSkeleton(segment(j, 1), segment(j, 2)) = false;
            end
            removedSegments{end+1} = segment;
        end
    end
    
    fprintf('Removed %d spurious bridge segments.\n', length(removedSegments));
end

function isBridge = isSpuriousBridge(segment, orientationField, params)
    % Determine if a segment is a spurious bridge
    
    if size(segment, 1) < 2
        isBridge = false;
        return;
    end
    
    % Condition 1: Segment length check
    if size(segment, 1) > params.minLength
        isBridge = false;
        return;
    end
    
    % Condition 2: Orientation consistency check
    inconsistencyRatio = analyzeOrientationConsistency(segment, orientationField, params);
    
    if inconsistencyRatio > params.inconsistencyThreshold
        isBridge = true;
    else
        isBridge = false;
    end
end

function inconsistencyRatio = analyzeOrientationConsistency(segment, orientationField, params)
    % Analyze how well segment follows the orientation field
    
    if size(segment, 1) < 2
        inconsistencyRatio = 0;
        return;
    end
    
    % Compute segment orientation
    segmentVector = segment(end, :) - segment(1, :);
    segmentAngle = atan2(segmentVector(1), segmentVector(2));
    
    % Check consistency at each point along the segment
    inconsistentCount = 0;
    totalPoints = 0;
    
    for i = 1:size(segment, 1)
        point = segment(i, :);
        localOrientation = orientationField(point(1), point(2));
        
        % Compute angular difference
        angleDiff = computeAngularDifference(segmentAngle, localOrientation);
        
        if angleDiff > params.angleThreshold
            inconsistentCount = inconsistentCount + 1;
        end
        totalPoints = totalPoints + 1;
    end
    
    inconsistencyRatio = inconsistentCount / totalPoints;
end

function angleDiff = computeAngularDifference(angle1, angle2)
    % Compute the minimum angular difference between two angles
    
    diff = abs(angle1 - angle2);
    angleDiff = min(diff, pi - diff);
end

%% ===================================================================
%  STEP 4: POST-PROCESSING
%% ===================================================================

function cleanedSkeleton = postProcessSkeleton(skeleton)
    % Post-process skeleton to remove isolated points and small artifacts
    
    % Remove isolated points (no neighbors)
    kernel = [1 1 1; 1 0 1; 1 1 1];
    neighborCount = conv2(double(skeleton), kernel, 'same');
    cleanedSkeleton = skeleton & (neighborCount > 0);
    
    % Remove very small connected components
    cc = bwconncomp(cleanedSkeleton);
    for i = 1:cc.NumObjects
        if length(cc.PixelIdxList{i}) < 5  % Remove components with < 5 pixels
            cleanedSkeleton(cc.PixelIdxList{i}) = false;
        end
    end
    
    % Optional: Apply one iteration of morphological cleaning
    cleanedSkeleton = bwmorph(cleanedSkeleton, 'clean');
end

%% ===================================================================
%  EVALUATION AND VISUALIZATION
%% ===================================================================

function metrics = evaluateBridgeRemoval(originalSkeleton, cleanedSkeleton, groundTruth)
    % Evaluate the effectiveness of bridge removal
    
    metrics = struct();
    
    % Basic statistics
    metrics.originalPoints = sum(originalSkeleton(:));
    metrics.cleanedPoints = sum(cleanedSkeleton(:));
    metrics.removedPoints = metrics.originalPoints - metrics.cleanedPoints;
    metrics.removalRatio = metrics.removedPoints / metrics.originalPoints;
    
    % If ground truth is available
    if nargin > 2 && ~isempty(groundTruth)
        tp = sum(cleanedSkeleton(:) & groundTruth(:));
        fp = sum(cleanedSkeleton(:) & ~groundTruth(:));
        fn = sum(~cleanedSkeleton(:) & groundTruth(:));
        
        metrics.precision = tp / (tp + fp);
        metrics.recall = tp / (tp + fn);
        metrics.f1Score = 2 * metrics.precision * metrics.recall / ...
                         (metrics.precision + metrics.recall);
    end
    
    fprintf('Evaluation Metrics:\n');
    fprintf('  Original points: %d\n', metrics.originalPoints);
    fprintf('  Cleaned points: %d\n', metrics.cleanedPoints);
    fprintf('  Removed points: %d (%.2f%%)\n', metrics.removedPoints, ...
            metrics.removalRatio * 100);
    
    if isfield(metrics, 'f1Score')
        fprintf('  Precision: %.3f\n', metrics.precision);
        fprintf('  Recall: %.3f\n', metrics.recall);
        fprintf('  F1-Score: %.3f\n', metrics.f1Score);
    end
end

function visualizeResults(image, result)
    % Visualize the bridge removal results
    
    figure('Position', [100, 100, 1200, 800]);
    
    % Original image and skeleton
    subplot(2, 3, 1);
    imshow(image, []);
    title('Original Image');
    
    subplot(2, 3, 2);
    imshow(result.originalSkeleton);
    title(sprintf('Original Skeleton (%d points)', result.numOriginalPoints));
    
    % Orientation field visualization
    subplot(2, 3, 3);
    visualizeOrientationField(image, result.orientationField);
    title('Orientation Field');
    
    % Cleaned skeleton
    subplot(2, 3, 4);
    imshow(result.cleanedSkeleton);
    title(sprintf('Cleaned Skeleton (%d points)', result.numCleanedPoints));
    
    % Overlay comparison
    subplot(2, 3, 5);
    overlaySkeletons(image, result.originalSkeleton, result.cleanedSkeleton);
    title('Comparison (Red: Original, Green: Cleaned)');
    
    % Removed segments
    subplot(2, 3, 6);
    visualizeRemovedSegments(image, result.removedSegments);
    title(sprintf('Removed Bridges (%d segments)', length(result.removedSegments)));
end

function visualizeOrientationField(image, orientationField, varargin)
    % Visualize orientation field as overlaid arrows
    
    p = inputParser;
    addParameter(p, 'step', 8);  % Arrow spacing
    addParameter(p, 'scale', 0.8);  % Arrow scale
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

function overlaySkeletons(image, originalSkel, cleanedSkel)
    % Overlay original and cleaned skeletons for comparison
    
    % Create RGB overlay
    rgbImage = repmat(mat2gray(image), [1, 1, 3]);
    
    % Original skeleton in red
    rgbImage(:, :, 1) = rgbImage(:, :, 1) + 0.5 * double(originalSkel);
    
    % Cleaned skeleton in green
    rgbImage(:, :, 2) = rgbImage(:, :, 2) + 0.5 * double(cleanedSkel);
    
    % Clip values
    rgbImage = min(rgbImage, 1);
    
    imshow(rgbImage);
end

function visualizeRemovedSegments(image, removedSegments)
    % Visualize the removed bridge segments
    
    imshow(image, []);
    hold on;
    
    colors = lines(length(removedSegments));
    
    for i = 1:length(removedSegments)
        segment = removedSegments{i};
        plot(segment(:, 2), segment(:, 1), 'Color', colors(i, :), ...
             'LineWidth', 2, 'Marker', 'o', 'MarkerSize', 3);
    end
    
    hold off;
end

%% ===================================================================
%  EXAMPLE USAGE AND DEMO
%% ===================================================================

function runBridgeRemovalDemo()
    % Demo function showing how to use the bridge removal system
    
    fprintf('=== Bridge Removal Demo ===\n');
    
    % Load or create test image
    % For demo purposes, create a synthetic fingerprint-like pattern
    image = createSyntheticFingerprintPattern(256, 256);
    
    % Add some noise and spurious bridges
    image = addSpuriousBridges(image);
    
    % Run bridge removal
    result = removeBridgesWithOrientationField(image, ...
        'blockSize', 16, ...
        'angleThreshold', pi/4, ...
        'inconsistencyThreshold', 0.6, ...
        'minLength', 10);
    
    % Evaluate results
    metrics = evaluateBridgeRemoval(result.originalSkeleton, result.cleanedSkeleton);
    
    % Visualize results
    visualizeResults(image, result);
    
    fprintf('Demo completed!\n');
end

function image = createSyntheticFingerprintPattern(height, width)
    % Create a synthetic fingerprint-like pattern for testing
    
    [X, Y] = meshgrid(1:width, 1:height);
    centerX = width / 2;
    centerY = height / 2;
    
    % Create concentric ridges with some curvature
    radius = sqrt((X - centerX).^2 + (Y - centerY).^2);
    angle = atan2(Y - centerY, X - centerX);
    
    % Sinusoidal pattern with varying frequency
    frequency = 0.3;
    phase = radius * frequency + 0.1 * angle;
    pattern = sin(phase);
    
    % Convert to binary and add some noise
    image = (pattern > 0) * 255;
    image = uint8(image);
    
    % Add Gaussian noise
    noise = randn(height, width) * 10;
    image = uint8(max(0, min(255, double(image) + noise)));
end

function noisyImage = addSpuriousBridges(image)
    % Add artificial spurious bridges to test the algorithm
    
    noisyImage = image;
    
    % Add some random line segments as spurious bridges
    [height, width] = size(image);
    
    for i = 1:5  % Add 5 spurious bridges
        % Random start and end points
        y1 = randi([20, height-20]);
        x1 = randi([20, width-20]);
        y2 = y1 + randi([-15, 15]);
        x2 = x1 + randi([-15, 15]);
        
        % Draw line
        [lineY, lineX] = bresenham(y1, x1, y2, x2);
        
        % Make sure indices are within bounds
        validIdx = lineY >= 1 & lineY <= height & lineX >= 1 & lineX <= width;
        lineY = lineY(validIdx);
        lineX = lineX(validIdx);
        
        % Add the spurious bridge
        for j = 1:length(lineY)
            noisyImage(lineY(j), lineX(j)) = 255;
        end
    end
end

function [y, x] = bresenham(y1, x1, y2, x2)
    % Simple Bresenham line algorithm
    
    dx = abs(x2 - x1);
    dy = abs(y2 - y1);
    
    if x1 < x2
        sx = 1;
    else
        sx = -1;
    end
    
    if y1 < y2
        sy = 1;
    else
        sy = -1;
    end
    
    err = dx - dy;
    
    x = [];
    y = [];
    
    while true
        x(end+1) = x1;
        y(end+1) = y1;
        
        if x1 == x2 && y1 == y2
            break;
        end
        
        e2 = 2 * err;
        
        if e2 > -dy
            err = err - dy;
            x1 = x1 + sx;
        end
        
        if e2 < dx
            err = err + dx;
            y1 = y1 + sy;
        end
    end
end

%% ===================================================================
%  ADVANCED FEATURES
%% ===================================================================

function result = adaptiveBridgeRemoval(image, qualityMap)
    % Adaptive bridge removal with quality-based thresholding
    
    if nargin < 2
        % Compute quality map based on local gradient magnitude
        [Gx, Gy] = gradient(double(image));
        qualityMap = sqrt(Gx.^2 + Gy.^2);
        qualityMap = (qualityMap - min(qualityMap(:))) / (max(qualityMap(:)) - min(qualityMap(:)));
    end
    
    % Adaptive thresholds based on local quality
    highQualityMask = qualityMap > 0.7;
    mediumQualityMask = qualityMap > 0.4 & qualityMap <= 0.7;
    lowQualityMask = qualityMap <= 0.4;
    
    % Different parameters for different quality regions
    params_high = struct('angleThreshold', pi/6, 'inconsistencyThreshold', 0.5);
    params_medium = struct('angleThreshold', pi/4, 'inconsistencyThreshold', 0.6);
    params_low = struct('angleThreshold', pi/3, 'inconsistencyThreshold', 0.7);
    
    % Apply bridge removal with adaptive parameters
    % (Implementation would need to be extended to handle spatially varying parameters)
    
    result = removeBridgesWithOrientationField(image, ...
        'angleThreshold', pi/4, 'inconsistencyThreshold', 0.6);
    
    result.qualityMap = qualityMap;
end

function orientationField = multiScaleOrientationField(image, scales)
    % Compute orientation field at multiple scales and combine
    
    if nargin < 2
        scales = [8, 16, 32];  % Default scales
    end
    
    orientationFields = cell(length(scales), 1);
    weights = zeros(length(scales), 1);
    
    % Compute OF at each scale
    for i = 1:length(scales)
        orientationFields{i} = computeOrientationField(image, scales(i));
        
        % Weight based on local coherence (simplified)
        [Gx, Gy] = gradient(double(image));
        coherence = sqrt(Gx.^2 + Gy.^2);
        weights(i) = mean(coherence(:));
    end
    
    % Normalize weights
    weights = weights / sum(weights);
    
    % Combine using circular averaging
    complexSum = zeros(size(image));
    for i = 1:length(scales)
        complexSum = complexSum + weights(i) * exp(2i * orientationFields{i});
    end
    
    orientationField = 0.5 * angle(complexSum);
end