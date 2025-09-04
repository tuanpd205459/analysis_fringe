%% Demo script cho skeleton curve interpolation

clc, clear,close all;
%% Tạo skeleton mẫu
% Tạo hình ảnh với đường cong hình S
[X, Y] = meshgrid(1:100, 1:100);
skeleton = zeros(100, 100);

% Vẽ đường cong spiral
t = linspace(0, 4*pi, 80);
x = round(50 + 15*cos(t/2).*cos(t));
y = round(50 + 15*cos(t/2).*sin(t));

% Loại bỏ điểm nằm ngoài ảnh
valid_idx = x >= 1 & x <= 100 & y >= 1 & y <= 100;
x = x(valid_idx);
y = y(valid_idx);

% Tạo skeleton
for i = 1:length(x)
    skeleton(y(i), x(i)) = 1;
end

%% Method 1: Sử dụng function đơn giản
fprintf('=== Testing simple function ===\n');

methods = {'spline', 'pchip', 'linear', 'bezier'};
figure('Position', [100, 100, 1200, 300]);

for i = 1:length(methods)
    method = methods{i};
    fprintf('Testing %s method...\n', method);
    
    try
        [curve_points, ordered_skeleton] = skeleton_to_curve(skeleton, ...
            'Method', method, 'NumPoints', 100, 'Visualize', false);
        
        subplot(1, 4, i);
        imshow(skeleton); hold on;
        plot(curve_points(:, 1), curve_points(:, 2), 'r-', 'LineWidth', 2);
        plot(curve_points(1, 1), curve_points(1, 2), 'go', 'MarkerSize', 6);
        plot(curve_points(end, 1), curve_points(end, 2), 'bo', 'MarkerSize', 6);
        title(sprintf('%s Method', upper(method)));
        
        fprintf('  - Generated %d curve points\n', size(curve_points, 1));
        
    catch ME
        fprintf('  - Error: %s\n', ME.message);
        subplot(1, 4, i);
        text(0.5, 0.5, 'Error', 'HorizontalAlignment', 'center');
        title(sprintf('%s - Error', upper(method)));
    end
end

%% Method 2: Sử dụng class nâng cao
fprintf('\n=== Testing advanced class ===\n');

interpolator = SkeletonCurveInterpolator();

% Test các phương pháp khác nhau
advanced_methods = {'spline', 'pchip', 'makima', 'bezier', 'catmull_rom'};

figure('Position', [100, 500, 1500, 300]);

for i = 1:length(advanced_methods)
    method = advanced_methods{i};
    fprintf('Testing %s method with class...\n', method);
    
    try
        curve_points = interpolator.interpolate(skeleton, ...
            'Method', method, 'NumPoints', 150);
        
        subplot(1, 5, i);
        imshow(skeleton); hold on;
        plot(curve_points(:, 1), curve_points(:, 2), 'r-', 'LineWidth', 2);
        plot(curve_points(1, 1), curve_points(1, 2), 'go', 'MarkerSize', 6);
        plot(curve_points(end, 1), curve_points(end, 2), 'bo', 'MarkerSize', 6);
        title(sprintf('%s', upper(method)));
        
        fprintf('  - Generated %d curve points\n', size(curve_points, 1));
        
    catch ME
        fprintf('  - Error: %s\n', ME.message);
    end
end

%% Detailed visualization với một method
fprintf('\n=== Detailed visualization ===\n');

% Sử dụng spline method với visualization chi tiết
interpolator_detail = SkeletonCurveInterpolator();
curve_detailed = interpolator_detail.interpolate(skeleton, ...
    'Method', 'spline', 'NumPoints', 200);

interpolator_detail.visualize('ShowSteps', true);

%% So sánh độ mượt
fprintf('\n=== Smoothness comparison ===\n');

figure('Position', [100, 900, 1200, 400]);

smoothing_values = [0, 0.1, 0.5, 1.0];

for i = 1:length(smoothing_values)
    smooth_val = smoothing_values(i);
    
    [curve_smooth, ~] = skeleton_to_curve(skeleton, ...
        'Method', 'spline', 'NumPoints', 100, 'Smoothing', smooth_val);
    
    subplot(1, 4, i);
    imshow(skeleton); hold on;
    plot(curve_smooth(:, 1), curve_smooth(:, 2), 'r-', 'LineWidth', 2);
    title(sprintf('Smoothing = %.1f', smooth_val));
end

%% Performance test
fprintf('\n=== Performance test ===\n');

num_trials = 10;
methods_perf = {'linear', 'spline', 'pchip', 'bezier'};

fprintf('Method\t\tAvg Time (s)\n');
fprintf('------------------------\n');

for method = methods_perf
    times = zeros(num_trials, 1);
    
    for trial = 1:num_trials
        tic;
        skeleton_to_curve(skeleton, 'Method', method{1}, 'NumPoints', 100);
        times(trial) = toc;
    end
    
    avg_time = mean(times);
    fprintf('%-12s\t%.4f\n', method{1}, avg_time);
end

fprintf('\nDemo completed successfully!\n');

%%
function skeleton = create_skeleton(binary_image, method)
% CREATE_SKELETON Tạo skeleton từ binary image
%
% Parameters:
%   binary_image - Binary image (logical hoặc 0/1)
%   method       - 'matlab' (bwmorph) hoặc 'custom' (tự implement)

if nargin < 2
    method = 'matlab';
end

switch lower(method)
    case 'matlab'
        % Sử dụng bwmorph của MATLAB
        skeleton = bwmorph(binary_image, 'skel', Inf);
        
    case 'custom'
        % Zhang-Suen skeletonization algorithm
        skeleton = zhang_suen_skeleton(binary_image);
        
    otherwise
        error('Unknown skeletonization method: %s', method);
end

end

function skeleton = zhang_suen_skeleton(binary_image)
% Zhang-Suen thinning algorithm

skeleton = binary_image > 0;
changing = true;

while changing
    changing = false;
    
    % Sub-iteration 1
    marked = false(size(skeleton));
    for i = 2:size(skeleton,1)-1
        for j = 2:size(skeleton,2)-1
            if skeleton(i,j) && zhang_suen_condition(skeleton, i, j, 1)
                marked(i,j) = true;
                changing = true;
            end
        end
    end
    skeleton(marked) = false;
    
    % Sub-iteration 2
    marked = false(size(skeleton));
    for i = 2:size(skeleton,1)-1
        for j = 2:size(skeleton,2)-1
            if skeleton(i,j) && zhang_suen_condition(skeleton, i, j, 2)
                marked(i,j) = true;
                changing = true;
            end
        end
    end
    skeleton(marked) = false;
end

end

function result = zhang_suen_condition(image, i, j, iteration)
% Zhang-Suen conditions

% 8-neighborhood
P = [image(i-1,j-1), image(i-1,j), image(i-1,j+1), ...
     image(i,j+1), image(i+1,j+1), image(i+1,j), ...
     image(i+1,j-1), image(i,j-1)];

% Number of black neighbors
N = sum(P);

% Number of 0-1 transitions
S = 0;
for k = 1:8
    if P(k) == 0 && P(mod(k,8)+1) == 1
        S = S + 1;
    end
end

% Conditions
cond1 = N >= 2 && N <= 6;
cond2 = S == 1;

if iteration == 1
    cond3 = P(2) * P(4) * P(6) == 0;
    cond4 = P(4) * P(6) * P(8) == 0;
else
    cond3 = P(2) * P(4) * P(8) == 0;
    cond4 = P(2) * P(6) * P(8) == 0;
end

result = cond1 && cond2 && cond3 && cond4;

end

function [curve_points, ordered_skeleton] = skeleton_to_curve(skeleton_image, varargin)
% SKELETON_TO_CURVE Ngoại suy đường cong từ skeleton image
%
% Syntax:
%   [curve_points, ordered_skeleton] = skeleton_to_curve(skeleton_image)
%   [curve_points, ordered_skeleton] = skeleton_to_curve(skeleton_image, 'Name', Value)
%
% Parameters:
%   skeleton_image - Binary image chứa skeleton (logical hoặc 0/1)
%   
% Name-Value pairs:
%   'Method'     - Phương pháp ngoại suy ('spline', 'pchip', 'linear', 'bezier')
%   'NumPoints'  - Số điểm trên đường cong (default: 100)
%   'Smoothing'  - Hệ số làm mịn cho spline (default: 0)
%   'Visualize'  - Hiển thị kết quả (default: false)
%
% Returns:
%   curve_points     - Ma trận [N x 2] chứa tọa độ (x,y) của đường cong
%   ordered_skeleton - Ma trận [M x 2] chứa tọa độ skeleton đã sắp xếp

% Parse input arguments
p = inputParser;
addRequired(p, 'skeleton_image', @(x) islogical(x) || isnumeric(x));
addParameter(p, 'Method', 'spline', @(x) ismember(x, {'spline', 'pchip', 'linear', 'bezier'}));
addParameter(p, 'NumPoints', 100, @(x) isnumeric(x) && x > 0);
addParameter(p, 'Smoothing', 0, @(x) isnumeric(x) && x >= 0);
addParameter(p, 'Visualize', false, @islogical);
parse(p, skeleton_image, varargin{:});

method = p.Results.Method;
num_points = p.Results.NumPoints;
smoothing = p.Results.Smoothing;
visualize = p.Results.Visualize;

% Tiền xử lý skeleton
skeleton_binary = skeleton_image > 0;

% Tìm tọa độ các điểm skeleton
[row, col] = find(skeleton_binary);
if length(row) < 2
    error('Skeleton phải có ít nhất 2 điểm');
end

skeleton_points = [col, row]; % [x, y]

% Sắp xếp các điểm theo thứ tự liên kết
ordered_skeleton = order_skeleton_points(skeleton_points);

% Thực hiện ngoại suy theo phương pháp được chọn
switch lower(method)
    case 'spline'
        curve_points = spline_interpolation(ordered_skeleton, num_points, smoothing);
    case 'pchip'
        curve_points = pchip_interpolation(ordered_skeleton, num_points);
    case 'linear'
        curve_points = linear_interpolation(ordered_skeleton, num_points);
    case 'bezier'
        curve_points = bezier_interpolation(ordered_skeleton, num_points);
    otherwise
        error('Phương pháp không được hỗ trợ: %s', method);
end

% Visualize nếu được yêu cầu
if visualize
    visualize_results(skeleton_binary, ordered_skeleton, curve_points, method);
end

end

function ordered_points = order_skeleton_points(points)
% Sắp xếp các điểm skeleton theo thứ tự liên kết

if size(points, 1) <= 2
    ordered_points = points;
    return;
end

% Tìm điểm đầu (endpoint)
start_idx = find_endpoint(points);

% Sắp xếp điểm theo nearest neighbor
ordered_points = zeros(size(points));
ordered_points(1, :) = points(start_idx, :);

used = false(size(points, 1), 1);
used(start_idx) = true;

for i = 2:size(points, 1)
    current_point = ordered_points(i-1, :);
    
    % Tìm điểm gần nhất chưa được sử dụng
    distances = sqrt(sum((points - current_point).^2, 2));
    distances(used) = inf;
    
    [~, next_idx] = min(distances);
    ordered_points(i, :) = points(next_idx, :);
    used(next_idx) = true;
end

end

function endpoint_idx = find_endpoint(points)
% Tìm điểm đầu của skeleton (điểm có ít neighbor nhất)

min_neighbors = inf;
endpoint_idx = 1;

for i = 1:size(points, 1)
    current_point = points(i, :);
    
    % Đếm số neighbor trong bán kính nhỏ
    distances = sqrt(sum((points - current_point).^2, 2));
    neighbors = sum(distances > 0 & distances <= sqrt(2));
    
    if neighbors < min_neighbors
        min_neighbors = neighbors;
        endpoint_idx = i;
    end
end

end

function curve_points = spline_interpolation(points, num_points, smoothing)
% Spline interpolation

% Tính cumulative arc length
arc_length = [0; cumsum(sqrt(sum(diff(points).^2, 2)))];

% Tạo parameter vector mới
t_new = linspace(0, arc_length(end), num_points);

% Áp dụng smoothing spline nếu cần
if smoothing > 0
    % Sử dụng csaps (Cubic Smoothing Spline) nếu có Curve Fitting Toolbox
    if exist('csaps', 'file')
        pp_x = csaps(arc_length, points(:, 1), smoothing);
        pp_y = csaps(arc_length, points(:, 2), smoothing);
        curve_x = ppval(pp_x, t_new);
        curve_y = ppval(pp_y, t_new);
    else
        % Fallback to regular spline
        curve_x = spline(arc_length, points(:, 1), t_new);
        curve_y = spline(arc_length, points(:, 2), t_new);
    end
else
    % Regular spline interpolation
    curve_x = spline(arc_length, points(:, 1), t_new);
    curve_y = spline(arc_length, points(:, 2), t_new);
end

curve_points = [curve_x', curve_y'];

end

function curve_points = pchip_interpolation(points, num_points)
% PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)

% Tính cumulative arc length
arc_length = [0; cumsum(sqrt(sum(diff(points).^2, 2)))];

% Tạo parameter vector mới
t_new = linspace(0, arc_length(end), num_points);

% PCHIP interpolation
curve_x = pchip(arc_length, points(:, 1), t_new);
curve_y = pchip(arc_length, points(:, 2), t_new);

curve_points = [curve_x', curve_y'];

end

function curve_points = linear_interpolation(points, num_points)
% Linear interpolation

% Tính cumulative arc length
arc_length = [0; cumsum(sqrt(sum(diff(points).^2, 2)))];

% Tạo parameter vector mới
t_new = linspace(0, arc_length(end), num_points);

% Linear interpolation
curve_x = interp1(arc_length, points(:, 1), t_new, 'linear');
curve_y = interp1(arc_length, points(:, 2), t_new, 'linear');

curve_points = [curve_x', curve_y'];

end

function curve_points = bezier_interpolation(points, num_points)
% Bezier curve interpolation

n = size(points, 1) - 1;
t = linspace(0, 1, num_points);

curve_points = zeros(num_points, 2);

for i = 1:num_points
    point = [0, 0];
    for j = 0:n
        % Bernstein polynomial
        coeff = nchoosek(n, j) * t(i)^j * (1-t(i))^(n-j);
        point = point + coeff * points(j+1, :);
    end
    curve_points(i, :) = point;
end

end

function visualize_results(skeleton_binary, ordered_skeleton, curve_points, method)
% Visualize kết quả

figure('Position', [100, 100, 1200, 400]);

% Subplot 1: Original skeleton
subplot(1, 3, 1);
imshow(skeleton_binary);
title('Original Skeleton');

% Subplot 2: Ordered skeleton points
subplot(1, 3, 2);
imshow(skeleton_binary); hold on;
plot(ordered_skeleton(:, 1), ordered_skeleton(:, 2), 'ro-', 'MarkerSize', 3, 'LineWidth', 1);
plot(ordered_skeleton(1, 1), ordered_skeleton(1, 2), 'go', 'MarkerSize', 8, 'LineWidth', 2);
plot(ordered_skeleton(end, 1), ordered_skeleton(end, 2), 'bo', 'MarkerSize', 8, 'LineWidth', 2);
title('Ordered Skeleton Points');
legend('Skeleton path', 'Start', 'End', 'Location', 'best');

% Subplot 3: Interpolated curve
subplot(1, 3, 3);
imshow(skeleton_binary); hold on;
plot(curve_points(:, 1), curve_points(:, 2), 'r-', 'LineWidth', 2);
plot(curve_points(1, 1), curve_points(1, 2), 'go', 'MarkerSize', 8, 'LineWidth', 2);
plot(curve_points(end, 1), curve_points(end, 2), 'bo', 'MarkerSize', 8, 'LineWidth', 2);
title(sprintf('Interpolated Curve (%s)', upper(method)));
legend('Interpolated curve', 'Start', 'End', 'Location', 'best');

end