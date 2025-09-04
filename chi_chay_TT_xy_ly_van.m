clc; clear; close all;
%% Load ảnh đã bỏ junction
load('BW.mat')
% xoá vùng nhỏ lẻ
BW = removeSmallComponents(BW, 6);  % xoá vùng liên thông < 10 pixel

% Kernel để đếm số hàng xóm (8-neighbors)
endPoints = findEndpoints(BW);

figure; imshow(BW); title('Skeleton gốc');

[row, col] = find(endPoints);
hold on; plot(col, row, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
%
% Timf endPoint
fprintf('--> Bước 2b: Ước lượng vector hướng theo đoạn liên thông\n');
vectors = fitEndpointVectors(BW, endPoints, 30);

imshow(BW); hold on;
for i = 1:size(vectors,1)
    cx = vectors(i,1);
    cy = vectors(i,2);
    vx = vectors(i,3);
    vy = vectors(i,4);
    quiver(cx, cy, 10*vx, 10*vy, 'r', 'LineWidth',2, 'MaxHeadSize',2);
end
title('Vector hướng tại các endpoint');
CC = bwconncomp(BW, 8);

%% 4. ƯỚC LƯỢNG VECTOR HƯỚNG TẠI ENDPOINTS (theo component 30 pixel)
fprintf('--> Bước 2c: Nối các endpoint theo vector hướng\n');
n = 6;
for count =1:n
    BW = bwmorph(BW, 'spur', 1);  % loại bỏ các nhánh nhỏ lẻ

    if count == 1 %nối vân dài + góc lệch nhỏ + khoảng cách nhỏ
        fprintf('--> lan chay dau tien\n');
        minCompSize = 12;   % chỉ nối nếu component đủ dài
        maxDist     = 6;   % khoảng cách tối đa giữa 2 endpoint
        vecAlignThr = cosd(15);  % = 0.866 ~ hướng lệch <= 30°
    end
    if count ==2 % Nối vân dài + góc lệch lớn + khoảng cách lớn
        fprintf('--> lan chay lan 2\n');

        minCompSize = 12;   % chỉ nối nếu component đủ dài
        maxDist     = 12;   % khoảng cách tối đa giữa 2 endpoint
        vecAlignThr = cosd(15);  % = 0.866 ~ hướng lệch <= 30°
    end
    if count == 3 % Nối vân dài + góc lệch nhỏ + khoảng cách lớn hơn
        fprintf('--> lan chay lan 3\n');

        minCompSize = 12;   % chỉ nối nếu component đủ dài
        maxDist     = 25;   % khoảng cách tối đa giữa 2 endpoint
        vecAlignThr = cosd(15);  % = 0.866 ~ hướng lệch <= 30°
    end
    if count == 4 % Nối vân ngắn + góc lệch lớn
        fprintf('--> lan chay lan 4\n');

        minCompSize = 5;   % chỉ nối nếu component đủ dài
        maxDist     = 20;   % khoảng cách tối đa giữa 2 endpoint
        vecAlignThr = cosd(15);  % = 0.866 ~ hướng lệch <= 30°
    end
    if count == 5 % Nối vân ngắn + góc lệch lớn
        fprintf('--> lan chay lan 5\n');

        minCompSize = 20;   % chỉ nối nếu component đủ dài
        maxDist     = 50;   % khoảng cách tối đa giữa 2 endpoint
        vecAlignThr = cosd(15);  % = 0.866 ~ hướng lệch <= 30°
    end
    if count == 6 % Nối vân ngắn + góc lệch lớn
        fprintf('--> lan chay lan 6\n');

        minCompSize = 20;   % chỉ nối nếu component đủ dài
        maxDist     = 50;   % khoảng cách tối đa giữa 2 endpoint
        vecAlignThr = cosd(30);  % = 0.866 ~ hướng lệch <= 30°
    end
    if count == 7 % Nối vân ngắn + góc lệch lớn
        fprintf('--> lan chay lan 7\n');

        minCompSize = 20;   % chỉ nối nếu component đủ dài
        maxDist     = 50;   % khoảng cách tối đa giữa 2 endpoint
        vecAlignThr = cosd(30);  % = 0.866 ~ hướng lệch <= 30°
    end
Nfit = 30;          % số pixel để fit PCA

[BW_final, allLines] = connectEndpoints_iterative(BW, Nfit, minCompSize, maxDist, vecAlignThr);

%     figure; imshow(BW); hold on;
%     for k = 1:numel(linesConnected)
%         lineXY = linesConnected{k};
%         plot(lineXY(:,1), lineXY(:,2), 'g-', 'LineWidth', 2);
%     end
%     title('Skeleton sau khi nối endpoint (màu xanh)');

end

%%

figure;
for count = 1:n
    subplot(2,3,count);
    imshow(BW,[]); hold on;
    title(sprintf('Sau lần nối %d', count));
end


% Kết thúc

%% Nối tiếp bằng cách kéo dài vân
drawline_count = 10;
% for count = 1:drawline_count
%     endPoints = findEndpoints(BW);
%     CC = bwconncomp(BW,8);
%     vectors = fitEndpointVectors(BW, endPoints, 30);
% 
%     % 4. Nối bằng hàm vừa viết
%     [BW, linesDrawn] = connectEndpointsProbe(BW, vectors, CC, 20, 30);
% 
%     % 5. Hiển thị
%     figure; imshow(BW,[]); hold on;
%     plot(vectors(:,1), vectors(:,2), 'ro'); % endpoint
%     for k = 1:numel(linesDrawn)
%         pts = linesDrawn{k};
%         plot(pts(:,1), pts(:,2), 'g-', 'LineWidth',2);
%     end
%     title(sprintf('Kết quả nối bằng DrawLine Probe lần thứ %d', count));
% end
%%





% Kết thúc
%% --- FUNCTION MZS ---


%% ------------------

function [BW_out, linePixels] = drawLine(BW, x1, y1, x2, y2)
    % Bresenham line
    [cx, cy] = bresenham(x1, y1, x2, y2);
    linePixels = [cx(:), cy(:)];
    
    BW_out = BW;
    idx = sub2ind(size(BW), cy, cx);
    BW_out(idx) = 1;
end

function [rr,cc] = bresenham(y1, x1, y2, x2)
    % Thuật toán Bresenham (integer line rasterization)
    x1=round(x1); x2=round(x2);
    y1=round(y1); y2=round(y2);
    dx = abs(x2-x1);
    dy = abs(y2-y1);
    steep = abs(dy)>abs(dx);
    if steep
        [x1,y1] = deal(y1,x1);
        [x2,y2] = deal(y2,x2);
        [dx,dy] = deal(dy,dx);
    end
    if x1 > x2
        [x1,x2] = deal(x2,x1);
        [y1,y2] = deal(y2,y1);
    end
    derr = 2*dy; err = 0;
    y = y1;
    if y2>y1, ystep=1; else, ystep=-1; end
    rr=[]; cc=[];
    for x=x1:x2
        if steep
            rr=[rr; x]; cc=[cc; y];
        else
            rr=[rr; y]; cc=[cc; x];
        end
        err = err + derr;
        if err > dx
            y = y + ystep;
            err = err - 2*dx;
        end
    end
end
function endPoints = findEndpoints(BW)
% findEndpoints - Tìm các điểm endpoint trên skeleton
%
% Cú pháp:
%   endPoints = findEndpoints(BW)
%
% Input:
%   BW - ảnh nhị phân (skeleton)
%
% Output:
%   endPoints - ảnh nhị phân, 1 tại vị trí endpoint
%
% Đặc điểm:
%   - Endpoint = pixel có đúng 1 hàng xóm trong 8 hướng
%   - Loại bỏ endpoint nằm sát biên ảnh (3 pixel)

    % Kernel để đếm số hàng xóm 8 hướng (bỏ tâm)
    kernel = ones(3,3);
    kernel(2,2) = 0;

    % Đếm số hàng xóm
    neighborCount = conv2(double(BW), kernel, 'same');

    % Endpoint = pixel skeleton có đúng 1 hàng xóm
    endPoints = (BW == 1) & (neighborCount == 1);

    % Loại bỏ endpoint ở biên (3 pixel)
    endPoints(1:3,:)       = 0;
    endPoints(end-2:end,:) = 0;
    endPoints(:,1:3)       = 0;
    endPoints(:,end-2:end) = 0;
end

%%
function vectors = fitEndpointVectors(BW, endPoints, Nfit)
% fitEndpointVectors - Tính vector hướng tại endpoint của skeleton
%
% Cú pháp:
%   vectors = fitEndpointVectors(BW, endPoints, Nfit)
%
% Input:
%   BW        - ảnh nhị phân skeleton
%   endPoints - ảnh nhị phân endpoint (1 tại endpoint)
%   Nfit      - số pixel dùng để fit PCA (ví dụ: 30)
%
% Output:
%   vectors - ma trận [N x 4], mỗi hàng:
%             [cx cy vx vy]
%             (cx, cy) = tọa độ endpoint
%             (vx, vy) = vector đơn vị hướng ra ngoài
    [y_idx, x_idx] = find(endPoints);  % tọa độ endpoints
    CC = bwconncomp(BW, 8);           % tìm các component
    vectors = [];
    for k = 1:length(x_idx)
        cx = x_idx(k); 
        cy = y_idx(k);
        % Kiểm tra endpoint thuộc component nào
        comp_id = 0;
        for c = 1:CC.NumObjects
            if ismember(sub2ind(size(BW), cy, cx), CC.PixelIdxList{c})
                comp_id = c; 
                break;
            end
        end
        if comp_id == 0, continue; end  % endpoint không thuộc component nào
        % Lấy tọa độ tất cả pixel trong component
        [yy, xx] = ind2sub(size(BW), CC.PixelIdxList{comp_id});
        % Tính khoảng cách từ endpoint
        dist2 = (xx - cx).^2 + (yy - cy).^2;
        [~, idx] = sort(dist2);
        idxN = idx(1:min(Nfit, numel(idx)));
        X = xx(idxN); 
        Y = yy(idxN);
        if numel(X) > 1
            % --- Fit hướng bằng PCA ---
            Xc = X - mean(X);
            Yc = Y - mean(Y);
            D = [Xc(:) Yc(:)];
            [~,~,V] = svd(D,'econ');
            v = V(:,1);  % vector chính (cột đầu tiên)
            v = v / norm(v);
            % --- Xác định hướng "ra ngoài" ---
            centroid = [mean(X); mean(Y)];
            c = centroid - [cx; cy];  % vector từ endpoint vào trong component
            if dot(v, c) > 0
                v = -v; % đảo dấu để hướng ra ngoài
            end
        else
            v = [0;0];
        end
        vectors = [vectors; cx cy v(1) v(2)];
    end
end

function [BW_new, linesConnected, new_endPoints] = connectEndpoints_fast(BW, vectors, CC, minCompSize, maxDist, vecAlignThr)
% BW             : skeleton binary
% vectors        : [cx cy vx vy] từ hàm computeEndpointVectors
% CC             : bwconncomp(BW,8)
% minCompSize    : kích thước tối thiểu của vân
% maxDist        : khoảng cách tối đa cho phép nối
% vecAlignThr    : ngưỡng cos(angle) hướng (ví dụ 0.7 ~ >45°)
%
% BW_new         : skeleton sau khi nối
% linesConnected : cell lưu danh sách pixel của các đoạn nối
% new_endPoints  : ma trận endpoint còn lại (sau khi xoá các điểm đã nối)

    BW_new = BW;              % copy để cập nhật nối
    linesConnected = {};      % cell lưu danh sách các đoạn đã nối
    used = false(size(vectors,1),1); % đánh dấu endpoint nào đã được nối

    for i = 1:size(vectors,1)-1
        if used(i), continue; end

        cx1 = vectors(i,1); cy1 = vectors(i,2);
        v1 = [vectors(i,3), vectors(i,4)];

        % kiểm tra component của endpoint i
        comp_id1 = findComponent(CC, [cy1,cx1]);
        if comp_id1==0 || numel(CC.PixelIdxList{comp_id1}) < minCompSize
            continue;
        end

        for j = i+1:size(vectors,1)
            if used(j), continue; end

            cx2 = vectors(j,1); cy2 = vectors(j,2);
            v2 = [vectors(j,3), vectors(j,4)];

            % kiểm tra component j
            comp_id2 = findComponent(CC, [cy2,cx2]);
            if comp_id2==0 || numel(CC.PixelIdxList{comp_id2}) < minCompSize
                continue;
            end

            % --- khoảng cách Euclidean giữa 2 endpoint ---
            d = hypot(cx1-cx2, cy1-cy2);
            if d > maxDist, continue; end

            % --- kiểm tra hướng vector (cùng hướng nối) ---
            dir12 = [cx2-cx1, cy2-cy1];
            dir12 = dir12 / (norm(dir12)+eps);
            cond1 = dot(v1, dir12) > vecAlignThr;   % v1 hướng về P2
            cond2 = dot(v2, -dir12) > vecAlignThr;  % v2 hướng về P1
            if ~(cond1 && cond2), continue; end

            % --- kiểm tra thêm khoảng cách vuông góc ---
            a = -v2(2); b = v2(1);
            c = v2(2)*cx2 - v2(1)*cy2;
            d_perp = abs(a*cx1 + b*cy1 + c) / sqrt(a^2 + b^2);
            if d_perp > 5, continue; end

            % --- nối 2 endpoint ---
            [BW_new, linePixels] = drawLine(BW_new, cx1, cy1, cx2, cy2);
            linesConnected{end+1} = linePixels; %#ok<AGROW>

            % Đánh dấu đã nối
            used(i) = true;
            used(j) = true;
            break; % endpoint i đã nối -> thoát vòng j
        end
    end

    % cập nhật danh sách endpoint còn lại
    new_endPoints = vectors(~used, 1:2);

end



%% Hàm phụ: tìm component chứa 1 pixel
function comp_id = findComponent(CC, p)
% p = [row, col]
comp_id = 0;
idx = sub2ind(CC.ImageSize, p(1), p(2));
for c = 1:CC.NumObjects
    if ismember(idx, CC.PixelIdxList{c})
        comp_id = c;
        return;
    end
end
end



function [BW_new, linesDrawn] = connectEndpointsProbe(BW, vectors, CC, minLen, probeLen)
%CONNECTENDPOINTSPROBE Nối các endpoint skeleton bằng đoạn thăm dò
% 
% [BW_new, linesDrawn] = connectEndpointsProbe(BW, vectors, CC, minLen, probeLen)
%
% INPUT:
%   BW        - ảnh nhị phân skeleton (logical)
%   vectors   - ma trận [x y vx vy] các endpoint và vector hướng
%   CC        - cấu trúc từ bwconncomp(BW,8)
%   minLen    - ngưỡng chiều dài vân tối thiểu được xét
%   probeLen  - độ dài đoạn line thăm dò (pixel)
%
% OUTPUT:
%   BW_new     - ảnh sau khi nối
%   linesDrawn - cell array chứa các đoạn line đã vẽ, 
%                mỗi phần tử = [x1 y1; x2 y2]

    BW_new = BW;
    linesDrawn = {}; 

    for i = 1:size(vectors,1)
        cx = vectors(i,1); 
        cy = vectors(i,2);
        v  = [vectors(i,3), vectors(i,4)];

        % tìm component chứa endpoint
        comp_id = 0;
        for c = 1:CC.NumObjects
            if ismember(sub2ind(size(BW), cy, cx), CC.PixelIdxList{c})
                comp_id = c; 
                break;
            end
        end
        if comp_id==0 || numel(CC.PixelIdxList{comp_id}) < minLen
            continue; % bỏ qua vân ngắn
        end

        % tạo điểm probe theo hướng vector
        x2 = round(cx + probeLen * v(1));
        y2 = round(cy + probeLen * v(2));

        % ép trong ảnh
        x2 = max(1, min(size(BW,2), x2));
        y2 = max(1, min(size(BW,1), y2));

        % vẽ đoạn probe
        probeLine = drawLine(false(size(BW)), cx, cy, x2, y2);

        % kiểm tra giao vân khác
        overlap = probeLine & BW;
        overlap(cy, cx) = 0; 
        if any(overlap(:))
            % lấy điểm giao đầu tiên
            [ry, cx_] = find(overlap, 1, 'first');
            BW_new = drawLine(BW_new, cx, cy, cx_, ry);

            % lưu line
            linesDrawn{end+1} = [cx cy; cx_ ry];
        end
    end
end

function vectors = fitEndpointCurves(BW, endPoints, Nfit)
% Tính vector tiếp tuyến cong tại endpoint của skeleton
% dùng spline fitting

    [y_idx, x_idx] = find(endPoints);  
    CC = bwconncomp(BW, 8);           
    vectors = [];

    for k = 1:length(x_idx)
        cx = x_idx(k); 
        cy = y_idx(k);

        % Tìm component chứa endpoint
        comp_id = 0;
        for c = 1:CC.NumObjects
            if ismember(sub2ind(size(BW), cy, cx), CC.PixelIdxList{c})
                comp_id = c; 
                break;
            end
        end
        if comp_id == 0, continue; end

        % Tìm đường đi (tracing) từ endpoint ra Nfit pixel
        path = traceSkeleton(BW, [cy,cx], Nfit);

        if size(path,1) < 3
            v = [0 0];
        else
            % Fit spline qua các điểm
            t = 1:size(path,1);
            ppX = spline(t, path(:,2)); % x theo t
            ppY = spline(t, path(:,1)); % y theo t

            % Lấy đạo hàm tại điểm đầu (t=1)
            dx = ppval(fnder(ppX,1),1);
            dy = ppval(fnder(ppY,1),1);

            v = [dx dy];
            v = v / norm(v+eps);
        end

        vectors = [vectors; cx cy v(1) v(2)];
    end
end

function path = traceSkeleton(BW, startPt, Nmax)
% traceSkeleton - lần theo skeleton từ một endpoint
%
% Input:
%   BW      : ảnh nhị phân skeleton
%   startPt : [row, col] = điểm bắt đầu (endpoint)
%   Nmax    : số pixel tối đa cần lấy
%
% Output:
%   path    : [N x 2] = [row col] của đường đi

    % Khởi tạo
    path = startPt;
    visited = false(size(BW));
    visited(startPt(1), startPt(2)) = true;

    % Điểm hiện tại
    cur = startPt;

    for k = 2:Nmax
        % Lấy láng giềng 8 hướng
        [rr, cc] = ndgrid(cur(1)-1:cur(1)+1, cur(2)-1:cur(2)+1);
        rr = rr(:); cc = cc(:);

        % Giữ các điểm nằm trong ảnh
        valid = rr>=1 & rr<=size(BW,1) & cc>=1 & cc<=size(BW,2);
        rr = rr(valid); cc = cc(valid);

        % Lấy các neighbor thuộc skeleton chưa đi qua
        neigh = [rr cc];
        idx = sub2ind(size(BW), rr, cc);
        mask = BW(idx) & ~visited(idx);

        if ~any(mask)
            break; % hết đường để đi
        end

        % Nếu có nhiều nhánh -> chọn một (ở skeleton thật thì chỉ có 1 hoặc 2)
        rr = rr(mask);
        cc = cc(mask);

        % Chọn pixel gần nhất (thông thường chỉ 1)
        d2 = (rr-cur(1)).^2 + (cc-cur(2)).^2;
        [~,imin] = min(d2);

        nxt = [rr(imin), cc(imin)];

        % Thêm vào path
        path = [path; nxt]; %#ok<AGROW>
        visited(nxt(1),nxt(2)) = true;
        cur = nxt;
    end
end
function d = pointToVectorDistance(M, P, v)
% POINTTOVECTORDISTANCE Khoảng cách từ điểm M đến đường thẳng
% qua P và có vector hướng v
%
% INPUT:
%   M = [x0, y0] : endpoint cần đo
%   P = [x1, y1] : endpoint sinh ra vector
%   v = [u, v]   : vector hướng tại P
%
% OUTPUT:
%   d : khoảng cách vuông góc từ M đến đường thẳng

    x0 = M(1); y0 = M(2);
    x1 = P(1); y1 = P(2);
    u = v(1);  w = v(2);

    % hệ số phương trình đường thẳng
    a = -w;
    b = u;
    c = w*x1 - u*y1;

    % khoảng cách từ M đến đường thẳng
    d = abs(a*x0 + b*y0 + c) / sqrt(a^2 + b^2);
end
function BW_clean = removeSmallComponents(BW, minSize)
    % Xoá các vùng liên thông có kích thước < minSize pixel
    %
    % Input:
    %   BW      - ảnh nhị phân (0/1)
    %   minSize - ngưỡng số pixel nhỏ nhất giữ lại
    %
    % Output:
    %   BW_clean - ảnh nhị phân sau khi lọc
    
    % Tìm vùng liên thông (8-neighbors)
    CC = bwconncomp(BW, 8);
    % Đếm số pixel trong từng vùng
    numPixels = cellfun(@numel, CC.PixelIdxList);
    
    % Giữ lại vùng đủ lớn
    BW_clean = BW;
    for i = 1:CC.NumObjects
        if numPixels(i) < minSize
            BW_clean(CC.PixelIdxList{i}) = 0; % xoá vùng nhỏ
        end
    end
    BW_clean = bwmorph(BW_clean, 'spur', 1);  % loại bỏ các nhánh nhỏ lẻ

end



function [BW_final, allLines] = connectEndpoints_iterative(BW, Nfit, minCompSize, maxDist, vecAlignThr)

    BW_final = BW;
    allLines = {};   % lưu toàn bộ đoạn nối
    iter = 1;

    while true
        % --- tìm endpoint ---
        endPoints = bwmorph(BW_final, 'endpoints');
        if nnz(endPoints) < 2
            break; % không còn endpoint để nối
        end

        % --- tính vector hướng cho từng endpoint ---
        vectors = fitEndpointVectors(BW_final, endPoints, Nfit);

        % --- phân tích component ---
        CC = bwconncomp(BW_final, 8);

        % --- thử nối endpoint ---
        [BW_new, linesConnected, new_endPoints] = connectEndpoints_fast( ...
            BW_final, vectors, CC, minCompSize, maxDist, vecAlignThr);

        % Nếu không có nối nào xảy ra -> dừng
        if isempty(linesConnected)
            break;
        end

        % Cập nhật skeleton & lưu các line đã nối
        BW_final = BW_new;
        allLines = [allLines; linesConnected(:)]; %#ok<AGROW>

        fprintf('Iteration %d: nối được %d cặp endpoint\n', iter, numel(linesConnected));
        iter = iter + 1;
    end

end
