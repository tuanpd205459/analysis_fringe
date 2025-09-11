% vòng lặp để tính % sai số
clc; clear; close all;

% --- Skeleton giả định (test) ---
%% 1. KHỞI TẠO
% fprintf('Bắt đầu quy trình mô phỏng và tái tạo...\n');
params = struct();
params = set_default_params(params);
auto_fft = 0;
% Bề mặt ground truth
% --- Tạo đối tượng pha gốc (Ground Truth) ---
% Lựa chọn: 'peaks', 'sin', 'gaussian', 'zernike', 'step',
%           'sharp_peak', 'multi_sharp_peak', 'sinc_spike',
%           'microsphere_array','test','test_ls','residual'
params.groundTruth= 'high_noise';
noise_level = 0.15;
% Bộ lọc Fourier
params.filter_type = 'circle';         % 'circle' | 'rectangle'
params.filter_radius = 20;             % (px) bán kính bộ lọc tròn
params.filter_width = 150;             % (px) chiều rộng HCN
params.filter_height = 120;            % (px) chiều cao HCN

% Ức chế DC (zero-order)
params.dc_suppression_radius = 25;     % (px) bán kính loại bỏ trung tâm

% Tham số ảnh
params.lambda = 632.8e-9;              % bước sóng ánh sáng (m)
params.pixel_size = 3.45e-6;           % kích thước điểm ảnh (m)

% Pha
params.unwrap_method = 'step';  % hoặc 'quality_guided', 'hybrid'
params.phase_smoothing = true;
params.smoothing_sigma = 2;            % độ mượt hậu xử lý (Gaussian)

% Debug/hiển thị
params.show_figures = true;
params.verbose = true;
N =512;
M = 512;

% --- Danh sách tên thuật toán ---
phase_names = {'TIE (FFT-based)', ...
               'Reliability-based', '2D-WLS', ...
               'Goldstein', 'Proposed'};
nPhase = numel(phase_names);

%% load ảnh thực
hologram = imread("8.bmp");
if size(hologram, 3) == 3
    hologram = rgb2gray(hologram);
    fprintf('Đã chuyển đổi ảnh RGB sang grayscale\n');
end

% hologram = rot90(hologram);

% --- Hiển thị Hologram ---
figure;
imshow(hologram, []);
title('Ảnh Hologram (Giao thoa) có nhiễu');
%%
%% 3. Tạo bề mặt interferogram
imwrite(hologram, 'hologram.bmp');

%% 5. Noise removal
hologram = imgaussfilt(hologram, 1);
% hologram = medfilt2(hologram, [3 3]);
% hologram = wiener2(hologram, [5 5]);
% figure;
% imshow(hologram);
% colorbar;
% title('hologram sau noise removal : ');
%% 6. Histogram equalization
% hologram = adapthisteq(hologram);
% figure;
% imshow(hologram);
% colorbar;
% title('hologram sau equaliztion histogram : ');

%% 7. ƯỚC LƯỢNG PHA BẰNG PHƯƠNG PHÁP PHÂN TÍCH VÂN

    % --- Bước 1: Nhị phân hóa ảnh bằng Otsu ---
%     fprintf('Bước 1/3: Nhị phân hóa ảnh bằng phương pháp Otsu...\n');
    thresh = graythresh(hologram);
    BW_Original = imbinarize(hologram, thresh);

%     fprintf('Ngưỡng Otsu: %.4f\n', thresh);
%     fprintf('Số pixel foreground: %d\n', sum(BW_Original(:)));

    % --- Bước 2: Skeletonize bằng Zhang-Suen ---
%     fprintf('Bước 2/3: Áp dụng thuật toán Zhang-Suen...\n');
    BW_Thinned = BW_Original;
    [rows, cols] = size(BW_Thinned);
    changing = true;
    iteration = 0;

    while changing
        iteration = iteration + 1;
        changing = false;
        BW_Del = true(rows, cols);

        % --- Step 1 của Zhang-Suen ---
        for i = 2:rows-1
            for j = 2:cols-1
                P = BW_Thinned(i-1:i+1, j-1:j+1);
                P = P(:)';
                % Sắp xếp theo thứ tự: P1(center), P2, P3, P4, P5, P6, P7, P8, P9, P2(lặp)
                P = [P(5), P(2), P(3), P(6), P(9), P(8), P(7), P(4), P(1), P(2)];

                if P(1) == 1  % Nếu pixel trung tâm là foreground
                    neighbors = sum(P(2:9));  % Số lượng neighbor foreground
                    transitions = sum(P(2:9) == 0 & P(3:10) == 1);  % Số transition 0->1

                    % Điều kiện Zhang-Suen Step 1
                    if neighbors >= 2 && neighbors <= 6 && transitions == 1 ...
                            && P(2)*P(4)*P(6) == 0 && P(4)*P(6)*P(8) == 0
                        BW_Del(i,j) = false;
                        changing = true;
                    end
                end
            end
        end
        BW_Thinned = BW_Thinned & BW_Del;

        % --- Step 2 của Zhang-Suen ---
        BW_Del = true(rows, cols);
        for i = 2:rows-1
            for j = 2:cols-1
                P = BW_Thinned(i-1:i+1, j-1:j+1);
                P = P(:)';
                P = [P(5), P(2), P(3), P(6), P(9), P(8), P(7), P(4), P(1), P(2)];

                if P(1) == 1
                    neighbors = sum(P(2:9));
                    transitions = sum(P(2:9) == 0 & P(3:10) == 1);

                    % Điều kiện Zhang-Suen Step 2
                    if neighbors >= 2 && neighbors <= 6 && transitions == 1 ...
                            && P(2)*P(4)*P(8) == 0 && P(2)*P(6)*P(8) == 0
                        BW_Del(i,j) = false;
                        changing = true;
                    end
                end
            end
        end
        BW_Thinned = BW_Thinned & BW_Del;

        % Hiển thị tiến trình mỗi 10 iterations
        if mod(iteration, 10) == 0
%             fprintf('  Iteration %d: %d pixels còn lại\n', iteration, sum(BW_Thinned(:)));
        end

        % Tránh vòng lặp vô hạn
        if iteration > 1000
            warning('Đã đạt giới hạn iteration (1000). Dừng thuật toán.');
            break;
        end
    end

% --- Trả về kết quả ---
skeleton_image = BW_Thinned;
binary_image = BW_Original;

skeleton = skeleton_image;

%%
BW = skeleton;
% fprintf('Running Modified ZS (MZS) thinning...\n');
S = MZS_thinning(BW);

% figure;
% subplot(1,2,1); imshow(BW); title('Input binary');
% subplot(1,2,2); imshow(S);  title('Skeleton (MZS)');

BW = S;

%% Xoa vung nho le
BW = removeSmallComponents(BW, 5);  % xoá vùng liên thông < 10 pixel

%% Xoá junction

[BW, junctionMap] = removeJunctions(BW);
BW = bwmorph(BW,"spur", 5);

% figure; imshow(BW); hold on;
% [row, col] = find(junctionMap);
% plot(col, row, 'go', 'MarkerSize',10,'LineWidth',1);
% title('Skeleton sau khi xoá junction');
%
% % Xóa các đoạn ngắn (bridge)
maxBridgeLen = 8;
BW = bwareaopen(BW, maxBridgeLen);
noi_bang_tay =1;
if(noi_bang_tay)
    nLoop = 4; % số vòng nối
    for count = 1:nLoop
        CC = bwconncomp(BW, 8);
        endPoints = bwmorph(BW, 'endpoints');
        endPoints(1,:) = 0;
        endPoints(end,:) = 0;
        endPoints(:,1) = 0;
        endPoints(:,end-1) = 0;
        if count == 1
            %         fprintf('--> Vòng nối %d\n', count);
            % Tham số nối thử
            minCompSize = 15;
            maxDist     = 10;    % vòng sau cho phép nối xa hơn
            vecAlignThr = cosd(10);    % ~0.866
            vectors = fitEndpointVectors(BW, endPoints, 15);
            max_perh = 5;

        end
        if  count == 2
            %         fprintf('--> Vòng nối %d ----\n', count);

            % Tham số nối thử
            minCompSize = 12;
            maxDist     = 20;    % vòng sau cho phép nối xa hơn
            vecAlignThr = cosd(15);    % ~0.866
            max_perh = 5;
            vectors = fitEndpointVectors(BW, endPoints, 15);

        end
        if  count == 3
            %         fprintf('--> Vòng nối %d ----\n', count);

            % Tham số nối thử
            minCompSize = 8;
            maxDist     = 25;    % vòng sau cho phép nối xa hơn
            vecAlignThr = cosd(15);    % ~0.866
            max_perh = 5;

            vectors = fitEndpointVectors(BW, endPoints, 15);

        end

        if  count == 4
            %         fprintf('--> Vòng nối %d ----\n', count);

            % Tham số nối thử
            minCompSize = 5;
            maxDist     = 30;    % vòng sau cho phép nối xa hơn
            vecAlignThr = cosd(20);    % ~0.866
            max_perh = 5;

            vectors = fitEndpointVectors(BW, endPoints, 15);

        end

        [BW, linesConnected] = connectEndpoints_v3(BW, vectors, CC, minCompSize, maxDist, vecAlignThr, max_perh);

    end



    app = noi_van_bang_tay_4925;
    app.SkeletonImage = BW;
    app.anhgoc = hologram;

    % Dừng ở đây cho đến khi app gọi uiresume
    uiwait(app.UIFigure);

    % Lấy dữ liệu xuất từ app
    BW = app.SkeletonImage;

else 

%% --- 4. Nối endpoint theo vòng lặp thử ---
nLoop = 8; % số vòng nối
for count = 1:nLoop
    CC = bwconncomp(BW, 8);
    endPoints = bwmorph(BW, 'endpoints');
    endPoints(1,:) = 0;
    endPoints(end,:) = 0;
    endPoints(:,1) = 0;
    endPoints(:,end-1) = 0;
    if count == 1
%         fprintf('--> Vòng nối %d\n', count);
        % Tham số nối thử
        minCompSize = 15;
        maxDist     = 10;    % vòng sau cho phép nối xa hơn
        vecAlignThr = cosd(10);    % ~0.866
        vectors = fitEndpointVectors(BW, endPoints, 15);
        max_perh = 5;

    end
    if  count == 2
%         fprintf('--> Vòng nối %d ----\n', count);

        % Tham số nối thử
        minCompSize = 12;
        maxDist     = 20;    % vòng sau cho phép nối xa hơn
        vecAlignThr = cosd(15);    % ~0.866
        max_perh = 5;
        vectors = fitEndpointVectors(BW, endPoints, 15);

    end
    if  count == 3
%         fprintf('--> Vòng nối %d ----\n', count);

        % Tham số nối thử
        minCompSize = 8;
        maxDist     = 25;    % vòng sau cho phép nối xa hơn
        vecAlignThr = cosd(15);    % ~0.866
        max_perh = 5;

        vectors = fitEndpointVectors(BW, endPoints, 15);

    end

    if  count == 4
%         fprintf('--> Vòng nối %d ----\n', count);

        % Tham số nối thử
        minCompSize = 5;
        maxDist     = 30;    % vòng sau cho phép nối xa hơn
        vecAlignThr = cosd(20);    % ~0.866
        max_perh = 5;

        vectors = fitEndpointVectors(BW, endPoints, 15);

    end
    if  count == 5
%         fprintf('--> Vòng nối %d ----\n', count);

        % Tham số nối thử
        minCompSize = 5;
        maxDist     = 35;    % vòng sau cho phép nối xa hơn
        vecAlignThr = cosd(30);    % ~0.866

        vectors = fitEndpointVectors(BW, endPoints, 10);
        max_perh = 5;

    end

    if  count == 6
%         fprintf('--> Vòng nối %d ----\n', count);

        % Tham số nối thử
        minCompSize = 5;
        maxDist     = 40;    % vòng sau cho phép nối xa hơn
        vecAlignThr = cosd(35);    % ~0.866
        max_perh = 5;

        vectors = fitEndpointVectors(BW, endPoints, 10);
    end

    if  count == 7
%         fprintf('--> Vòng nối %d ----\n', count);

        % Tham số nối thử
        minCompSize = 5;
        maxDist     = 50;    % vòng sau cho phép nối xa hơn
        vecAlignThr = cosd(50);    % ~0.866

        vectors = fitEndpointVectors(BW, endPoints, 15);
        max_perh = 10;

    end
    if  count == 8
%         fprintf('--> Vòng nối %d ----\n', count);

        % Tham số nối thử
        minCompSize = 5;
        maxDist     = 100;    % vòng sau cho phép nối xa hơn
        vecAlignThr = cosd(50);    % ~0.866

        vectors = fitEndpointVectors(BW, endPoints, 15);
        max_perh = 25;
    end

    [BW, linesConnected] = connectEndpoints_v3(BW, vectors, CC, minCompSize, maxDist, vecAlignThr, max_perh);

end
BW_NEW = BW;
% save("BW_NEW.mat","BW_NEW");

%% Nối vân ở biên
% --- Tìm endpoint ---
endPoints = bwmorph(BW,'endpoints');

vectors = fitEndpointVectors(BW, endPoints, 20);

margin = 20;
extendLength = 20;
% --- Tham số nối ---
BW = extendLineNearBorder(BW, vectors, extendLength, margin);
% 
figure; imshow(BW,[]);
hold on; plot(vectors(:,1), vectors(:,2),'ro')
title("Nối vân ở biên");


end

% save("after_estimate.mat");
% Hoàn thành

skeleton_image = BW;

%% 7. ƯỚC LƯỢNG PHA BẰNG PHƯƠNG PHÁP PHÂN TÍCH VÂN
% fprintf('--> Bước 3: Ước lượng pha thô bằng phân tích vân...\n');
% Làm mảnh và gán bậc vân

[~, labels, img] = assign_fringe_order(skeleton_image, false);

% Tái tạo bề mặt từ vân
[phi_est, ~] = reconSurface_linearPushed(img, labels, 632.8e-9, 'None', false);

% phi_est = rot90(phi_est,-1);
% Sau khi có phi_est
% systematic_error = (pi/5) * (2*rand(size(phi_est))-1); % random ±λ/20
% phi_est = phi_est + systematic_error;

% phi_est = phi_est - min(phi_est(:));
% 
% [X, Y] = meshgrid(1:N, 1:M);
% plane_phase = 2*pi*(fx*X + fy*Y);
% plane_phase = plane_phase - min(plane_phase(:));
% phi_est = phi_est - plane_phase - (max(phi_est(:)- max(plane_phase(:))))/2;

% figure;
% surf(phi_est,"EdgeColor","none");
% title("Anh pha phi estimate co nhieu");

% figure;
% imagesc(phi_est - object_phase_without_noise);
% title("Anh sai lech giua phi est va ground truth");
% colorbar;

%%
fprintf('--> Bước 2: Tái tạo pha Wrapped bằng FFT...\n');
if(auto_fft)
    [wrapped_phase, ~] = reconstruct_phase_auto(hologram, params);
else
    [wrapped_phase, ~] = reconstruct_phase_interactively(hologram, struct());
end

%% 8. GIẢI BỌC PHA VÀ TINH CHỈNH
% fprintf('--> Bước 4: Giải bọc pha và tinh chỉnh kết quả...\n');
% --- Giải bọc pha sử dụng pha ước lượng ---
% [est_phase_flat, wrapped_phase, object_phase] = crop_multiple_to_smallest(est_phase_flat, wrapped_phase, object_phase);

[finalUnwrappedPhase, kMap] = unwrapUsingEstimate(phi_est, wrapped_phase);


%% 10. Refine artifacts points

% [finalUnwrappedPhase, ~, ~] = correct_sparse_artifacts_iterative(finalUnwrappedPhase, ...
%     'BoundaryCondition', 'symmetric', 'BoundaryWidth', 2, 'MaxIterations', 150);

figure("Name","Kết quả sau refine");
surf(finalUnwrappedPhase, 'EdgeColor', 'none');
title("Kết quả finalUnwrappedPhase sau khi refine");
xlabel('x'); ylabel('y'); zlabel('(rad)');
colormap; colorbar; 

% Cắt biên để hiển thị tốt hơn
offset = 4;
finalUnwrappedPhase = finalUnwrappedPhase(offset+1:end-offset, offset+1:end-offset);
%% 11. CÁC THUẬT TOÁN UNWRAPPING KHÁC
% unwrapped_Phase_LS_DCT = unwrapping.unwrapPhase(wrapped_phase, 'ls', 'dct'); % LS với DCT
unwrapped_Phase_TIE_FFT = unwrapping.unwrapPhase(wrapped_phase, 'tie', 'fft'); % TIE với FFT
unwrapped_Phase_noncontinue = unwrapping.unwrapPhase(wrapped_phase, 'linh'); % Phương pháp của a Linh
unwrapped_Phase_2dweight = unwrapping.unwrapPhase(wrapped_phase, '2dweight'); % 2D weighted phase unwrapping
unwrapped_Phase_goldstein = goldstein_unwrap(wrapped_phase);
% Proposed method
unwrapped_Phase_proposal = finalUnwrappedPhase;
[ unwrapped_Phase_TIE_FFT, unwrapped_Phase_noncontinue,...
    unwrapped_Phase_2dweight, unwrapped_Phase_goldstein, unwrapped_Phase_proposal]...
    = crop_multiple_to_smallest(unwrapped_Phase_TIE_FFT, unwrapped_Phase_noncontinue,...
    unwrapped_Phase_2dweight, unwrapped_Phase_goldstein, unwrapped_Phase_proposal);
[M,N] = size(unwrapped_Phase_TIE_FFT);
finalUnwrappedPhase = unwrapped_Phase_proposal;

unwrapped_Phase_TIE_FFT  =  unwrapped_Phase_TIE_FFT- min(unwrapped_Phase_TIE_FFT(:));
unwrapped_Phase_noncontinue  =  unwrapped_Phase_noncontinue- min(unwrapped_Phase_noncontinue(:));
unwrapped_Phase_2dweight  =  unwrapped_Phase_2dweight- min(unwrapped_Phase_2dweight(:));
unwrapped_Phase_goldstein = unwrapped_Phase_goldstein - min(unwrapped_Phase_goldstein(:));
finalUnwrappedPhase  =  finalUnwrappedPhase- min(finalUnwrappedPhase(:));

[unwrapped_Phase_TIE_FFT, phi_plane] = remove_plane_manual(unwrapped_Phase_TIE_FFT);
phase_offset = phi_plane;
%   unwrapped_Phase_LS_DCT = unwrapped_Phase_LS_DCT - phase_offset;
unwrapped_Phase_noncontinue = unwrapped_Phase_noncontinue - phase_offset;
unwrapped_Phase_2dweight = unwrapped_Phase_2dweight - phase_offset;
unwrapped_Phase_goldstein = unwrapped_Phase_goldstein - phase_offset;
finalUnwrappedPhase = finalUnwrappedPhase - phase_offset;
% offset về 0
unwrapped_Phase_TIE_FFT  =  unwrapped_Phase_TIE_FFT- min(unwrapped_Phase_TIE_FFT(:));
unwrapped_Phase_noncontinue  =  unwrapped_Phase_noncontinue- min(unwrapped_Phase_noncontinue(:));
unwrapped_Phase_2dweight  =  unwrapped_Phase_2dweight- min(unwrapped_Phase_2dweight(:));
unwrapped_Phase_goldstein = unwrapped_Phase_goldstein - min(unwrapped_Phase_goldstein(:));
finalUnwrappedPhase  =  finalUnwrappedPhase- min(finalUnwrappedPhase(:));
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

% titles = {'Ground truth', 'TIE+FFT', ...
%           'Reliability-based', '2D Weighted', 'Goldstein', 'Proposed method'};

%% 6. PHÂN TÍCH SAI SỐ (TIẾP THEO)
% close all;

%% HIỂN THỊ KẾT QUẢ SAU TOÀN BỘ VÒNG LẶP
phase_names = {'TIE (FFT-based)', ...
               'Reliability-based', '2D-WLS', ...
               'Goldstein', 'Proposed'};
figure;
surf(unwrapped_Phase_TIE_FFT,"EdgeColor","none");
colorbar;
title('TIE (FFT-based)');

figure;
surf(unwrapped_Phase_noncontinue,"EdgeColor","none");
colorbar;
title('Reliability-based');

figure;
surf(unwrapped_Phase_goldstein,"EdgeColor","none");
colorbar;
title('Goldstein');

figure;
surf(finalUnwrappedPhase,"EdgeColor","none");
colorbar;
title('Proposed ');
figure;
surf(unwrapped_Phase_2dweight,"EdgeColor","none");
colorbar;
title('2D-WLS ');
%% ve mat cat ngang
% Giả sử tất cả các ma trận có cùng kích thước
row = round(size(finalUnwrappedPhase,1)/2);  % chọn hàng giữa

figure; hold on; grid on;

plot(unwrapped_Phase_TIE_FFT(row,:), 'LineWidth', 2, 'DisplayName', 'TIE-FFT');
plot(unwrapped_Phase_noncontinue(row,:), 'LineWidth', 2, 'DisplayName', 'Non-continue');
plot(unwrapped_Phase_2dweight(row,:), 'LineWidth', 2, 'DisplayName', '2D-weight');
plot(unwrapped_Phase_goldstein(row,:), 'LineWidth', 2, 'DisplayName', 'Goldstein');
plot(finalUnwrappedPhase(row,:), 'LineWidth', 2, 'DisplayName', 'Final');

xlabel('Pixel index');
ylabel('Phase value');
title('Cross-section comparison');
legend('show');
%%

% Danh sách tên thuật toán
phase_names = {'TIE (FFT-based)', ...
               'Reliability-based', ...
               '2D-WLS', ...
               'Goldstein', ...
               'Proposed'};

% Gom tất cả vào cell để dễ vẽ
phase_data = {unwrapped_Phase_TIE_FFT, ...
              unwrapped_Phase_noncontinue, ...
              unwrapped_Phase_2dweight, ...
              unwrapped_Phase_goldstein, ...
              finalUnwrappedPhase};

% Tìm min/max chung để scale màu đồng bộ
all_vals = cellfun(@(x) x(:), phase_data, 'UniformOutput', false);
all_vals = cat(1, all_vals{:});
cmin = min(all_vals);
cmax = max(all_vals);

% Vẽ
figure('Position',[100 100 1200 600]);
tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

for k = 1:numel(phase_data)
    nexttile;
    surf(phase_data{k}, 'EdgeColor','none');
    view(2); % nhìn từ trên xuống như ảnh
    axis equal tight;
    colormap jet;
    caxis([cmin cmax]); % scale chung
    title(phase_names{k}, 'FontSize',12,'FontWeight','bold');
    colorbar;
end

% --- Ô cuối cùng: mặt cắt ngang ---
nexttile;
row = round(size(finalUnwrappedPhase,1)/2); % lấy hàng giữa
hold on; grid on;

colors = lines(numel(phase_data));
for k = 1:numel(phase_data)
    plot(phase_data{k}(row,:), 'LineWidth', 2, ...
         'DisplayName', phase_names{k}, 'Color', colors(k,:));
end

xlabel('Pixel index');
ylabel('Phase value');
title('Cross-section (middle row)','FontSize',12,'FontWeight','bold');
legend('show','Location','best');


%% Hàm phụ trợ
%% --- FUNCTION MZS ---
function S = MZS_thinning(BW)
    S = BW > 0;
    prev = false(size(S));
    while true
        % Sub-iteration 1: even pixels
        marker = MZS_iteration(S, 0);
        S(marker) = 0;

        % Sub-iteration 2: odd pixels
        marker = MZS_iteration(S, 1);
        S(marker) = 0;

        if isequal(S, prev), break; end
        prev = S;
    end
end

function marker = MZS_iteration(S, parity)
    % Pad để tránh lỗi biên
    P = padarray(S, [1 1], 0, 'both');

    % Lấy lân cận (theo thứ tự ZS)
    P2 = P(1:end-2,2:end-1); % north
    P3 = P(1:end-2,3:end);   % northeast
    P4 = P(2:end-1,3:end);   % east
    P5 = P(3:end,3:end);     % southeast
    P6 = P(3:end,2:end-1);   % south
    P7 = P(3:end,1:end-2);   % southwest
    P8 = P(2:end-1,1:end-2); % west
    P9 = P(1:end-2,1:end-2); % northwest
    P1 = P(2:end-1,2:end-1); % center

    % Tổng số hàng xóm (B)
    B = P2+P3+P4+P5+P6+P7+P8+P9;

    % C(p1) theo công thức trong paper
    C = (~P2 & (P3|P4)) + (~P4 & (P5|P6)) + ...
        (~P6 & (P7|P8)) + (~P8 & (P9|P2));

    % Điều kiện chung
    cond = (C==1);

    if parity == 0   % subfield chẵn
        cond = cond & (mod(bsxfun(@plus,(1:size(S,1))',(1:size(S,2))),2)==0);
        cond = cond & (B>=2 & B<=7);
        cond = cond & (~(P2 & P4 & P6));
        cond = cond & (~(P4 & P6 & P8));
    else             % subfield lẻ
        cond = cond & (mod(bsxfun(@plus,(1:size(S,1))',(1:size(S,2))),2)==1);
        cond = cond & (B>=1 & B<=7);
        cond = cond & (~(P2 & P4 & P8));
        cond = cond & (~(P2 & P6 & P8));

        % Bổ sung điều kiện giữ pixel để bảo toàn 2x2 / diagonal
        diagNeighbors = (P3|P5|P7|P9);
        cond = cond & ~( (B==1) & diagNeighbors );
    end

    marker = P1 & cond;
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

function [BW_clean, junction] = removeJunctions(BW)
% REMOVEJUNCTIONS - Phát hiện và loại bỏ junction pixels trong skeleton
%
% Syntax:
%   [BW_clean, junction] = removeJunctions(BW)
%
% Input:
%   BW - ảnh nhị phân skeleton
%
% Output:
%   BW_clean - skeleton sau khi xoá junction
%   junction - ảnh nhị phân, 1 tại vị trí junction pixel
%
% Đặc điểm:
%   - Junction pixel: có >=3 hàng xóm trong 8 hướng
%   - Bỏ qua các pixel sát biên (4 pixel)

% --- 1. Đếm số hàng xóm ---
kernel = ones(3,3); kernel(2,2) = 0;  % 8-neighborhood
neighborCount = conv2(double(BW), kernel, 'same');

% --- 2. Junction: skeleton pixel có >= 3 hàng xóm ---
junction = (BW == 1) & (neighborCount >= 3);

% --- 3. Không xét biên (4 pixel) ---
junction(1:2,:)       = 0;
junction(end-1:end,:) = 0;
junction(:,1:2)       = 0;
junction(:,end-1:end) = 0;

% --- 4. Xóa junction ---
BW_clean = BW;
BW_clean(junction) = 0;
end

function [BW, linePix] = drawLine(BW, x1, y1, x2, y2)
% Vẽ line nối từ (x1,y1) đến (x2,y2) bằng thuật toán Bresenham
[h, w] = size(BW);
[lineX, lineY] = bresenham(x1, y1, x2, y2);

linePix = [lineX(:), lineY(:)];

for k = 1:length(lineX)
    cx = lineX(k);
    cy = lineY(k);
    if cx >= 1 && cx <= w && cy >= 1 && cy <= h
        BW(cy, cx) = 1;
    end
end

end

function [x, y] = bresenham(x1, y1, x2, y2)

% Thuật toán Bresenham
x1 = round(x1); y1 = round(y1);
x2 = round(x2); y2 = round(y2);

dx = abs(x2 - x1);
dy = abs(y2 - y1);

sx = sign(x2 - x1);
sy = sign(y2 - y1);

err = dx - dy;

x = []; y = [];
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

%% ----- Hàm phụ trợ -----

%% connect khi xét qua toàn bộ endpoint và tìm điểm phù hợp nhất
function [BW_new, linesConnected] = connectEndpoints_v3(BW, vectors, CC, minCompSize, maxDist, vecAlignThr,maxPerp)
% BW           : skeleton binary
% vectors      : [cx cy vx vy] từ hàm computeEndpointVectors
% CC           : bwconncomp(BW,8)
% minCompSize  : kích thước tối thiểu của vân
% maxDist      : khoảng cách tối đa cho phép nối
% vecAlignThr  : ngưỡng cos(angle) hướng (ví dụ 0.7 ~ >45°)

BW_new = BW; % copy để cập nhật nối
linesConnected = {}; % cell lưu danh sách các đoạn đã nối

used = false(size(vectors,1),1); % đánh dấu endpoint đã được nối

for i = 1:size(vectors,1)-1
    if used(i), continue; end  % bỏ qua nếu endpoint i đã nối

    cx1 = vectors(i,1); cy1 = vectors(i,2);
    v1  = [vectors(i,3), vectors(i,4)];

    % kiểm tra component của endpoint i
    comp_id1 = findComponent(CC, [cy1,cx1]);
    if comp_id1==0 || numel(CC.PixelIdxList{comp_id1}) < minCompSize
        continue;
    end

    best_j   = 0;
    bestCost = inf;

    for j = i+1:size(vectors,1)
        if used(j), continue; end

        cx2 = vectors(j,1); cy2 = vectors(j,2);
        v2  = [vectors(j,3), vectors(j,4)];

        % kiểm tra component j
        comp_id2 = findComponent(CC, [cy2,cx2]);
        if comp_id2==0 || numel(CC.PixelIdxList{comp_id2}) < minCompSize
            continue;
        end

        % --- khoảng cách Euclidean ---
        d = hypot(cx1-cx2, cy1-cy2);
        if d > maxDist, continue; end

        % --- hướng vector ---
        dir12 = [cx2-cx1, cy2-cy1];
        dir12 = dir12 / (norm(dir12)+eps);

        cond1 = dot(v1, dir12) > vecAlignThr;
        cond2 = dot(v2, -dir12) > vecAlignThr;
        if ~(cond1 && cond2), continue; end

        % --- khoảng cách vuông góc ---
        a = -v2(2);
        b =  v2(1);
        c =  v2(2)*cx2 - v2(1)*cy2;
        d_perp = abs(a*cx1 + b*cy1 + c) / sqrt(a^2 + b^2);

        if d_perp > maxPerp, continue; end

        % --- tính "cost" để chọn ứng viên tốt nhất ---
        % --- khoảng cách chuẩn hóa ---
        d_norm = d / maxDist;  % [0,1]

        % --- sai lệch góc ---
        ang1 = acos(dot(v1, dir12));     % góc v1 với hướng nối
        ang2 = acos(dot(v2, -dir12));    % góc v2 với hướng nối ngược
        ang_err = (ang1 + ang2)/2;       % sai số góc trung bình (rad)

        ang_norm = ang_err / (pi/2);     % chuẩn hóa [0,1], 0 tốt, 1 tệ

        % --- khoảng cách vuông góc (cũng chuẩn hóa) ---
        d_perp_norm = min(d_perp/10,1);  % giới hạn max =1
        w1=0.2; w2=0.7; w3=0.1;
        % --- cost tổng hợp ---
        cost = w1*d_norm + w2*ang_norm + w3*d_perp_norm;
        if cost < bestCost
            bestCost = cost;
            best_j   = j;
        end
    end

    % Sau khi duyệt hết, nếu tìm thấy ứng viên tốt nhất thì nối
    if best_j > 0
        cx2 = vectors(best_j,1); cy2 = vectors(best_j,2);
        [BW_new, linePixels] = drawLine(BW_new, cx1, cy1, cx2, cy2);
        linesConnected{end+1} = linePixels; %#ok<AGROW>
        used([i best_j]) = true;
    end

end
end

%% helper: estimate local direction via PCA on nearest pixels in same component
function BW_out = extendLineNearBorder(BW, vectors, extendLen, margin)
% extendLineNearBorder - Nối dài endpoint ra ngoài NẾU nó gần biên ảnh
%
% Input:
%   BW        - ảnh nhị phân
%   vectors   - [cx, cy, vx, vy] cho mỗi endpoint
%   extendLen - số pixel muốn nối dài thêm
%   margin    - ngưỡng khoảng cách từ biên (ví dụ 5)
%
% Output:
%   BW_out    - ảnh nhị phân sau khi vẽ đoạn thẳng nối dài

[H,W] = size(BW);
BW_out = BW;

for i = 1:size(vectors,1)
    cx = vectors(i,1);
    cy = vectors(i,2);
    vx = vectors(i,3);
    vy = vectors(i,4);

    % --- CHỈ vẽ nếu endpoint gần biên ---
    if ~(cx <= margin || cx >= W-margin || cy <= margin || cy >= H-margin)
        continue; % bỏ qua nếu không gần biên
    end

    % Tính điểm mới (C = B + extendLen*v)
    x3 = cx + extendLen*vx;
    y3 = cy + extendLen*vy;

    % Bresenham từ (cx,cy) đến (x3,y3)
    [xLine, yLine] = bresenham2(round(cx), round(cy), round(x3), round(y3));

    % Loại pixel ngoài biên
    mask = xLine>=1 & xLine<=W & yLine>=1 & yLine<=H;
    xLine = xLine(mask);
    yLine = yLine(mask);

    % Vẽ vào ảnh
    BW_out(sub2ind([H,W], yLine, xLine)) = 1;
end

end

%% --- Hàm Bresenham ---
function [x,y] = bresenham2(x1,y1,x2,y2)
x1=round(x1); y1=round(y1);
x2=round(x2); y2=round(y2);

dx=abs(x2-x1); dy=abs(y2-y1);
sx=sign(x2-x1); sy=sign(y2-y1);

x=x1; y=y1;
xx=[]; yy=[];

if dx > dy
    err = dx/2;
    while x ~= x2
        xx(end+1)=x; yy(end+1)=y;
        x = x + sx;
        err = err - dy;
        if err < 0
            y = y + sy;
            err = err + dx;
        end
    end
else
    err = dy/2;
    while y ~= y2
        xx(end+1)=x; yy(end+1)=y;
        y = y + sy;
        err = err - dx;
        if err < 0
            x = x + sx;
            err = err + dy;
        end
    end
end
xx(end+1)=x2; yy(end+1)=y2;
x=xx; y=yy;
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
%     BW_clean = bwmorph(BW_clean, 'spur', 1);  % loại bỏ các nhánh nhỏ lẻ

end
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
%% ========================================================================

% -------------------------------------------------------------------------
function [unwrappedPhase, kMap] = unwrapUsingEstimate(estimatedPhase, wrappedPhase)
    % Giải Wrapped pha `wrappedPhase` dựa trên pha ước lượng `estimatedPhase`.
%     wrappedEstimate = wrapToPi(estimatedPhase);
    kMap = round((estimatedPhase - wrappedPhase) / (2*pi));
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

%     % Hiển thị thống kê
%     fprintf('Đã phát hiện %d vân\n', fringe_order);
%     fprintf('Số vân được gán nhãn: %d\n', sum(~isnan(labels)));
%     if ~isempty(valid_labels)
%         fprintf('Phạm vi bậc vân: %d đến %d\n', min(unique_labels), max(unique_labels));
%     end

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
F = scatteredInterpolant(X, Y, Z, 'natural', 'nearest');
Zq = F(xq, yq);
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

function [wrappedPhase, params] = reconstruct_phase_auto(hologram, params)
% Tái tạo pha từ hologram bằng cách lọc trong miền tần số với lựa chọn tự động.
%
% Chức năng sẽ tự động tìm phổ bậc +1 ở nửa trên của miền tần số,
% tạo một bộ lọc (tròn hoặc HCN) và tiến hành tái tạo pha.
%
% Tham số (params) có thể chứa:
% params.filter_type: 'circle' hoặc 'rectangle' (mặc định: 'circle')
% params.filter_radius: Bán kính của bộ lọc tròn (mặc định: 40)
% params.filter_width: Chiều rộng bộ lọc HCN (mặc định: 80)
% params.filter_height: Chiều cao bộ lọc HCN (mặc định: 80)
% params.dc_suppression_radius: Bán kính để loại bỏ thành phần DC (mặc định: 25)

% --- Kiểm tra và đặt giá trị mặc định cho params ---
if ~exist('params', 'var')
    params = struct();
end
if ~isfield(params, 'filter_type')
    params.filter_type = 'circle'; % 'circle' hoặc 'rectangle'
end
if ~isfield(params, 'filter_radius')
    params.filter_radius = 50; % Bán kính của bộ lọc tròn
end
if ~isfield(params, 'filter_width')
    params.filter_width = 100; % Chiều rộng bộ lọc HCN
end
if ~isfield(params, 'filter_height')
    params.filter_height = 100; % Chiều cao bộ lọc HCN
end
if ~isfield(params, 'dc_suppression_radius')
    params.dc_suppression_radius = 25; % Bán kính vùng trung tâm để loại bỏ
end

% --- Xử lý ban đầu ---
hologramGray = myConvGrayScale(hologram);
[numRows, numCols] = size(hologramGray);
fourierTransform = fftshift(fft2(hologramGray));
spectrumMagnitude = abs(fourierTransform);

% --- Tự động tìm kiếm phổ bậc +1 ---

% Tọa độ tâm của phổ
u0 = floor(numCols / 2) + 1;
v0 = floor(numRows / 2) + 1;

% Tạo một bản sao của phổ cường độ để tìm kiếm
searchSpectrum = spectrumMagnitude;

% Loại bỏ thành phần DC (bậc 0) để tránh chọn nhầm
[U, V] = meshgrid(1:numCols, 1:numRows);
dist_from_center = sqrt((U - u0).^2 + (V - v0).^2);
searchSpectrum(dist_from_center <= params.dc_suppression_radius) = 0;

% Chỉ tìm kiếm ở nửa trên của phổ (nơi thường chứa phổ bậc +1)
upperHalfSpectrum = searchSpectrum(1:v0-1, :);

% Tìm tọa độ của điểm có cường độ lớn nhất
[~, maxIdx] = max(upperHalfSpectrum(:));
[v_max, u_max] = ind2sub(size(upperHalfSpectrum), maxIdx);
% (v_max, u_max) là tọa độ của tâm vùng ROI được chọn tự động

% --- Hiển thị phổ Fourier và vùng được chọn tự động ---
figure('Name','Phổ Fourier và Vùng chọn tự động');
imshow(log(1 + spectrumMagnitude), []);
hold on;

% Vẽ hình dạng bộ lọc tương ứng
if strcmp(params.filter_type, 'circle')
    theta = 0:0.01:2*pi;
    x_circle = params.filter_radius * cos(theta) + u_max;
    y_circle = params.filter_radius * sin(theta) + v_max;
    plot(x_circle, y_circle, 'g', 'LineWidth', 2);
    title(['Phổ bậc +1 (Tròn) tại (', num2str(u_max), ', ', num2str(v_max), ')']);
else % rectangle
    rect_x = u_max - params.filter_width/2;
    rect_y = v_max - params.filter_height/2;
    rectangle('Position', [rect_x, rect_y, params.filter_width, params.filter_height], ...
        'EdgeColor', 'g', 'LineWidth', 2);
    title(['Phổ bậc +1 (HCN) tại (', num2str(u_max), ', ', num2str(v_max), ')']);
end
hold off;

% --- Tạo bộ lọc và trích xuất phổ ---

% Tạo mask tương ứng với loại bộ lọc
if strcmp(params.filter_type, 'circle')
    % Bộ lọc hình tròn
    roi_mask = sqrt((U - u_max).^2 + (V - v_max).^2) <= params.filter_radius;
else
    % Bộ lọc hình chữ nhật
    roi_mask = (abs(U - u_max) <= params.filter_width/2) & ...
        (abs(V - v_max) <= params.filter_height/2);
end

% Áp dụng mask để chỉ giữ lại phổ bậc +1
filteredContent = fourierTransform .* roi_mask;

% % Dịch chuyển vùng phổ đã chọn về lại tâm của ma trận
% dich_chuyen = 0;
% % Tính toán độ dịch chuyển cần thiết
% v_shift = v0 - v_max;
% u_shift = u0 - u_max - dich_chuyen;
% 
% % Dùng circshift để dịch chuyển
% filteredSpectrum = circshift(filteredContent, [v_shift, u_shift]);

filteredSpectrum = filteredContent;
% --- Hiển thị kết quả phổ sau khi lọc và dịch chuyển ---
figure('Name','Phổ sau khi xử lý');
imshow(log(1 + abs(filteredSpectrum)), []);
title(['Phổ bậc +1 (', params.filter_type, ') sau khi lọc và dịch về tâm']);

% --- Tái tạo trường sóng phức và lấy pha ---
finalPhaseComplex = ifft2(ifftshift(filteredSpectrum));

% Lấy pha từ trường phức
wrappedPhase = angle(finalPhaseComplex);
end
function [wrappedPhase, params] = reconstruct_phase_interactively(hologram, params)
% RECONSTRUCT_PHASE_INTERACTIVELY_MASK Tái tạo pha từ hologram bằng cách
% dùng MẶT NẠ để lọc phổ bậc +1 một cách tương tác.
%
%   Input:
%       hologram - Ảnh hologram đầu vào (có thể là ảnh màu hoặc ảnh xám).
%       params   - Một struct chứa các tham số (tùy chọn).
%
%   Output:
%       wrappedPhase - Pha đã tái tạo (bị gói trong khoảng [-pi, pi]).
%       params       - Struct tham số được cập nhật (tùy chọn).

% 1. Chuyển đổi hologram sang ảnh xám nếu cần thiết.
if size(hologram, 3) == 3
    hologramGray = rgb2gray(hologram);
else
    hologramGray = hologram;
end

[numRows, numCols] = size(hologramGray);

% 2. Thực hiện biến đổi Fourier 2D và dịch chuyển thành phần tần số 0 về tâm.
fourierTransform = fftshift(fft2(double(hologramGray)));

% 3. Hiển thị phổ Fourier để người dùng lựa chọn.
figure('Name', 'Fourier Spectrum - Select +1 Order');
imshow(log(1 + abs(fourierTransform)), []);
title('Vẽ một hình chữ nhật quanh phổ bậc +1 rồi double-click');
xlabel('Tần số không gian (u)');
ylabel('Tần số không gian (v)');

% 4. Cho phép người dùng chọn vùng quan tâm (ROI) bằng tay.
[~, xRec, yRec, widthRec, heightRec] = myDrawRec();

% 5. TẠO MỘT MẶT NẠ (MASK) TỪ VÙNG ĐÃ CHỌN
%    Tạo một ma trận toàn số 0...
mask = zeros(numRows, numCols);
%    ...và đặt vùng chữ nhật đã chọn thành 1.
mask(yRec:yRec+heightRec-1, xRec:xRec+widthRec-1) = 1;

% 6. ÁP DỤNG MẶT NẠ VÀ DỊCH CHUYỂN VỀ TÂM
%    Nhân phổ gốc với mặt nạ để loại bỏ các tần số bên ngoài vùng chọn.
filteredSpectrum = fourierTransform .* mask;


% 7. Thực hiện biến đổi Fourier ngược để tái tạo trường sóng phức.
complexField = ifft2(ifftshift(filteredSpectrum));

% 8. Lấy pha từ trường phức.
wrappedPhase = angle(complexField);

end
function [pos, xRec, yRec, widthRec, heightRec] = myDrawRec()
% Cho phép người dùng vẽ một hình chữ nhật (ROI) trên ảnh hiện tại.
hFig = gcf;
hROI = drawrectangle();
centerRec = [hROI.Position(1) + hROI.Position(3)/2, hROI.Position(2) + hROI.Position(4)/2];
hold on;
hMarker = plot(centerRec(1), centerRec(2), 'r+', 'MarkerSize', 10, 'LineWidth', 2);
hold off;
addlistener(hROI, 'MovingROI', @(src, evt) updateCenterRectangle(src, hMarker));

% Đợi người dùng double-click để xác nhận
wait(hROI);

pos = round(hROI.Position);
xRec = pos(1); yRec = pos(2);
widthRec = pos(3); heightRec = pos(4);

% Đóng cửa sổ sau khi đã chọn xong
if ishandle(hFig)
    close(hFig);
end
end

% -------------------------------------------------------------------------
function updateCenterRectangle(roi, centerMarker)
% Cập nhật vị trí dấu cộng ở tâm ROI khi đang di chuyển.
centerMarker.XData = roi.Position(1) + roi.Position(3)/2;
centerMarker.YData = roi.Position(2) + roi.Position(4)/2;
drawnow;
end

% -------------------------------------------------------------------------
function output = myConvGrayScale(inputImage)
% Chuyển ảnh đầu vào sang ảnh grayscale kiểu double.
if size(inputImage, 3) > 1
    inputImage = rgb2gray(inputImage);
end
output = double(inputImage);
end
function params = set_default_params(params)

    if ~exist('params', 'var') || isempty(params)
        params = struct();
    end

    def = struct(...
        'filter_type', 'circle', ...
        'filter_radius', 50, ...
        'filter_width', 100, ...
        'filter_height', 100, ...
        'dc_suppression_radius', 25, ...
        'lambda', 632.8e-9, ...
        'pixel_size', 3.45e-6, ...
        'unwrap_method', 'least_square', ...
        'phase_smoothing', true, ...
        'smoothing_sigma', 2, ...
        'show_figures', true, ...
        'verbose', true ...
    );

    fn = fieldnames(def);
    for i = 1:length(fn)
        if ~isfield(params, fn{i})
            params.(fn{i}) = def.(fn{i});
        end
    end

end
function [phi_corrected, phi_plane] = remove_plane_manual(phi)
%REMOVE_PLANE_MANUAL Cho phép người dùng chọn điểm hoặc vẽ HCN để nội suy và loại mặt phẳng nghiêng
% [phi_corrected, phi_plane] = remove_plane_manual(phi)
% - phi: bản đồ pha đầu vào
% - phi_corrected: bản đồ sau khi loại nghiêng
% - phi_plane: mặt phẳng đã nội suy

[N, M] = size(phi);
[X, Y] = meshgrid(1:M, 1:N);

% Kiểm tra và xử lý NaN/Inf trong dữ liệu đầu vào
if any(~isfinite(phi(:)))
    warning('Dữ liệu chứa NaN hoặc Inf. Đang thay thế bằng giá trị trung bình...');
    phi_mean = nanmean(phi(:));
    phi(~isfinite(phi)) = phi_mean;
end

% % --- Hiển thị ảnh ban đầu để người dùng chọn phương thức ---
% figure;
% surf(phi, "EdgeColor", "none");
% colormap jet;
% colorbar;
% title('Bản đồ pha gốc');
% 
% figure;
% imagesc(phi);
% axis image;
% colormap jet;
% colorbar;
% title('Bản đồ pha gốc');

% --- Hộp thoại lựa chọn phương thức ---
% choice = questdlg('Chọn phương thức để xác định mặt phẳng:', ...
%     'Lựa chọn nội suy', ...
%     'Chọn điểm', 'Vẽ HCN', 'Chọn điểm');
choice = "Vẽ HCN";
% --- Lấy điểm dựa trên lựa chọn của người dùng ---
switch choice
    case 'Chọn điểm'
        % --- Chức năng GINPUT nguyên bản: chọn điểm thủ công ---
        title('Chọn các điểm trên mặt phẳng cần nội suy (ấn Enter khi xong)');
        [x_pts, y_pts] = ginput();

        if isempty(x_pts)
            disp('Không có điểm nào được chọn. Đang hủy bỏ...');
            phi_corrected = phi;
            phi_plane = zeros(N, M);
            return;
        end

    case 'Vẽ HCN'
        % --- Chức năng GETRECT mới: vẽ hình chữ nhật ---
        title('Vẽ một hình chữ nhật trên vùng cần nội suy');
        %         rect = getrect; % [xmin ymin width height]
        %
        %         % Lấy tọa độ 4 góc từ hình chữ nhật
        %         xmin = rect(1);
        %         ymin = rect(2);
        %         width = rect(3);
        %         height = rect(4);
        %         x_pts = [xmin; xmin + width; xmin + width; xmin];
        %         y_pts = [ymin; ymin; ymin + height; ymin + height];
        % Lấy kích thước của ma trận phi
        [rows, cols] = size(phi);

        % Xác định tọa độ x (cột) và y (hàng) của 4 góc
        % Thứ tự: trên-trái, trên-phải, dưới-phải, dưới-trái
        offset = 5;
        x_pts = [offset;    cols-offset; cols-offset; offset];
        y_pts = [offset;    offset;    rows-offset; rows-offset];
        width = cols -2*offset;
        height = rows - 2*offset;
        
        if width == 0 || height == 0
            disp('Hình chữ nhật không hợp lệ. Đang hủy bỏ...');
            phi_corrected = phi;
            phi_plane = zeros(N, M);
            return;
        end

    case ''
        % Người dùng đã đóng hộp thoại
        disp('Không có lựa chọn nào được thực hiện. Đang hủy bỏ...');
        phi_corrected = phi;
        phi_plane = zeros(N, M);
        return;
end

% --- Kiểm tra và làm sạch tọa độ điểm ---
% Đảm bảo tọa độ nằm trong phạm vi hợp lệ
x_pts = max(1, min(M, x_pts));
y_pts = max(1, min(N, y_pts));

% --- Lấy giá trị Z tại các điểm đã chọn ---
z_pts = interp2(phi, x_pts, y_pts);

% Kiểm tra và loại bỏ các điểm có giá trị NaN
valid_pts = isfinite(x_pts) & isfinite(y_pts) & isfinite(z_pts);

if sum(valid_pts) < 3
    warning('Không đủ điểm hợp lệ để fit mặt phẳng (cần ít nhất 3 điểm). Trả về dữ liệu gốc.');
    phi_corrected = phi;
    phi_plane = zeros(N, M);
    return;
end

% Lọc các điểm hợp lệ
x_pts = x_pts(valid_pts);
y_pts = y_pts(valid_pts);
z_pts = z_pts(valid_pts);

% % --- Hiển thị lại ảnh với các điểm đã chọn ---
% figure;
% imagesc(phi);
% axis image;
% colormap jet;
% hold on;
% plot(x_pts, y_pts, 'rx', 'MarkerSize', 12, 'LineWidth', 2);
% 
% if strcmp(choice, 'Vẽ HCN')
%     % Vẽ lại hình chữ nhật để xác nhận
%     rect_x = [x_pts' x_pts(1)];
%     rect_y = [y_pts' y_pts(1)];
%     plot(rect_x, rect_y, 'r-', 'LineWidth', 2);
% end
% 
% for i = 1:length(x_pts)
%     text(x_pts(i) + 5, y_pts(i), sprintf('%d', i), ...
%         'Color', 'w', 'FontSize', 10, 'FontWeight', 'bold');
% end
% title('Pha gốc với các điểm nội suy đã chọn');
% hold off;

% --- Fit mặt phẳng với xử lý lỗi ---
try
    % Phương pháp 1: Sử dụng fit() với dữ liệu đã làm sạch
    tbl = table(x_pts, y_pts, z_pts, 'VariableNames', {'x', 'y', 'z'});
    fit_model = fit([tbl.x, tbl.y], tbl.z, 'poly11'); % poly11: f(x,y) = p00 + p10*x + p01*y

    % Tạo mặt phẳng đã khớp trên toàn bộ lưới tọa độ
    phi_plane = fit_model(X, Y);


end

% Kiểm tra kết quả phi_plane
if any(~isfinite(phi_plane(:)))
    warning('Mặt phẳng fit chứa NaN hoặc Inf. Đang thay thế...');
    phi_plane(~isfinite(phi_plane)) = 0;
end

% --- Trừ mặt phẳng (nghiêng) khỏi pha gốc ---
phi_corrected = phi - phi_plane;

% % --- Hiển thị kết quả ---
% figure;
% sgtitle('Kết quả loại bỏ mặt phẳng nghiêng');
% 
% subplot(1,3,1);
% imagesc(phi);
% axis image;
% colormap turbo;
% colorbar;
% title('Pha gốc');
% 
% subplot(1,3,2);
% imagesc(phi_plane);
% axis image;
% colormap turbo;
% colorbar;
% title('Mặt phẳng đã fit');
% 
% subplot(1,3,3);
% imagesc(phi_corrected);
% axis image;
% colormap turbo;
% colorbar;
% title('Pha đã loại nghiêng');
% 
% % In thông tin về quá trình fit
% fprintf('Đã sử dụng %d điểm để fit mặt phẳng.\n', length(x_pts));
% fprintf('Phạm vi giá trị pha gốc: [%.3f, %.3f]\n', min(phi(:)), max(phi(:)));
% fprintf('Phạm vi giá trị pha đã hiệu chỉnh: [%.3f, %.3f]\n', min(phi_corrected(:)), max(phi_corrected(:)));

end
