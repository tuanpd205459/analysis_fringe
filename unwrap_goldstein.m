function [im_unwrapped, branch_cuts] = unwrap_goldstein(im_phase, im_mask, im_mag, max_box_radius)
% GOLDSTEIN_UNWRAP_FUNC Unwrap pha bằng thuật toán Goldstein Branch Cut
%
% Inputs:
%   im_phase       : Ma trận pha bị cuốn (Wrapped Phase) [-pi, pi]
%   im_mask        : (Tùy chọn) Mặt nạ nhị phân (1 = vùng tốt, 0 = nền/nhiễu). 
%                    Mặc định là toàn bộ ảnh = 1.
%   im_mag         : (Tùy chọn) Ảnh biên độ/cường độ (Magnitude). Dùng để chọn điểm
%                    bắt đầu (seed) tốt nhất. Mặc định là ones.
%   max_box_radius : (Tùy chọn) Bán kính tìm kiếm branch cut. Mặc định = 4.
%
% Outputs:
%   im_unwrapped   : Ảnh pha đã được trải (Unwrapped Phase).
%   branch_cuts    : Bản đồ các đường cắt (dùng để debug/hiển thị).

    %% 1. Xử lý tham số đầu vào mặc định
    if nargin < 4
        max_box_radius = 10;
    end
    if nargin < 3 || isempty(im_mag)
        im_mag = ones(size(im_phase)); % Nếu không có ảnh biên độ, giả sử đồng nhất
    end
    if nargin < 2 || isempty(im_mask)
        im_mask = ones(size(im_phase)); % Nếu không có mask, lấy toàn bộ ảnh
    end

    %% 2. Tính toán Residues (Điểm bất thường)
    % Hàm PhaseResidues_r1 cần có sẵn trong thư mục làm việc của bạn
    residue_charge = PhaseResidues_r1(im_phase, im_mask); 

    %% 3. Tạo Branch Cuts (Đường cắt)
    % Hàm BranchCuts_r1 cần có sẵn trong thư mục làm việc của bạn
    branch_cuts = BranchCuts_r1(residue_charge, max_box_radius, im_mask);
    
    % Cập nhật mask: Loại bỏ các điểm nằm trên đường cắt để FloodFill không đi qua
    process_mask = im_mask;
    process_mask(branch_cuts == 1) = 0; 
    
    % Loại bỏ các điểm có magnitude = 0 hoặc mask = 0 khỏi ứng cử viên điểm bắt đầu
    im_mag_masked = im_mag .* process_mask;

    %% 4. Tự động tìm điểm bắt đầu (Seed Point)
    % Chọn điểm có cường độ cao nhất để làm mốc tham chiếu pha chuẩn.
    % Thay thế cho việc dùng ginput thủ công.
    
    % Xóa biên để tránh chọn điểm sát mép
    im_mag_masked(1,:) = 0; im_mag_masked(end,:) = 0;
    im_mag_masked(:,1) = 0; im_mag_masked(:,end) = 0;
    
    max_val = max(im_mag_masked(:));
    if max_val == 0
        % Trường hợp ảnh đen xì hoặc mask che hết, lấy điểm giữa ảnh
        rowref = round(size(im_phase, 1) / 2);
        colref = round(size(im_phase, 2) / 2);
    else
        [rowrefn, colrefn] = find(im_mag_masked >= 0.99 * max_val);
        rowref = rowrefn(1);
        colref = colrefn(1);
    end

    %% 5. Thực hiện Unwrap (Flood Fill)
    % Hàm FloodFill_r1 cần có sẵn trong thư mục làm việc của bạn
    im_unwrapped = FloodFill_r1(im_phase, im_mag, branch_cuts, process_mask, colref, rowref);

end