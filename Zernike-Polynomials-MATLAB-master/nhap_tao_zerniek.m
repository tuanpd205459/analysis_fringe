% Script ví dụ để tái tạo và hiển thị một bề mặt Zernike phức hợp

% Đảm bảo các file zernike.m, zernike_radial.m, và 
% reconstructZernikeSurface.m nằm trong cùng thư mục hoặc trên path.

% --- 1. Định nghĩa các hệ số Zernike ---
% Chúng ta sẽ tạo một bề mặt có 0.8 đơn vị Defocus và -0.5 đơn vị Astigmatism.
% --- 1. Định nghĩa 15 hệ số Zernike đầu tiên ---
% Chúng ta sẽ tạo một bề mặt phức hợp bao gồm các quang sai chính.
% Mỗi vị trí trong vector tương ứng với một chỉ số Zernike (j).
%
%   j | (n,m)  | Quang sai             | Giá trị ví dụ |
%   --------------------------------------------------------------
%   1 | (0,0)  | Piston                |      0        | Bỏ qua Piston (chỉ là độ dời)
%   2 | (1,-1) | Tilt (Y)              |      0        | Bỏ qua Tilt (chỉ là độ nghiêng)
%   3 | (1,1)  | Tilt (X)              |      0        |
%   4 | (2,-2) | Astigmatism (xiên)    |     -0.3      | Thêm một ít quang sai loạn thị
%   5 | (2,0)  | Defocus (lệch tiêu)   |      0.6      | Thêm một lượng đáng kể Defocus
%   6 | (2,2)  | Astigmatism (thẳng)  |      0        |
%   7 | (3,-3) | Trefoil (tỏa tròn 3 lá)|     0.25      | Thêm quang sai bậc cao Trefoil
%   8 | (3,-1) | Coma (dọc)            |     -0.4      | Thêm quang sai Coma (hình sao chổi)
%   9 | (3,1)  | Coma (ngang)          |      0        |
%  10 | (3,3)  |                       |      0        |
%  11 | (4,-4) |                       |      0        |
%  12 | (4,-2) | Secondary Astigmatism |      0        |
%  13 | (4,0)  | Spherical (cầu sai)   |      0.35     | Thêm cầu sai bậc nhất
%  14 | (4,2)  |                       |      0        |
%  15 | (4,4)  |                       |      0        |
%
%%
clc, clear, close all;
%%
coefficients = [0, 0, 0, -0.3, 0.6, 0, 0.25, -0.4, 0, 0, 0, 0, 0.35, 0, 0];

% Kích thước lưới
gridSize = 512;

% --- 2. Gọi hàm để tái tạo bề mặt ---
fprintf('Đang tái tạo bề mặt từ %d hệ số...\n', numel(coefficients));
wavefront = reconstructZernikeSurface(coefficients, gridSize);
fprintf('Hoàn thành.\n');

% --- 3. Trực quan hóa kết quả ---
figure('Name', 'Bề mặt Zernike tái tạo');

% Dùng 'surf' để có cái nhìn 3D trực quan
surf(wavefront, 'EdgeColor', 'none', 'FaceColor', 'interp');
title('Bề mặt tái tạo từ hệ số Defocus và Astigmatism');
xlabel('X');
ylabel('Y');
zlabel('Phase');
colorbar; % Hiển thị thang màu
colormap('jet');


function W = reconstructZernikeSurface(coefficients, gridSize)
%reconstructZernikeSurface Tái tạo bề mặt wavefront từ một vector hệ số Zernike.
%
%   Input:
%       coefficients (vector): Một vector hàng chứa các hệ số Zernike. Thứ tự
%                              phải tuân theo chuẩn ANSI.
%       gridSize (integer, optional): Kích thước của lưới vuông đầu ra.
%                                      Mặc định là 256.
%
%   Output:
%       W (matrix): Bề mặt wavefront cuối cùng, là tổng có trọng số của
%                   các đa thức Zernike.

    % --- 0. Xử lý đầu vào tùy chọn ---
    if nargin < 2
        gridSize = 256; % Giá trị mặc định
    end
    
    % --- 1. Tạo lưới tọa độ ---
    x = linspace(-1, 1, gridSize);
    y = linspace(-1, 1, gridSize);
    [X, Y] = meshgrid(x, y);
    [t, r] = cart2pol(X, Y);

    % --- 2. Khởi tạo bề mặt và lặp qua các hệ số ---
    
    % Số lượng đa thức cần thêm chính là độ dài của vector hệ số
    num_polynomials = numel(coefficients);
    
    % Khởi tạo bề mặt cuối cùng là một ma trận toàn số 0
    W = zeros(gridSize, gridSize);
    
    j = 1; % Bắt đầu với chỉ số Zernike đầu tiên
    n = 0; % Bắt đầu với bậc xuyên tâm n=0
    
    % Vòng lặp sẽ tiếp tục cho đến khi chúng ta đã xử lý tất cả các hệ số
    while j <= num_polynomials
        % Lặp qua các bậc góc m hợp lệ cho n hiện tại
        for m = -n:2:n
            if j > num_polynomials
                break; % Thoát khỏi vòng lặp nếu đã hết hệ số
            end
            
            % Lấy hệ số (độ lớn) tương ứng
            C = coefficients(j);
            
            % Chỉ tính toán nếu hệ số khác 0 để tiết kiệm thời gian
            if C ~= 0
                % Gọi hàm zernike bạn đã cung cấp để tạo 1 đa thức đơn lẻ
                zern_term = zernike(r, t, n, m);
                
                % Cộng đa thức này vào bề mặt tổng, với trọng số là hệ số C
                W = W + C * zern_term;
            end
            
            % Chuyển sang hệ số tiếp theo
            j = j + 1;
        end
        % Chuyển sang bậc xuyên tâm tiếp theo
        n = n + 1;
        
        if j > num_polynomials
            break; % Thoát khỏi vòng lặp ngoài
        end
    end
    
    % --- 3. Mask (che) các giá trị bên ngoài vòng tròn đơn vị ---
    % Thay vì dùng hàm 'elliptical_crop', chúng ta có thể làm trực tiếp
    % để code được độc lập.
    W(r > 1) = NaN;

end