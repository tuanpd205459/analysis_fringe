function [Z, n_modes, m_modes] = tao_da_thuc_zernike(N, indices)
%TAO_DA_THUC_ZERNIKE - Tạo các đa thức Zernike trên một lưới vuông.
%
% Hàm này tạo ra một tập các đa thức Zernike chuẩn hóa (orthonormal)
% trên một đĩa tròn đơn vị, được lấy mẫu trên một lưới vuông kích thước N x N.
%
% Cú pháp:
%   [Z, n, m] = tao_da_thuc_zernike(N, indices)
%
% ĐẦU VÀO:
%   N       (integer): Kích thước của lưới vuông đầu ra (N x N).
%   indices (vector):  Vector chứa các chỉ số Noll (j) của các đa thức
%                      cần tạo (ví dụ: [3, 4, 5] cho defocus và astigmatism).
%
% ĐẦU RA:
%   Z       (3D matrix): Ma trận 3D kích thước N x N x numel(indices),
%                        trong đó mỗi lớp Z(:,:,k) là một đa thức Zernike.
%   n_modes (vector):    Vector chứa bậc xuyên tâm 'n' tương ứng với mỗi chỉ số.
%   m_modes (vector):    Vector chứa bậc phương vị 'm' tương ứng với mỗi chỉ số.

% --- 1. Tạo lưới tọa độ chuẩn hóa ---
[x, y] = meshgrid(linspace(-1, 1, N));

% Chuyển sang tọa độ cực
rho = sqrt(x.^2 + y.^2);
theta = atan2(y, x);

% --- 2. Khởi tạo các biến đầu ra ---
num_modes = numel(indices);
Z = zeros(N, N, num_modes);
n_modes = zeros(1, num_modes);
m_modes = zeros(1, num_modes);

% --- 3. Vòng lặp để tạo từng đa thức Zernike ---
for k = 1:num_modes
    j = indices(k);

    % --- Chuyển đổi chỉ số Noll (j) sang (n, m) CHÍNH XÁC ---
    [n, m] = noll_to_nm(j);
    
    n_modes(k) = n;
    m_modes(k) = m;

    % --- Tính đa thức xuyên tâm R_n^|m|(rho) ---
    R = radial_polynomial(n, abs(m), rho);

    % --- Tổ hợp thành đa thức Zernike Z_n^m ---
    if m > 0
        Z_temp = R .* cos(m * theta);
    elseif m < 0
        Z_temp = R .* sin(abs(m) * theta);
    else % m == 0
        Z_temp = R;
    end

    % --- Chuẩn hóa (để thành orthonormal) ---
    if m == 0
        norm_factor = sqrt(n + 1);
    else
        norm_factor = sqrt(2 * (n + 1));
    end
    Z_temp = norm_factor * Z_temp;

    % Áp dụng mặt nạ đĩa tròn (chỉ có giá trị khi rho <= 1)
    Z_temp(rho > 1) = 0;

    % Lưu kết quả
    Z(:, :, k) = Z_temp;
end
end

% =========================================================================
% HÀM PHỤ TRỢ: Chuyển đổi chỉ số Noll sang (n,m)
% =========================================================================
function [n, m] = noll_to_nm(j)
% Chuyển đổi chỉ số Noll j sang (n, m) theo chuẩn OSA/ANSI
% Tham khảo: Noll, R. J. (1976). "Zernike polynomials and atmospheric turbulence"

% Tìm bậc xuyên tâm n
n = 0;
while (n+1)*(n+2)/2 < j
    n = n + 1;
end

% Tính m dựa trên vị trí trong hàng thứ n
p = j - n*(n+1)/2;  % Vị trí trong hàng (từ 1)

% Tính m theo quy tắc Noll
if mod(p, 2) == 1  % p lẻ
    m = (p - 1) / 2;
    if mod((n+m)/2, 2) == 1
        m = -m;
    end
else  % p chẵn
    m = p / 2;
    if mod((n+m)/2, 2) == 0
        m = -m;
    end
end

% Đảm bảo m có cùng tính chẵn lẻ với n
if mod(n-abs(m), 2) ~= 0
    error('Lỗi: n-|m| phải là số chẵn cho chỉ số Noll j=%d', j);
end
end

% =========================================================================
% HÀM PHỤ TRỢ: Tính đa thức xuyên tâm
% =========================================================================
function R = radial_polynomial(n, m, rho)
% Tính đa thức xuyên tâm R_n^m(rho)
% n: bậc xuyên tâm
% m: |m| (giá trị tuyệt đối của bậc phương vị)
% rho: ma trận bán kính

R = zeros(size(rho));

% Kiểm tra điều kiện n-m phải chẵn
if mod(n - m, 2) ~= 0
    return; % R = 0 nếu n-m là lẻ
end

% Tính theo công thức tổng
for s = 0:((n - m) / 2)
    coeff = (-1)^s * factorial(n - s) / ...
            (factorial(s) * factorial((n + m)/2 - s) * factorial((n - m)/2 - s));
    R = R + coeff * rho.^(n - 2*s);
end
end

% =========================================================================
% HÀM KIỂM TRA: Test tính đúng đắn của các đa thức Zernike
% =========================================================================
function test_zernike_polynomials()
% Kiểm tra một số đa thức Zernike cơ bản

fprintf('Testing Zernike polynomials...\n');

% Test case 1: j=1 (piston)
[Z1, n1, m1] = tao_da_thuc_zernike(64, 1);
fprintf('j=1: n=%d, m=%d (Expected: n=0, m=0)\n', n1, m1);

% Test case 2: j=2,3 (tip, tilt)
[Z23, n23, m23] = tao_da_thuc_zernike(64, [2, 3]);
fprintf('j=2: n=%d, m=%d (Expected: n=1, m=1)\n', n23(1), m23(1));
fprintf('j=3: n=%d, m=%d (Expected: n=1, m=-1)\n', n23(2), m23(2));

% Test case 3: j=4 (defocus)
[Z4, n4, m4] = tao_da_thuc_zernike(64, 4);
fprintf('j=4: n=%d, m=%d (Expected: n=2, m=0)\n', n4, m4);

% Test case 4: j=5,6 (astigmatism)
[Z56, n56, m56] = tao_da_thuc_zernike(64, [5, 6]);
fprintf('j=5: n=%d, m=%d (Expected: n=2, m=2)\n', n56(1), m56(1));
fprintf('j=6: n=%d, m=%d (Expected: n=2, m=-2)\n', n56(2), m56(2));

fprintf('Test completed.\n');
end

% =========================================================================
% HÀM TẠO BẢNG THAM CHIẾU
% =========================================================================
function print_noll_reference_table()
% In bảng tham chiếu các chỉ số Noll phổ biến

fprintf('\n=== BẢNG THAM CHIẾU CÁC ĐA THỨC ZERNIKE (NOLL INDICES) ===\n');
fprintf('j\tn\tm\tTên gọi\n');
fprintf('--\t--\t--\t--------\n');
fprintf('1\t0\t0\tPiston\n');
fprintf('2\t1\t1\tTip (tilt về X)\n');
fprintf('3\t1\t-1\tTilt (tilt về Y)\n');
fprintf('4\t2\t0\tDefocus\n');
fprintf('5\t2\t2\tAstigmatism (0°)\n');
fprintf('6\t2\t-2\tAstigmatism (45°)\n');
fprintf('7\t3\t1\tComa (Y)\n');
fprintf('8\t3\t-1\tComa (X)\n');
fprintf('9\t3\t3\tTrefoil (Y)\n');
fprintf('10\t3\t-3\tTrefoil (X)\n');
fprintf('11\t4\t0\tSpherical aberration\n');
fprintf('12\t4\t2\tSecondary astigmatism (0°)\n');
fprintf('13\t4\t-2\tSecondary astigmatism (45°)\n');
fprintf('14\t4\t4\tQuadrafoil (0°)\n');
fprintf('15\t4\t-4\tQuadrafoil (45°)\n');
fprintf('=====================================\n\n');
end