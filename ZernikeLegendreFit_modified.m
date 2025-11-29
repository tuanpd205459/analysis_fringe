% Function for fitting a surface to Zernike polynomials
%
% Usage:
%     [output_coeff, z_recon_map] = ZernikeLegendreFit(z_map, index_type, coeff_max,
%                                                      J, K, center_x, center_y)
% Ver 1.2 update: Support rectangular input matrix z_map (raw and column 
%                 number does not have to be equal)
% Ver 1.1 update: Support Fringe Zernike index. Please see Discription for
%                 details.
%
% Description:
%     This function takes as input a freeform surface and fits it to
%     Zernike polynomials, returning the coefficients of the fit as well
%     as the reconstructed map.
%
% Inputs:
%     z_map - The sag table of the freeform surface.
%     index_type - A string presenting the Zernike type. Currently support
%                  "2indices" (default) and "fringe".
%     coeff_max - The highest fitting order.
%                 1) if index_type == "2indices", this should be a 1x2 cell
%                 where ~{1} = M, ~{2} = N.
%                 2) if index_tyoe == "fringe", this shoule an integer J.
%     J - The sampling number on azimuthal direction, default: 2*M+1
%     K - The sampling number on radial direction, default: 2*(N+1)
%     center_x, center_y - The center of the freeform surface, default:
%                          will be calculated by FindCenter function.
%
% Outputs:
%     output_coeff - The fitted coefficients,
%                    1) if index_type == "2indices", this should be a 1x2
%                    cell, where ~{1} = amn, ~{2} = bmn.
%                    2) if index_type == "fringe", this should be a 1x1
%                    cell, where ~{1} = cj.
%     z_recon_map - The reconstructed surface
%
%
%
% Author: Yiwen Fan (yfan22@ur.rochester.edu)
% Date: 07/16/2023
% License: MIT License
%
% Acknowledgments:
%     This work was inspired by the research of Greg Forbes and was
%     funded by 1. University of Rochester, 2. Center for Freeform Optics.
%
%     This work was partially contributed by Ilhan Kaya in 2010, who wrote
%     function jacobiZernike to calculate the value of Jacobi polynomials
%     recursively. This function was later modified by Yiwen Fan as
%     function jacobiZernike_table.
function [output_coeff, z_recon_map] = ZernikeLegendreFit(z_map, index_type, coeff_max, J, K, center_j, center_i)
if nargin <= 2
    error("fit_Zernike_quadrature requires at least 3 input: z_map, index_type, coeff_max.");
end
valid_index_type = string({"2indices", "fringe"});
index_type = lower(index_type);
if ~ischar(index_type) && ~isstring(index_type)
    error("Notation type must be a string or charagter array");
end
if ~ismember(index_type, valid_index_type)
    error("Invalid index type. Valid options are: %s", strjoin(valid_notations, ', '));
end
if index_type == "2indices"
    try
        m_max = coeff_max(1);
        n_max = coeff_max(2);
    catch ME
        fprintf('Error occurred: %s\n', ME.message);
    end
elseif index_type == "fringe"
    if ~isscalar(coeff_max)
        error('m_max must be a single number, not an array');
    end
    j = coeff_max;
    n_max = fix(sqrt(j) - 1);
    m_max = max(fix(1+sqrt(j)) , -fix(-1-sqrt(j+1)));
    flag_amn = zeros(m_max + 1, n_max + 1);
    flag_bmn = zeros(m_max + 1, n_max + 1);
    for j = 1:coeff_max
        [n, m] = fringe22index(j);
        if m >= 0
            flag_amn(m+1, n+1) = j;
        elseif m < 0
            flag_bmn(-m+1, n+1) = j;
        end
    end
end
if nargin <= 5
    [center_j, center_i] = FindCenter(z_map);
end
if nargin <= 4
    K = 2 * (n_max + 1);
end
if nargin <= 3
    J = 2 * m_max + 1;
end
r = 1; intp_type = "cubic";
[y_pix, x_pix] = size(z_map); 
r_x_pix = (x_pix-1)/2; r_y_pix = (y_pix-1)/2;
x = linspace(-r, r, x_pix); y = linspace(-r, r, y_pix);
[x_map, y_map] = meshgrid(x, y);
[i_map, j_map] = meshgrid((1:x_pix), (1:y_pix));
[theta_map, r_map] = cart2pol(x_map, y_map);
del = pi/J;
theta = (0:(2*J))*del;
theta = theta(1:end-1);
% Jacobi Matrix
Jm = jacobiMatrix(0, K);  % Legendre matrix
[V, e] = eig(Jm);
u_zeropoints = diag(e)';  % zeropoints
w_m = 2*(V(1,:).^2);      % weights
u2 = u_zeropoints; u = sqrt(u2);
[x_temp, y_temp] = pol2cart(theta, u');
x_temp = x_temp * r_x_pix; y_temp = y_temp * r_y_pix;
x_temp = x_temp + center_i; y_temp = y_temp + center_j;
Stemp_m = interp2(i_map, j_map, z_map, x_temp, y_temp, intp_type);
% Azimuthal fit
A_qua = zeros(m_max+1, K); B_qua = zeros(m_max+1, K);
for k = 1:K
    Stemp = Stemp_m(k,:);
    FFT_S = fft(Stemp)/size(theta,2);
    re = real(FFT_S); im = imag(FFT_S);
    A_qua(:,k) = 2*re(1:m_max+1);
    B_qua(:,k) = -2 * im(1:m_max+1);
end
A_qua(1,:) = A_qua(1,:)/2;
B_qua(1,:) = B_qua(1,:)/2;
% Radial fit
amn = zeros(m_max+1, n_max+1); bmn = zeros(m_max+1, n_max+1);
for m = 0:m_max
    Pmns = jacobiZernike_table(n_max+1, m, u2')' .* (u.^m);
    Pmns_weighted = Pmns .* w_m;
    for n = 0:n_max
        Pmn = Pmns_weighted(n+1, :);
        if index_type == "2indices" || (index_type == "fringe" && flag_amn(m+1, n+1) > 0)
            amn(m+1, n+1) = dot(A_qua(m+1, :), Pmn) *(2*n+m+1)/2;
        end
        if index_type == "2indices" || (index_type == "fringe" && flag_bmn(m+1, n+1) > 0)
            bmn(m+1, n+1) = dot(B_qua(m+1, :), Pmn) *(2*n+m+1)/2;
        end
    end
end

%% --- MODIFICATION START: REMOVE TILT, DEFOCUS, SPHERICAL ---
if index_type == "fringe"
    % Fringe indices to remove:
    % 2: Tilt X
    % 3: Tilt Y
    % 4: Defocus
    % 9: Primary Spherical Aberration
    remove_indices = [2, 3, 4, 9];
    
    for idx = remove_indices
        if idx <= coeff_max
            [n_rem, m_rem] = fringe22index(idx);
            
            % amn/bmn are accessed via (m+1, n+1) because MATLAB is 1-based
            % and m, n can be 0.
            if m_rem >= 0
                % Remove from amn (Cosine terms)
                if (m_rem + 1 <= size(amn, 1)) && (n_rem + 1 <= size(amn, 2))
                    amn(m_rem+1, n_rem+1) = 0;
                end
            elseif m_rem < 0
                % Remove from bmn (Sine terms)
                if (-m_rem + 1 <= size(bmn, 1)) && (n_rem + 1 <= size(bmn, 2))
                    bmn(-m_rem+1, n_rem+1) = 0;
                end
            end
        end
    end
end
%% --- MODIFICATION END ---





% Reconstructrion
[i_mesh, j_mesh] = meshgrid((1:x_pix),(1:y_pix));
i_mesh = (i_mesh - center_i)/r_x_pix; j_mesh = (j_mesh - center_j)/r_y_pix;
[theta,rho] = cart2pol(i_mesh, j_mesh); % Converting to polar coords
u_map = rho(:);
theta = theta(:);
u2_map = u_map.^2;
u2_map(u2_map>0.99) = NaN;
z_recon_map = zeros(y_pix, x_pix);
for m = 0:m_max
    idx = n_max;
    Pmns_map = jacobiZernike_table(idx+1, m, reshape(u2_map, 1,[])');
    Pmns_map = Pmns_map' .* (reshape(u_map, 1,[]).^m);
    for n = 0:idx
        % Pmn_map = reshape(Pmns_map(n+1, :), x_pix, y_pix);
        Pmn_map = reshape(Pmns_map(n+1, :), y_pix, x_pix);
        a = amn(m+1, n+1); b = bmn(m+1, n+1);
        a_map = a * cos(m * theta_map) .* Pmn_map;
        b_map = b * sin(m * theta_map) .* Pmn_map;
        z_recon_map = z_recon_map + a_map + b_map;
    end
end
z_recon_map(u_map>0.99) = NaN;
if lower(index_type) == "2indices"
    output_coeff = cell(1,2);
    output_coeff{1} = amn; output_coeff{2} = bmn;
elseif lower(index_type) == "fringe"
    coeff_j = zeros(1, coeff_max);
    for i = 1:m_max + 1
        for j = 1:n_max + 1
            if flag_amn(i,j) > 0
                coeff_j(flag_amn(i,j)) = amn(i, j);
            end
            if flag_bmn(i,j) > 0
                coeff_j(flag_bmn(i,j)) = bmn(i, j);
            end
        end
    end
    output_coeff = cell(1,1);
    output_coeff{1} = coeff_j;
end
end
%% FindCenter
function [avg_i, avg_j] = FindCenter(Z)
i_index = 1:size(Z,1); j_index = 1:size(Z,2);
[valid_i, valid_j] = find(~isnan(Z));
avg_i = mean(i_index(valid_i),'all');
avg_j = mean(j_index(valid_j),'all');
end
%% jacobiMatrix
function T_mtrx = jacobiMatrix(m, n_max)
% Returns a n_max*n_max matrix for a constant m term (jacobi beta)
T_mtrx = zeros(n_max, n_max);
an_ = -(m+1); bn_ = m+2;
for n = 1:n_max-1
    s = m+2*n;
    an = -(s+1)*((s-n).^2+n.^2+s)./(n+1)./(s-n+1)./s;
    bn = (s+2)*(s+1)./(n+1)./(s-n+1);
    cn = (s+2)*(s-n)*n./(n+1)./(s-n+1)./s;
    % three terms of Jacobi matrix, them perform a diagonal similarity
    % transorfation
    T_mtrx(n, n) = -an_/bn_;
    T_mtrx(n, n+1) = sqrt(cn/(bn_*bn));
    T_mtrx(n+1, n) = T_mtrx(n, n+1);
    an_ = an; bn_ = bn;
end
T_mtrx(n_max, n_max) = -an_/bn_;
end
%% jacobiZernike_table
function JZ=jacobiZernike_table(k,m,xe)
% k is the degree of orthogonal jacobi polynomial
% k=nf=(nw-m)/2 (relation between wolf - forbes)
% m is the azimuthal order
% xe grid points in radial coordinate.
% ilhan kaya 11/15/2010
% Yiwen modified these to export a whole table. 04/25/2023
% name change from jacobiZernike to jacobiZernike_table
JZ = zeros(size(xe,1),k+1);
JZ(:,1)=ones(size(xe,1),1); % n=0
JZ(:,2)=(m+2)*xe-(m+1);     % n=1 --> an = -(m+1); bn = m+2; cn = 0;
%recurrence Forbes Opt. Exps. Vol. 18 No. 12 June 2010
% P_(n+1)(x)= (an+bn*x)*P_n(x)-cn*P_(n-1)(x)
for n = 1:k-1
    s = m+2*n;
    an = -(s+1)*((s-n).^2+n.^2+s)./(n+1)./(s-n+1)./s;
    bn = (s+2)*(s+1)./(n+1)./(s-n+1);
    cn = (s+2)*(s-n)*n./(n+1)./(s-n+1)./s;
    JZ(:,n+2)=(an+bn*xe).*JZ(:,n+1)-cn*JZ(:,n);
end
end
%% Fringe index to 2 indices
function [n_, m_] = fringe22index(j)
for n = 0:1:j*2+2
    for m = -n:1:n+1
        if m < 0
            sgn_m = -1;
        else
            sgn_m = 1;
        end
        temp = (1+(n+abs(m))/2)^2 - 2*abs(m) + (1-sgn_m)/2;
        if j == temp
            m_ = m;
            n_ = (n-abs(m_))/2;
            return
        end
    end
end
end