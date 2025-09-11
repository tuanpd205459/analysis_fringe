clc; clear; close all;

%% --- Demo data ---
N = 256;
[x, y] = meshgrid(linspace(-2,2,N));

% Ground truth phase (paraboloid)
phi_true = x.^2 + y.^2;

% Wrapped phase (từ ground truth)
phi_wrapped = angle(exp(1i*phi_true)) + 1*randn(size(phi_true));

% Estimated phase (ground truth + nhiễu lớn hơn)
phi_est = phi_true + 2.0*randn(size(phi_true));  % cố ý sai nhiều

%% --- Phase consistency check ---
error_map = angle(exp(1i*(phi_wrapped - phi_est)));   % [-pi, pi]
abs_error = abs(error_map);
reliability_map = 1 - abs_error/pi;   % [0,1]

%% --- Hybrid unwrapping ---
[Ny, Nx] = size(phi_wrapped);
unwrapped = nan(Ny, Nx);     
visited   = false(Ny, Nx);   

% Threshold cho consistency check
T = 0.5*pi;  % nếu |wrapped - est| < T thì cho phép dùng estimate

% Seed = pixel reliability cao nhất
[~, idx] = max(reliability_map(:));
[y0, x0] = ind2sub(size(phi_wrapped), idx);

unwrapped(y0,x0) = phi_wrapped(y0,x0);
visited(y0,x0) = true;

% Priority queue: [ -reliability , y , x , phi ]
pq = [-reliability_map(y0,x0), y0, x0, unwrapped(y0,x0)];

% 4-connected neighborhood
nbrs = [0 1; 0 -1; 1 0; -1 0];

while ~isempty(pq)
    % Lấy pixel reliability cao nhất
    [~, k] = min(pq(:,1));
    current = pq(k,:); 
    pq(k,:) = [];
    
    y = current(2);
    x = current(3);
    phi_ref = current(4);
    
    for nb = 1:4
        yy = y + nbrs(nb,1);
        xx = x + nbrs(nb,2);
        if yy<1 || yy>Ny || xx<1 || xx>Nx, continue; end
        if visited(yy,xx), continue; end
        
        % unwrap bằng propagation
        phi_nb_wrapped = phi_wrapped(yy,xx);
        phi_nb = phi_ref + angle(exp(1i*(phi_nb_wrapped - phi_ref)));
        
        % Nếu reliability thấp, tham khảo estimated phase
        if reliability_map(yy,xx) < 0.5
            diff_est = angle(exp(1i*(phi_nb_wrapped - phi_est(yy,xx))));
            if abs(diff_est) < T
                % Điều chỉnh theo estimated phase
                phi_nb = phi_est(yy,xx);
            end
        end
        
        unwrapped(yy,xx) = phi_nb;
        visited(yy,xx) = true;
        
        % push vào priority queue
        pq(end+1,:) = [-reliability_map(yy,xx), yy, xx, phi_nb];
    end
end

%% --- Visualization ---
figure;
subplot(2,2,1);
imagesc(phi_wrapped); axis image; colorbar;
title('Wrapped Phase');

subplot(2,2,2);
imagesc(phi_est); axis image; colorbar;
title('Estimated Phase');

subplot(2,2,3);
imagesc(unwrapped); axis image; colorbar;
title('Hybrid Unwrapped Phase');

subplot(2,2,4);
imagesc(phi_true); axis image; colorbar;
title('Ground Truth Phase');

colormap jet;
