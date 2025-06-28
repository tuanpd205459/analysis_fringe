clc, clear, close all;


    % Giả lập dữ liệu
    [X, Y] = meshgrid(1:200, 1:200);
    phi_true = 0.03 * X + 0.02 * Y;
    varphi_w = angle(exp(1i * phi_true));                % Pha bọc
    phi_min = phi_true + 1 + 0.2* randn(size(phi_true));   % Pha tham chiếu

    % -------- Bước 4: Giải pha bọc --------
    K = ceil((phi_min - varphi_w) / (2*pi));
    Phi = varphi_w + 2*pi*K;

    % -------- Bước 5: Mở khóa pha hai chiều --------
    Phi_corr = Phi;

    % Mở khoá theo hàng (X direction)
    [rows, cols] = size(Phi_corr);
    for r = 1:rows
        k = 0;
        for c = 2:cols
            delta = Phi_corr(r, c) - Phi_corr(r, c-1);
            if delta > pi
                k = k - 1;
            elseif delta < -pi
                k = k + 1;
            end
            Phi_corr(r, c) = Phi_corr(r, c) + 2*pi*k;
        end
    end

    % Mở khoá theo cột (Y direction)
    for c = 1:cols
        k = 0;
        for r = 2:rows
            delta = Phi_corr(r, c) - Phi_corr(r-1, c);
            if delta > pi
                k = k - 1;
            elseif delta < -pi
                k = k + 1;
            end
            Phi_corr(r, c) = Phi_corr(r, c) + 2*pi*k;
        end
    end

    % -------- Hiển thị kết quả 3D --------
    figure('Name','Unwrapped Phase (Bidirectional)','NumberTitle','off');
    subplot(1,3,1); mesh(X, Y, varphi_w); title('Wrapped Phase'); xlabel('X'); ylabel('Y'); zlabel('\varphi_w'); shading interp; colormap jet;
    subplot(1,3,2); mesh(X, Y, phi_true); title('True Phase'); xlabel('X'); ylabel('Y'); zlabel('\phi_{true}'); shading interp; colormap jet;
    subplot(1,3,3); mesh(X, Y, Phi_corr); title('Unwrapped Phase (2D Corrected)'); xlabel('X'); ylabel('Y'); zlabel('\Phi'); shading interp; colormap jet;

