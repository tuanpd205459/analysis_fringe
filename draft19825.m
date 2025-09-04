clear;clc;close all;
%% 1. Thiết lập mặt phẳng lấy mẫu
M = 512; % Chiều cao
N = 512; % Chiều rộng
p = 6; 
f0 = 1 /p; % Tần số sóng mang
a = 100; % Biên độ pha (biên độ nền)
b = 100; % Biên độ của sóng hình sin
x=-M/2:M/2-1;  
y=-N/2:N/2-1;
[X,Y] = meshgrid(x,y); % Tạo lưới tọa độ 2D

%% 2. Tạo đối tượng (dạng hình hộp chữ nhật)
rect_width = 120; % Chiều rộng hình hộp
rect_height = 120; % Chiều cao hình hộp
object = rectpuls(X,rect_width).*rectpuls(Y,rect_height); % Tạo đối tượng hình hộp chữ nhật
Object = object +1 ;
% object = object*5;
% figure,imagesc(object),axis equal,axis off;
% object = 2*peaks(512); % Tùy chọn, tạo đối tượng 'peaks'
figure,imagesc(object),axis equal,axis off; % Hiển thị đối tượng

%% 3. Tạo ảnh vân giao thoa (interferogram)
snr = 25; % Tỷ lệ tín hiệu trên nhiễu (Signal-to-Noise Ratio)
I_r = a + b*cos(2* pi* f0* X); % Ảnh vân tham chiếu (không có đối tượng)
I_o = a + b*cos(2* pi* f0* X + object); % Ảnh vân mục tiêu (có đối tượng)

%% 4. Thêm nhiễu Gaussian trắng vào ảnh vân mục tiêu
I_O_Gauss = awgn(I_o,snr,'measured','dB');% Thêm nhiễu Gaussian trắng
figure;imshow(I_O_Gauss,[]);
title('Ảnh vân giao thoa có nhiễu');

%% 5. Phân tích phổ
I_O = I_O_Gauss(:,:,1);
I_R = I_r(:,:,1);
[value_x,value_y] = size(I_O);
I_O=im2double(I_O);
I_R=im2double(I_R);
I_fft_O=fftshift(fft2(I_O)); % Biến đổi Fourier 2D của ảnh vân mục tiêu
I_fft_R=fftshift(fft2(I_R)); % Biến đổi Fourier 2D của ảnh vân tham chiếu
% figure,imshow(log(1+abs(I_fft_O)),[]); title("Phổ vân mục tiêu"); % Hiển thị phổ của ảnh vân mục tiêu
% figure,imshow(log(1+abs(I_fft_R)),[]); title("Phổ vân tham chiếu");% Hiển thị phổ của ảnh vân tham chiếu
% figure,plot(abs(I_fft_O(value_x/2+1,:))); title("plot vân mục tiêu");
% figure,plot(abs(I_fft_R(value_x/2+1,:)));title("plot vân tham chiếu");
[maxvalue,zuobiao]=max(abs(I_fft_O(value_x/2+1,1:value_y/2-10))); % Tìm vị trí đỉnh phổ

%% 6. Tạo bộ lọc Gauss
% Bộ lọc hình tròn (commented out)
% W = 20;
% Z = value_x/2-zuobiao;
% circle=(X-Z).^2+Y.^2;   % Tính khoảng cách từ mỗi điểm đến tâm
% H=ones(M,N); 
% H(find(circle >= W*W))=0;  % Đặt giá trị ngoài bán kính về 0
% figure,mesh(H);

% Bộ lọc Gaussian
W = 23; % Bán kính của bộ lọc
m=value_x/2;
n = zuobiao;
H = zeros(value_x,value_y);
for i=1:value_x
    for j=1:value_y
        D=sqrt((i-m)^2+(j-n)^2); % Khoảng cách từ điểm (i,j) đến tâm bộ lọc
        H(i,j)=exp(-1/2*D^2/W^2); % Giá trị của bộ lọc Gauss
    end
end

%% 7. Lọc dải thông và tách tần số cơ bản
jipin_O = I_fft_O.*H; % Lọc ảnh vân mục tiêu
jipin_R = I_fft_R.*H; % Lọc ảnh vân tham chiếu
jipin_ifft_O=ifft2(ifftshift(jipin_O)); % Biến đổi Fourier ngược
jipin_ifft_R=ifft2(ifftshift(jipin_R));
jipin_ifft_R= -conj(jipin_ifft_R); % Lấy liên hợp phức và đảo dấu
jipin_ifft = jipin_ifft_R.*jipin_ifft_O; % Nhân phức để lấy pha

% 1. Hiển thị phổ gốc
figure;
subplot(2,3,1); imagesc(log(1+abs(I_fft_O))); colormap gray; axis image;
title('Phổ FFT của O');

subplot(2,3,2); imagesc(log(1+abs(I_fft_R))); colormap gray; axis image;
title('Phổ FFT của R');

% 2. Hiển thị bộ lọc Gaussian H
subplot(2,3,3); imagesc(H); colormap jet; colorbar; axis image;
title('Bộ lọc Gaussian H');

% 3. Phổ sau khi lọc
subplot(2,3,4); imagesc(log(1+abs(jipin_O))); colormap gray; axis image;
title('Phổ O sau khi lọc');

subplot(2,3,5); imagesc(log(1+abs(jipin_R))); colormap gray; axis image;
title('Phổ R sau khi lọc');

% 4. Ảnh sau IFFT
subplot(2,3,6); imagesc(real(jipin_ifft_O)); colormap gray; axis image;
title('Ảnh O sau ifft');


%% 8. Mở gói pha (unwrapping)
% unph = -atan2(imag(jipin_ifft),real(jipin_ifft)); % Pha gói (wrapped phase)
unph = -angle(jipin_ifft);

figure,imagesc(unph),axis equal,axis off; % Hiển thị pha gói
title('Pha đã được gói');
figure, surf(unph,"EdgeColor","none"), title("Anh wrapped ");
colorbar;


% 1) Pha & biên độ
C = jipin_ifft;
amp = abs(C);
phi_wrapped = angle(C);

figure; 
subplot(2,2,1); imagesc(amp); axis image; colorbar; title('Amplitude');
subplot(2,2,2); imagesc(phi_wrapped); axis image; colorbar; title('Wrapped phase');

% 2) Histogram pha để thấy phân bố
subplot(2,2,3); histogram(phi_wrapped(:),200); title('Histogram of wrapped phase');

% 3) Kiểm tra vùng có amplitude nhỏ
subplot(2,2,4);
th = 0.12 * max(amp(:)); % thử thay 0.12 thành 0.05/0.2 để so sánh
imshow(amp > th); title(['Mask: amp > ' num2str(th)]);

%% 9. Thuật toán Least-Square (Bình phương tối thiểu)
dx = psf2otf_test([-1,1;0,0],[value_x,value_y]); % Toán tử vi phân x
dy = psf2otf_test([-1,0;1,0],[value_x,value_y]); % Toán tử vi phân y
DTD = abs(dx).^2 + abs(dy).^2;
dadx = real(ifft2(fft2(unph).*dx)); % Gradient x của pha gói
dady = real(ifft2(fft2(unph).*dy)); % Gradient y của pha gói
dadx_G = dadx-pi*round(dadx/pi); % Loại bỏ bước nhảy 2*pi từ gradient
dady_G = dady-pi*round(dady/pi);
ph_L2 = real(ifft2((fft2(dadx_G).*conj(dx)+fft2(dady_G).*conj(dy))./(DTD + eps))); % Phục hồi pha
nmse_L2 = NMSE(object,im2gray(ph_L2)); % Tính NMSE (Sai số bình phương trung bình chuẩn hóa)
disp(nmse_L2);

%% 10. Thuật toán Total Variation (Tổng biến thiên)
ph_TV = unph; % Khởi tạo pha TV
% Thông số khởi tạo TV
lambda_L1 = 0.001;
lambda0_L1 = 2*lambda_L1;
lambda_max_L1 = 1e5;
while lambda0_L1 < lambda_max_L1
    gx = real(ifft2(fft2(ph_TV).*dx));
    gy = real(ifft2(fft2(ph_TV).*dy));
    gx_L = sign(gx) .* max(abs(gx) - lambda_L1/lambda0_L1,0);
    gy_L = sign(gy) .* max(abs(gy) - lambda_L1/lambda0_L1,0);
    Gx = fft2(gx_L).*conj(dx);
    Gy = fft2(gy_L).*conj(dy);
    fenzi = fft2(dadx_G).*conj(dx)+fft2(dady_G).*conj(dy)  + lambda0_L1*(Gx+Gy);
    fenmu = (1 + lambda0_L1)*DTD+eps;
    ph_TV = real(ifft2(fenzi./fenmu)); 
    lambda0_L1 = lambda0_L1 * 2;
end
nmse_TV = NMSE(object,im2gray(ph_TV));
disp(nmse_TV);

%% 11. Thuật toán Dark Sparse Prior
ph_SP= unph; % Khởi tạo pha SP
% Thông số L0
lambda = 0.00001;
lambda_max = 1e5;
lambda_L0 = 2*lambda;
while lambda_L0 < lambda_max
    Q = ph_SP.*(abs(ph_SP).^2 > lambda/lambda_L0);
    % Thông số L1
    beta = 0.001;
    beta_L1 = 2*beta;
    beta_max = 1e5;
    while beta_L1 < beta_max
        gx = real(ifft2(fft2(ph_SP).*dx));
        gy = real(ifft2(fft2(ph_SP).*dy));
        gx_L1 = sign(gx) .* max(abs(gx) - beta/beta_L1,0);
        gy_L1 = sign(gy) .* max(abs(gy) - beta/beta_L1,0);
        Gx = fft2(gx_L1).*conj(dx);
        Gy = fft2(gy_L1).*conj(dy);
        
        fenzi = fft2(dadx_G).*conj(dx)+fft2(dady_G).*conj(dy) + lambda_L0*fft2(Q) + beta_L1*(Gx+Gy);
        fenmu = lambda_L0+(1+  beta_L1)*DTD;
        ph_SP = real(ifft2(fenzi./(fenmu+eps)));
        beta_L1 = beta_L1*2;
    end
    lambda_L0 = lambda_L0 * 2;
end
nmse_SP = NMSE(object,im2gray(ph_SP));
disp(nmse_SP);

%% 12. Hiển thị kết quả pha
figure,imagesc(ph_L2);axis off;axis equal;
title('Pha mở gói bằng LS');
figure,imagesc(ph_TV);axis off;axis equal;
title('Pha mở gói bằng TV');
figure,imagesc(ph_SP);axis off;axis equal;
title('Pha mở gói bằng SP');
figure,surf(ph_L2,"EdgeColor","none");
title('Pha mở gói bằng LS');
figure,surf(ph_TV, "EdgeColor","none");
title('Pha mở gói bằng TV');
figure,surf(ph_SP, "EdgeColor","none");
title('Pha mở gói bằng SP');


% figure,imagesc(object-ph_L2);axis off;axis equal; % Sai số LS
% figure,imagesc(object-ph_TV);axis off;axis equal; % Sai số TV
% figure,imagesc(object-ph_SP);axis off;axis equal; % Sai số SP

%% 13. Biểu đồ mặt cắt ngang
x = 1:1:value_x; 
y = ph_L2(value_x/2,:); % Lấy mặt cắt giữa của pha LS
figure
plot(x,y,'r','linewidth',1.4);
hold on
y = ph_TV(value_x/2,:); % Lấy mặt cắt giữa của pha TV
plot(x,y,'g','linewidth',1.4);
hold on
y = ph_SP(value_x/2,:); % Lấy mặt cắt giữa của pha SP
plot(x,y,'b','linewidth',1.4);
hold on
y = object(value_x/2,:); % Lấy mặt cắt giữa của đối tượng gốc
plot(x,y,'k','linewidth',1.4);
ylabel('Pha (rad)');
xlabel('Số điểm ảnh');
xlim([0 512])
ylim([-0.1 1.2])
lgd = legend({'LS','TV','SPUP','GT'},'Location','northeast');
set(lgd,'unit','centimeters','FontSize',14);

% NMSE.m
function nmse = NMSE(original_image, reconstructed_image)
    % Loại bỏ offset và chuẩn hóa các giá trị pha trước khi tính toán
    original_image = original_image - mean(original_image(:));
    reconstructed_image = reconstructed_image - mean(reconstructed_image(:));

    % Tính toán sai số bình phương trung bình chuẩn hóa (NMSE)
    mse = mean((original_image(:) - reconstructed_image(:)).^2);
    original_energy = mean(original_image(:).^2);
    nmse = mse / original_energy;
end 