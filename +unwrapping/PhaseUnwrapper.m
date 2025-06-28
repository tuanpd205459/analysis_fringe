classdef PhaseUnwrapper
    methods(Static)
        %% Unwrap LS FD DCT #1
        function [phase_unwrap]=unwrap_LS_FD_DCT(phase_wrap)
            psi=phase_wrap;
            edx = [zeros([size(psi,1),1]), wrapToPi(diff(psi, 1, 2)), zeros([size(psi,1),1])];
            edy = [zeros([1,size(psi,2)]); wrapToPi(diff(psi, 1, 1)); zeros([1,size(psi,2)])];
            rho = diff(edx, 1, 2) + diff(edy, 1, 1);
            phase_unwrap = solvePoisson(rho); % get the result by solving the poisson equation
        end

        %% Unwrap LS FD DCT iter #2
        function [phase_unwrap,N]= unwrap_LS_FD_DCT_iter(phase_wrap)
            phi1 = unwrap_LS_FD_DCT(phase_wrap);
            phi1=phi1+mean2(phase_wrap)-mean2(phi1); %adjust piston
            K1=round((phi1-phase_wrap)/2/pi);  %calculate integer K
            phase_unwrap=phase_wrap+2*K1*pi;
            residue=wrapToPi(phase_unwrap-phi1);
            phi1=phi1+unwrap_LS_FD_DCT(residue);
            phi1=phi1+mean2(phase_wrap)-mean2(phi1); %adjust piston
            K2=round((phi1-phase_wrap)/2/pi);  %calculate integer K
            phase_unwrap=phase_wrap+2*K2*pi;
            residue=wrapToPi(phase_unwrap-phi1);
            rms1=sqrt(mean2((residue).^2));
            N=0;
            delta_rms=rms1;
            while sum(sum(abs(K2-K1)))>0 && delta_rms>10^-5
                K1=K2;
                phic=unwrap_LS_FD_DCT(residue);
                phi1=phi1+phic;
                phi1=phi1+mean2(phase_wrap)-mean2(phi1); %adjust piston
                K2=round((phi1-phase_wrap)/2/pi);  %calculate integer K
                phase_unwrap=phase_wrap+2*K2*pi;
                residue=wrapToPi(phase_unwrap-phi1);
                rms2=sqrt(mean2((residue).^2));
                delta_rms=abs(rms2-rms1);
                rms1=rms2;
                N=N+1;
            end
        end
        %% Unwrap LS FD FFT     #3
        function [phase_unwrap]=unwrap_LS_FD_FFT(phase_wrap)
            [M,N]=size(phase_wrap);
            p1=[phase_wrap,flip(phase_wrap,2);flip(phase_wrap,1),flip(flip(phase_wrap,2),1)]; %mirror padded
            [height,width]=size(p1);
            mx=-width/2:width/2-1;my=-height/2:height/2-1;%samle
            fx=2*pi*1i*mx/width;
            fy=2*pi*1i*my/height;
            [fx,fy]=meshgrid(fx,fy);
            f=fx.^2+fy.^2;
            psi=p1;
            edx = [zeros([size(psi,1),1]), wrapToPi(diff(psi, 1, 2)), zeros([size(psi,1),1])];
            edy = [zeros([1,size(psi,2)]); wrapToPi(diff(psi, 1, 1)); zeros([1,size(psi,2)])];
            rho = diff(edx, 1, 2) + diff(edy, 1, 1);
            temp3=fftshift(fft2(rho));
            temp3=temp3./(f+eps);
            p2=ifft2(ifftshift(temp3));
            p2=real(p2);
            phase_unwrap=p2(1:M,1:N);
        end
        %% Unwrap LS FD FFT iter    #4
        function [phase_unwrap,N]= unwrap_LS_FD_FFT_iter(phase_wrap)
            phi1 = unwrap_LS_FD_FFT(phase_wrap);
            phi1=phi1+mean2(phase_wrap)-mean2(phi1); %adjust piston
            K1=round((phi1-phase_wrap)/2/pi);  %calculate integer K
            phase_unwrap=phase_wrap+2*K1*pi;
            residue=wrapToPi(phase_unwrap-phi1);
            phi1=phi1+unwrap_LS_FD_FFT(residue);
            phi1=phi1+mean2(phase_wrap)-mean2(phi1); %adjust piston
            K2=round((phi1-phase_wrap)/2/pi);  %calculate integer K
            phase_unwrap=phase_wrap+2*K2*pi;
            residue=wrapToPi(phase_unwrap-phi1);
            rms1=sqrt(mean2((residue).^2));
            delta_rms=rms1;
            N=0;
            while sum(sum(abs(K2-K1)))>0 && delta_rms>10^-5
                K1=K2;
                phic=unwrap_LS_FD_FFT(residue);
                phi1=phi1+phic;
                phi1=phi1+mean2(phase_wrap)-mean2(phi1); %adjust piston
                K2=round((phi1-phase_wrap)/2/pi);  %calculate integer K
                phase_unwrap=phase_wrap+2*K2*pi;
                residue=wrapToPi(phase_unwrap-phi1);
                rms2=sqrt(mean2((residue).^2));
                delta_rms=abs(rms2-rms1);
                rms1=rms2;
                N=N+1;
            end
        end

        %% Unwrap TIE FD DCT    #5
        function [phase_unwrap,rho]=unwrap_TIE_FD_DCT(phase_wrap)
            psi=exp(1i*phase_wrap);
            edx = [zeros([size(psi,1),1]), wrapToPi(diff(psi, 1, 2)), zeros([size(psi,1),1])];
            edy = [zeros([1,size(psi,2)]); wrapToPi(diff(psi, 1, 1)); zeros([1,size(psi,2)])];
            lap = diff(edx, 1, 2) + diff(edy, 1, 1);
            rho=imag(conj(psi).*lap);
            % get the result by solving the poisson equation
            phase_unwrap = solvePoisson(rho);
        end

        %% Unwrap TIE FD DCT iter   #6
        function [phase_unwrap,N]= unwrap_TIE_FD_DCT_iter(phase_wrap)
            phi1 = unwrap_TIE_FD_DCT(phase_wrap);
            phi1=phi1+mean2(phase_wrap)-mean2(phi1); %adjust piston
            K1=round((phi1-phase_wrap)/2/pi);  %calculate integer K
            phase_unwrap=phase_wrap+2*K1*pi;
            residue=wrapToPi(phase_unwrap-phi1);
            phi1=phi1+unwrap_TIE_FD_DCT(residue);
            phi1=phi1+mean2(phase_wrap)-mean2(phi1); %adjust piston
            K2=round((phi1-phase_wrap)/2/pi);  %calculate integer K
            phase_unwrap=phase_wrap+2*K2*pi;
            residue=wrapToPi(phase_unwrap-phi1);
            rms1=sqrt(mean2((residue).^2));
            delta_rms=rms1;
            N=0;
            while sum(sum(abs(K2-K1)))>0 && delta_rms>10^-5
                K1=K2;
                phic=unwrap_TIE_FD_DCT(residue);
                phi1=phi1+phic;
                phi1=phi1+mean2(phase_wrap)-mean2(phi1); %adjust piston
                K2=round((phi1-phase_wrap)/2/pi);  %calculate integer K
                phase_unwrap=phase_wrap+2*K2*pi;
                residue=wrapToPi(phase_unwrap-phi1);
                rms2=sqrt(mean2((residue).^2));
                delta_rms=abs(rms2-rms1);
                rms1=rms2;
                N=N+1;
            end
        end

        %% unwrap TIE FD FFT    #7
        function [phase_unwrap] = unwrap_TIE_FD_FFT(phase_wrap)
            [M,N]=size(phase_wrap);
            p1=[phase_wrap,flip(phase_wrap,2);flip(phase_wrap,1),flip(flip(phase_wrap,2),1)]; %mirror padded
            [height,width]=size(p1);
            mx=-width/2:width/2-1;my=-height/2:height/2-1;%samle
            fx=2*pi*1i*mx/width;
            fy=2*pi*1i*my/height;
            [fx,fy]=meshgrid(fx,fy);
            f=fx.^2+fy.^2;
            psi=exp(1i*p1);
            edx = [zeros([size(psi,1),1]), wrapToPi(diff(psi, 1, 2)), zeros([size(psi,1),1])];
            edy = [zeros([1,size(psi,2)]); wrapToPi(diff(psi, 1, 1)); zeros([1,size(psi,2)])];
            lap = diff(edx, 1, 2) + diff(edy, 1, 1);
            rho=imag(conj(psi).*lap);
            temp2=rho;
            temp3=fftshift(fft2(temp2));
            temp3=temp3./(f+eps);
            p2=ifft2(ifftshift(temp3));
            phase_unwrap=p2(1:M,1:N);
        end
        %% unwrap TIE FD FFT iter   #8
        function [phase_unwrap,N]=unwrap_TIE_FD_FFT_iter(phase_wrap)
            phi1 = unwrap_TIE_FD_FFT(phase_wrap);
            phi1=phi1+mean2(phase_wrap)-mean2(phi1); %adjust piston
            K1=round((phi1-phase_wrap)/2/pi);  %calculate integer K
            phase_unwrap=phase_wrap+2*K1*pi;
            residue=wrapToPi(phase_unwrap-phi1);
            phi1=phi1+unwrap_TIE_FD_FFT(residue);
            phi1=phi1+mean2(phase_wrap)-mean2(phi1); %adjust piston
            K2=round((phi1-phase_wrap)/2/pi);  %calculate integer K
            phase_unwrap=phase_wrap+2*K2*pi;
            residue=wrapToPi(phase_unwrap-phi1);
            rms1=sqrt(mean2((residue).^2));
            delta_rms=rms1;
            N=0;
            while sum(sum(abs(K2-K1)))>0 && delta_rms>10^-5
                K1=K2;
                phic=unwrap_TIE_FD_FFT(residue);
                phi1=phi1+phic;
                phi1=phi1+mean2(phase_wrap)-mean2(phi1); %adjust piston
                K2=round((phi1-phase_wrap)/2/pi);  %calculate integer K
                phase_unwrap=phase_wrap+2*K2*pi;
                residue=wrapToPi(phase_unwrap-phi1);
                rms2=sqrt(mean2((residue).^2));
                delta_rms=abs(rms2-rms1);
                rms1=rms2;
                N=N+1;
            end
        end

        %% unwrap TIE FFT DCT   #9

        function [phase_unwrap]=unwrap_TIE_FFT_DCT(phase_wrap)
            [M,N]=size(phase_wrap);
            p1=[phase_wrap,flip(phase_wrap,2);flip(phase_wrap,1),flip(flip(phase_wrap,2),1)]; %mirror padded
            [height,width]=size(p1);
            mx=-width/2:width/2-1;my=-height/2:height/2-1;%samle
            fx=2*pi*1i*mx/width;
            fy=2*pi*1i*my/height;
            [fx,fy]=meshgrid(fx,fy);
            f=fx.^2+fy.^2;
            u0=exp(1i*p1);
            temp1=fftshift(fft2(u0));
            temp1=temp1.*f;
            temp2=ifft2(ifftshift(temp1));
            temp2=imag(temp2.*exp(-1i*p1));%temp2 is rou;
            rho=temp2(1:M,1:N);
            phase_unwrap = solvePoisson(rho);
        end
        %% unwrap TIE FFT DCT iter  #10

        function [phase_unwrap,N]=unwrap_TIE_FFT_DCT_iter(phase_wrap)
            phi1 = unwrap_TIE_FFT_DCT(phase_wrap);
            phi1=phi1+mean2(phase_wrap)-mean2(phi1); %adjust piston
            K1=round((phi1-phase_wrap)/2/pi);  %calculate integer K
            phase_unwrap=phase_wrap+2*K1*pi;
            residue=wrapToPi(phase_unwrap-phi1);
            phi1=phi1+unwrap_TIE_FFT_DCT(residue);
            phi1=phi1+mean2(phase_wrap)-mean2(phi1); %adjust piston
            K2=round((phi1-phase_wrap)/2/pi);  %calculate integer K
            phase_unwrap=phase_wrap+2*K2*pi;
            residue=wrapToPi(phase_unwrap-phi1);
            rms1=sqrt(mean2((residue).^2));
            delta_rms=rms1;
            N=0;
            while sum(sum(abs(K2-K1)))>0 && delta_rms>10^-1   %note:10^-1 is more suitable for the method while the treshold for other five methods is set to 10^-5(see details in Ref[2])
                K1=K2;
                phic=unwrap_TIE_FFT_DCT(residue);
                phi1=phi1+phic;
                phi1=phi1+mean2(phase_wrap)-mean2(phi1); %adjust piston
                K2=round((phi1-phase_wrap)/2/pi);  %calculate integer K
                phase_unwrap=phase_wrap+2*K2*pi;
                residue=wrapToPi(phase_unwrap-phi1);
                rms2=sqrt(mean2((residue).^2));
                delta_rms=abs(rms2-rms1);
                rms1=rms2;
                N=N+1;
            end
        end
        %% unwrap TIE FFT FFT   #11
        function [phase_unwrap]=unwrap_TIE_FFT_FFT(phase_wrap)
            [M,N]=size(phase_wrap);
            p1=[phase_wrap,flip(phase_wrap,2);flip(phase_wrap,1),flip(flip(phase_wrap,2),1)]; %mirror padded
            [height,width]=size(p1);
            mx=-width/2:width/2-1;my=-height/2:height/2-1;%samle
            fx=2*pi*1i*mx/width;
            fy=2*pi*1i*my/height;
            [fx,fy]=meshgrid(fx,fy);
            f=fx.^2+fy.^2;
            u0=exp(1i*p1);
            temp1=fftshift(fft2(u0));
            temp1=temp1.*f;
            temp2=ifft2(ifftshift(temp1));
            temp2=imag(temp2.*exp(-1i*p1));%temp2 is rho;
            temp3=fftshift(fft2(temp2));
            temp3=temp3./(f+eps);
            p2=ifft2(ifftshift(temp3));
            phase_unwrap=p2(1:M,1:N);
        end
        %% Unwrap TIE FFT FFT iter  #12
        function [phase_unwrap,N]=unwrap_TIE_FFT_FFT_iter(phase_wrap)
            phi1 = unwrap_TIE_FFT_FFT(phase_wrap);
            phi1=phi1+mean2(phase_wrap)-mean2(phi1); %adjust piston
            K1=round((phi1-phase_wrap)/2/pi);  %calculate integer K
            phase_unwrap=phase_wrap+2*K1*pi;
            residue=wrapToPi(phase_unwrap-phi1);
            phi1=phi1+unwrap_TIE_FFT_FFT(residue);
            phi1=phi1+mean2(phase_wrap)-mean2(phi1); %adjust piston
            K2=round((phi1-phase_wrap)/2/pi);  %calculate integer K
            phase_unwrap=phase_wrap+2*K2*pi;
            residue=wrapToPi(phase_unwrap-phi1);
            rms1=sqrt(mean2((residue).^2));
            delta_rms=rms1;
            N=0;
            while sum(sum(abs(K2-K1)))>0 && delta_rms>10^-5
                K1=K2;
                phic=unwrap_TIE_FFT_FFT(residue);
                phi1=phi1+phic;
                phi1=phi1+mean2(phase_wrap)-mean2(phi1); %adjust piston
                K2=round((phi1-phase_wrap)/2/pi);  %calculate integer K
                phase_unwrap=phase_wrap+2*K2*pi;
                residue=wrapToPi(phase_unwrap-phi1);
                rms2=sqrt(mean2((residue).^2));
                delta_rms=abs(rms2-rms1);
                rms1=rms2;
                N=N+1;
            end
        end
        %% function Solve Poisson
        function phi = solvePoisson(rho)
            % solve the poisson equation using dct
            dctRho = dct2(rho);
            [N, M] = size(rho);
            [I, J] = meshgrid(0:M-1, 0:N-1);
            dctPhi = dctRho ./ 2 ./ (cos(pi*I/M) + cos(pi*J/N) - 2);
            dctPhi(1,1) = 0; % handling the inf/nan value

            % now invert to get the result
            phi = idct2(dctPhi);

        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Thuật toán anh Linh gửi
        % Fast unwrapping 2D phase image using the algorithm given in:                 %
        %     M. A. Herráez, D. R. Burton, M. J. Lalor, and M. A. Gdeisat,             %
        %     "Fast two-dimensional phase-unwrapping algorithm based on sorting by     %
        %     reliability following a noncontinuous path", Applied Optics, Vol. 41,    %
        %     Issue 35, pp. 7437-7444 (2002).                                          %
        %                                                                              %
        % If using this code for publication, please kindly cite the following:        %
        % * M. A. Herraez, D. R. Burton, M. J. Lalor, and M. A. Gdeisat, "Fast         %
        %   two-dimensional phase-unwrapping algorithm based on sorting by reliability %
        %   following a noncontinuous path", Applied Optics, Vol. 41, Issue 35,        %
        %   pp. 7437-7444 (2002).                                                      %
        % * M. F. Kasim, "Fast 2D phase unwrapping implementation in MATLAB",          %
        %   https://github.com/mfkasim91/unwrap_phase/ (2017).                         %
        %                                                                              %
        % Input:                                                                       %
        % * img: The wrapped phase image either from -pi to pi or from 0 to 2*pi.      %
        %        If there are unwanted regions, it should be filled with NaNs.         %
        %                                                                              %
        % Output:                                                                      %
        % * res_img: The unwrapped phase with arbitrary offset.                        %
        %                                                                              %
        % Author:                                                                      %
        %     Muhammad F. Kasim, University of Oxford (2017)                           %
        %     Email: firman.kasim@gmail.com                                            %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        function res_img = unwrap_phase_Linh(img)
            [Ny, Nx] = size(img);

            % get the reliability
            reliability = get_reliability(img); % (Ny,Nx)

            % get the edges
            [h_edges, v_edges] = get_edges(reliability); % (Ny,Nx) and (Ny,Nx)

            % combine all edges and sort it
            edges = [h_edges(:); v_edges(:)];
            edge_bound_idx = Ny * Nx; % if i <= edge_bound_idx, it is h_edges
            [~, edge_sort_idx] = sort(edges, 'descend');

            % get the indices of pixels adjacent to the edges
            idxs1 = mod(edge_sort_idx - 1, edge_bound_idx) + 1;
            idxs2 = idxs1 + 1 + (Ny - 1) .* (edge_sort_idx <= edge_bound_idx);

            % label the group
            group = reshape(1:numel(img), Ny*Nx, 1);
            is_grouped = zeros(Ny*Nx,1);
            group_members = cell(Ny*Nx,1);
            for i = 1:size(is_grouped,1)
                group_members{i} = i;
            end
            num_members_group = ones(Ny*Nx,1);

            % propagate the unwrapping
            res_img = img;
            num_nan = sum(isnan(edges)); % count how many nan-s and skip them
            for i = num_nan+1 : length(edge_sort_idx)
                % get the indices of the adjacent pixels
                idx1 = idxs1(i);
                idx2 = idxs2(i);

                % skip if they belong to the same group
                if (group(idx1) == group(idx2))
                    continue;
                end

                % idx1 should be ungrouped (swap if idx2 ungrouped and idx1 grouped)
                % otherwise, activate the flag all_grouped.
                % The group in idx1 must be smaller than in idx2. If initially
                % group(idx1) is larger than group(idx2), then swap it.
                all_grouped = 0;
                if is_grouped(idx1)
                    if ~is_grouped(idx2)
                        idxt = idx1;
                        idx1 = idx2;
                        idx2 = idxt;
                    elseif num_members_group(group(idx1)) > num_members_group(group(idx2))
                        idxt = idx1;
                        idx1 = idx2;
                        idx2 = idxt;
                        all_grouped = 1;
                    else
                        all_grouped = 1;
                    end
                end

                % calculate how much we should add to the idx1 and group
                dval = floor((res_img(idx2) - res_img(idx1) + pi) / (2*pi)) * 2*pi;

                % which pixel should be changed
                g1 = group(idx1);
                g2 = group(idx2);
                if all_grouped
                    pix_idxs = group_members{g1};
                else
                    pix_idxs = idx1;
                end

                % add the pixel value
                if dval ~= 0
                    res_img(pix_idxs) = res_img(pix_idxs) + dval;
                end

                % change the group
                len_g1 = num_members_group(g1);
                len_g2 = num_members_group(g2);
                group_members{g2}(len_g2+1:len_g2+len_g1) = pix_idxs;
                group(pix_idxs) = g2; % assign the pixels to the new group
                num_members_group(g2) = num_members_group(g2) + len_g1;

                % mark idx1 and idx2 as already being grouped
                is_grouped(idx1) = 1;
                is_grouped(idx2) = 1;
            end
        end

        function rel = get_reliability(img)
            rel = zeros(size(img));

            % get the shifted images (N-2, N-2)
            img_im1_jm1 = img(1:end-2, 1:end-2);
            img_i_jm1   = img(2:end-1, 1:end-2);
            img_ip1_jm1 = img(3:end  , 1:end-2);
            img_im1_j   = img(1:end-2, 2:end-1);
            img_i_j     = img(2:end-1, 2:end-1);
            img_ip1_j   = img(3:end  , 2:end-1);
            img_im1_jp1 = img(1:end-2, 3:end  );
            img_i_jp1   = img(2:end-1, 3:end  );
            img_ip1_jp1 = img(3:end  , 3:end  );

            % calculate the difference
            gamma = @(x) sign(x) .* mod(abs(x), pi);
            H  = gamma(img_im1_j   - img_i_j) - gamma(img_i_j - img_ip1_j  );
            V  = gamma(img_i_jm1   - img_i_j) - gamma(img_i_j - img_i_jp1  );
            D1 = gamma(img_im1_jm1 - img_i_j) - gamma(img_i_j - img_ip1_jp1);
            D2 = gamma(img_im1_jp1 - img_i_j) - gamma(img_i_j - img_ip1_jm1);

            % calculate the second derivative
            D = sqrt(H.*H + V.*V + D1.*D1 + D2.*D2);

            % assign the reliability as 1 / D
            rel(2:end-1, 2:end-1) = 1./D;

            % assign all nan's in rel with non-nan in img to 0
            % also assign the nan's in img to nan
            rel(isnan(rel) & ~isnan(img)) = 0;
            rel(isnan(img)) = nan;
        end

        function [h_edges, v_edges] = get_edges(rel)
            [Ny, Nx] = size(rel);
            h_edges = [rel(1:end, 2:end) + rel(1:end, 1:end-1), nan(Ny, 1)];
            v_edges = [rel(2:end, 1:end) + rel(1:end-1, 1:end); nan(1, Nx)];
        end


        %% 2D weight phase unwrapping
        function phi = phase_unwrap_2Dweight(psi, weight)
            if (nargin < 2) % unweighted phase unwrap
                % get the wrapped differences of the wrapped values
                dx = [zeros([size(psi,1),1]), wrapToPi(diff(psi, 1, 2)), zeros([size(psi,1),1])];
                dy = [zeros([1,size(psi,2)]); wrapToPi(diff(psi, 1, 1)); zeros([1,size(psi,2)])];
                rho = diff(dx, 1, 2) + diff(dy, 1, 1);

                % get the result by solving the poisson equation
                phi = solvePoisson(rho);

            else % weighted phase unwrap
                % check if the weight has the same size as psi
                if (~all(size(weight) == size(psi)))
                    error('Argument error: Size of the weight must be the same as size of the wrapped phase');
                end

                % vector b in the paper (eq 15) is dx and dy
                dx = [wrapToPi(diff(psi, 1, 2)), zeros([size(psi,1),1])];
                dy = [wrapToPi(diff(psi, 1, 1)); zeros([1,size(psi,2)])];

                % multiply the vector b by weight square (W^T * W)
                WW = weight .* weight;
                WWdx = WW .* dx;
                WWdy = WW .* dy;

                % applying A^T to WWdx and WWdy is like obtaining rho in the unweighted case
                WWdx2 = [zeros([size(psi,1),1]), WWdx];
                WWdy2 = [zeros([1,size(psi,2)]); WWdy];
                rk = diff(WWdx2, 1, 2) + diff(WWdy2, 1, 1);
                normR0 = norm(rk(:));

                % start the iteration
                eps = 1e-8;
                k = 0;
                phi = zeros(size(psi));
                while (~all(rk == 0))
                    zk = solvePoisson(rk);
                    k = k + 1;

                    if (k == 1)
                        pk = zk;
                    else
                        betak = sum(sum(rk .* zk)) / sum(sum(rkprev .* zkprev));
                        pk = zk + betak * pk;
                    end

                    % save the current value as the previous values
                    rkprev = rk;
                    zkprev = zk;

                    % perform one scalar and two vectors update
                    Qpk = applyQ(pk, WW);
                    alphak = sum(sum(rk .* zk)) / sum(sum(pk .* Qpk));
                    phi = phi + alphak * pk;
                    rk = rk - alphak * Qpk;

                    % check the stopping conditions
                    if ((k >= numel(psi)) || (norm(rk(:)) < eps * normR0))
                        break;
                    end
                end
            end
        end


        % apply the transformation (A^T)(W^T)(W)(A) to 2D matrix
        function Qp = applyQ(p, WW)
            % apply (A)
            dx = [diff(p, 1, 2), zeros([size(p,1),1])];
            dy = [diff(p, 1, 1); zeros([1,size(p,2)])];

            % apply (W^T)(W)
            WWdx = WW .* dx;
            WWdy = WW .* dy;

            % apply (A^T)
            WWdx2 = [zeros([size(p,1),1]), WWdx];
            WWdy2 = [zeros([1,size(p,2)]); WWdy];
            Qp = diff(WWdx2,1,2) + diff(WWdy2,1,1);
        end

    end
end