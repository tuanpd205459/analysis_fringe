function Z = my_get_zernike_poly(j, rho, theta)

    switch j
        % === n = 0 (Piston) ===
        case 1  % Z1: Piston
            Z = ones(size(rho));
            
        % === n = 1 (Tilt) ===
        case 2  % Z2: Tilt X (Tip)
            Z = rho .* cos(theta);
        case 3  % Z3: Tilt Y (Tilt)
            Z = rho .* sin(theta);
            
        % === n = 2 (Defocus, Astigmatism) ===
        case 4  % Z4: Defocus
            Z = 2*rho.^2 - 1;
        case 5  % Z5: Oblique Astigmatism
            Z = rho.^2 .* cos(2*theta);
        case 6  % Z6: Vertical Astigmatism
            Z = rho.^2 .* sin(2*theta);
            
        % === n = 3 (Coma, Trefoil) ===
        case 7  % Z7: Vertical Coma
            Z = (3*rho.^3 - 2*rho) .* cos(theta);
        case 8  % Z8: Horizontal Coma
            Z = (3*rho.^3 - 2*rho) .* sin(theta);
        case 9  % Z9: Trefoil Y
            Z = rho.^3 .* cos(3*theta);
        case 10 % Z10: Trefoil X
            Z = rho.^3 .* sin(3*theta);
            
        % === n = 4 (Spherical, Secondary Astig, Quadrafoil) ===
        case 11 % Z11: Primary Spherical
            Z = 6*rho.^4 - 6*rho.^2 + 1;
        case 12 % Z12: Secondary Astigmatism Y
            Z = (4*rho.^4 - 3*rho.^2) .* cos(2*theta);
        case 13 % Z13: Secondary Astigmatism X
            Z = (4*rho.^4 - 3*rho.^2) .* sin(2*theta);
        case 14 % Z14: Quadrafoil Y
            Z = rho.^4 .* cos(4*theta);
        case 15 % Z15: Quadrafoil X
            Z = rho.^4 .* sin(4*theta);
            
        % === n = 5 (Secondary Coma, etc.) ===
        case 16 % Z16: Secondary Coma X
            Z = (10*rho.^5 - 12*rho.^3 + 3*rho) .* cos(theta);
        case 17 % Z17: Secondary Coma Y
            Z = (10*rho.^5 - 12*rho.^3 + 3*rho) .* sin(theta);
        case 18 % Z18: Secondary Trefoil Y
            Z = (5*rho.^5 - 4*rho.^3) .* cos(3*theta);
        case 19 % Z19: Secondary Trefoil X
            Z = (5*rho.^5 - 4*rho.^3) .* sin(3*theta);
        case 20 % Z20: Pentafoil Y
            Z = rho.^5 .* cos(5*theta);
        case 21 % Z21: Pentafoil X
            Z = rho.^5 .* sin(5*theta);
            
        % === n = 6 (Secondary Spherical, Tertiary Astig, etc.) ===
        case 22 % Z22: Secondary Spherical
            Z = 20*rho.^6 - 30*rho.^4 + 12*rho.^2 - 1;
        case 23 % Z23: Tertiary Astigmatism Y
            Z = (15*rho.^6 - 20*rho.^4 + 6*rho.^2) .* cos(2*theta);
        case 24 % Z24: Tertiary Astigmatism X
            Z = (15*rho.^6 - 20*rho.^4 + 6*rho.^2) .* sin(2*theta);
        case 25 % Z25: Secondary Quadrafoil Y
            Z = (6*rho.^6 - 5*rho.^4) .* cos(4*theta);
        case 26 % Z26: Secondary Quadrafoil X
            Z = (6*rho.^6 - 5*rho.^4) .* sin(4*theta);
        case 27 % Z27: Hexafoil Y
            Z = rho.^6 .* cos(6*theta);
        case 28 % Z28: Hexafoil X
            Z = rho.^6 .* sin(6*theta);
            
        % === n = 7 (Tertiary Coma, etc.) ===
        case 29 % Z29: Tertiary Coma X
            Z = (35*rho.^7 - 60*rho.^5 + 30*rho.^3 - 4*rho) .* cos(theta);
        case 30 % Z30: Tertiary Coma Y
            Z = (35*rho.^7 - 60*rho.^5 + 30*rho.^3 - 4*rho) .* sin(theta);
        case 31 % Z31: Tertiary Trefoil Y
            Z = (21*rho.^7 - 30*rho.^5 + 10*rho.^3) .* cos(3*theta);
        case 32 % Z32: Tertiary Trefoil X
            Z = (21*rho.^7 - 30*rho.^5 + 10*rho.^3) .* sin(3*theta);
        case 33 % Z33: Secondary Pentafoil Y
            Z = (7*rho.^7 - 6*rho.^5) .* cos(5*theta);
        case 34 % Z34: Secondary Pentafoil X
            Z = (7*rho.^7 - 6*rho.^5) .* sin(5*theta);
        case 35 % Z35: Heptafoil Y
            Z = rho.^7 .* cos(7*theta);
        case 36 % Z36: Heptafoil X
            Z = rho.^7 .* sin(7*theta);
            
        otherwise
            error(['Hàm chỉ hỗ trợ tối đa 36 số hạng Zernike. Bạn đang yêu cầu Z', num2str(j)]);
    end
end
    
