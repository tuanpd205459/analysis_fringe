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