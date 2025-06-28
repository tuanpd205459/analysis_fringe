%% function tính Mean line
function outMeanLine = myMeanLine(inputRow, poly_order)
%global poly_order
N = length(inputRow);
x = 1:N;
p = polyfit(x, inputRow, poly_order);
% Tính toán giá trị của đường trung bình cong
outMeanLine = polyval(p, x);
end