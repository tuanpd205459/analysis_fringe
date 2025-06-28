%% Hàm tính sai lệch z(x,y) - để tính các thông số nhám 3D
function difference_Matrix = myDifference (inputMatrix, poly_order)
    % Khởi tạo mảng để lưu kết quả Sz cho từng hàng
    difference_Matrix = zeros(size(inputMatrix));
    for i = 1:size(inputMatrix, 1)
        row_vector = inputMatrix(i, :);
        % Tính giá trị trung binh của hàng i
        meanLine_row = processing.postProcess.myMeanLine(row_vector,poly_order);
        % Tính sai lech hang i
        difference_Matrix (i,:) = inputMatrix(i,:) - meanLine_row;
    end
end