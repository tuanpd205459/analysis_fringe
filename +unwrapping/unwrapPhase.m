function unwrapped_Phase = unwrapPhase(wrappedPhase, methodGroup, methodType)
addpath(fullfile(fileparts(mfilename('fullpath')), 'source_code'));
% UNWRAPPHASE - Hàm thực hiện tháo gỡ pha (phase unwrapping) với nhiều thuật toán khác nhau.
%
% Cách sử dụng:
%   unwrapped_Phase = unwrapPhase(wrappedPhase);
%   unwrapped_Phase = unwrapPhase(wrappedPhase, methodGroup);
%   unwrapped_Phase = unwrapPhase(wrappedPhase, methodGroup, methodType);
%
% Đầu vào:
%   - wrappedPhase  : Ma trận pha bị quấn (wrapped phase).
%   - methodGroup   : Nhóm thuật toán tháo gỡ pha (string):
%       + 'poisson'   : Phương pháp Poisson (mặc định dùng DCT).
%       + 'ls'        : Least-Squares (LS).
%       + 'tie'       : Transport of Intensity Equation (TIE).
%       + 'linh'      : Phương pháp của a Linh.
%       + '2dweight'  : Phương pháp 2D weighted phase unwrapping.
%   - methodType    : Phương pháp con (chỉ áp dụng cho LS và TIE) (string):
%       + 'dct'   : Dùng Discrete Cosine Transform (DCT) (mặc định).
%       + 'fft'   : Dùng Fast Fourier Transform (FFT).
%       + 'iter'  : Sử dụng phương pháp có lặp (iterative).
%
% Đầu ra:
%   - unwrapped_Phase: Ma trận pha đã tháo gỡ.
%
% Ví dụ:
%   unwrapped_Phase = unwrapPhase(wrappedPhase, 'ls', 'dct'); % LS với DCT
%   unwrapped_Phase = unwrapPhase(wrappedPhase, 'tie', 'fft'); % TIE với FFT
%   unwrapped_Phase = unwrapPhase(wrappedPhase, 'linh'); % Phương pháp của a Linh
%   unwrapped_Phase = unwrapPhase(wrappedPhase, '2dweight'); % 2D weighted phase unwrapping
%
% Nếu không nhập methodGroup và methodType, mặc định sử dụng 'ls' với 'dct'.
%


% Mặc định nếu không nhập vào
if nargin < 2 || isempty(methodGroup)
    methodGroup = 'ls'; % Least-Squares mặc định
end
if nargin < 3 || isempty(methodType)
    methodType = 'dct'; % Mặc định dùng DCT, không lặp
end

% Xử lý theo nhóm phương pháp
switch lower(methodGroup)
    case 'poisson'
        unwrapped_Phase = unwrap_LS_FD_DCT(wrappedPhase);
    
    case 'ls' % Least-Squares (LS)
        switch lower(methodType)
            case 'dct'
                unwrapped_Phase = unwrap_LS_FD_DCT(wrappedPhase);
            case 'fft'
                unwrapped_Phase = unwrap_LS_FD_FFT(wrappedPhase);
            case 'iter'
                [unwrapped_Phase, ~] = unwrap_LS_FD_DCT_iter(wrappedPhase);
            otherwise
                error('Phương pháp LS không hợp lệ! Chọn ''dct'', ''fft'', hoặc ''iter''.');
        end

    case 'tie' % Transport of Intensity Equation (TIE)
        switch lower(methodType)
            case 'dct'
                unwrapped_Phase = unwrap_TIE_FD_DCT(wrappedPhase);
            case 'fft'
                unwrapped_Phase = unwrap_TIE_FFT_FFT(wrappedPhase);
            case 'iter'
                [unwrapped_Phase, ~] = unwrap_TIE_FD_DCT_iter(wrappedPhase);
            otherwise
                error('Phương pháp TIE không hợp lệ! Chọn ''dct'', ''fft'', hoặc ''iter''.');
        end

    case 'linh' % Phương pháp a Linh
        unwrapped_Phase = unwrap_phase_linh(wrappedPhase);
        
    case '2dweight' % Phương pháp 2D weighted phase unwrapping
        unwrapped_Phase = phase_unwrap_2dweight(wrappedPhase);

    otherwise
        error('Phương pháp không hợp lệ! Chọn ''poisson'', ''ls'', ''tie'', ''linh'' hoặc ''2dweight''.');
end

end
