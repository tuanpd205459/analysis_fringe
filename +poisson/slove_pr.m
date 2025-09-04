function [ PV,RMS ] = slove_pr( data )
%   Get PV and RMS values of data
%   for example: [PV,RMS]=slove_pr(data);
%   note: data is a 2D array
PV=max(max(data))-min(min(data));
RMS=(sum(data(~isnan(data)).^2)/length(data(~isnan(data)))).^0.5;
end

