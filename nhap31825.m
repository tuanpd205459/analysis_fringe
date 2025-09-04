clc,clear, close all;

load("BW_NEW.mat");
figure;imshow(BW_NEW); title("finall");
% BW = bwmorph(BW,'skel',Inf); % skeleton hóa
%%
BW =BW_NEW;



