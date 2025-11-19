function [outputMatrix] = Roberts(inputImg)
%Roberts applies the Roberts operator for edge detection.
%   Calculates the magnitude of the gradient using the standard Roberts
%   kernels in the diagonal directions.

    % Define standard Roberts kernels
    K_Roberts_diag1 = [1 0; 0 -1]; % Diagonal 1 Edge Detector
    K_Roberts_diag2 = [0 1; -1 0]; % Diagonal 2 Edge Detector

    % Convert image to double precision
    img_double = double(inputImg); 

    % --- Convolution ---
    % Use the 'same' option to keep the output size identical to the input
    G1 = conv2(img_double, K_Roberts_diag1, 'same');
    G2 = conv2(img_double, K_Roberts_diag2, 'same');

    % --- Gradient Magnitude Calculation ---
    % Calculate the magnitude M = sqrt(G1^2 + G2^2)
    outputMatrix = sqrt((G1 .^ 2) + (G2 .^ 2));

end