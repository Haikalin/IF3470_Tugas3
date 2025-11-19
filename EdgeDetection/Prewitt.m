function [outputMatrix] = Prewitt(inputImg)
%Prewitt applies the Sobel operator for edge detection.
%   Calculates the magnitude of the gradient using the standard Prewitt
%   kernels in the horizontal (Gx) and vertical (Gy) directions.

    % Define standard Prewitt kernels
    K_Prewitt_x = [-1 0 1; -1 0 1; -1 0 1]; % Horizontal Edge Detector
    K_Prewitt_y = [1 1 1; 0 0 0; -1 -1 -1]; % Vertical Edge Detector

    % Convert image to double precision
    img_double = double(inputImg); 

    % --- Convolution ---
    % Use the 'same' option to keep the output size identical to the input
    Gx = conv2(img_double, K_Prewitt_x, 'same');
    Gy = conv2(img_double, K_Prewitt_y, 'same');

    % --- Gradient Magnitude Calculation ---
    % Calculate the magnitude M = sqrt(Gx^2 + Gy^2)
    outputMatrix = sqrt((Gx .^ 2) + (Gy .^ 2));

end