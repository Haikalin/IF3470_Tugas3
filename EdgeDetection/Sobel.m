function [outputMatrix] = Sobel(inputImg)
%SOBEL Applies the Sobel operator for edge detection.
%   Calculates the magnitude of the gradient using the standard Sobel
%   kernels in the horizontal (Gx) and vertical (Gy) directions.

    % Define standard Sobel kernels
    Gx = [-1 0 1; -2 0 2; -1 0 1]; % Horizontal Edge Detector
    Gy = [-1 -2 -1; 0 0 0; 1 2 1]; % Vertical Edge Detector

    % --- Input Validation and Conversion ---
    % Sobel works best on grayscale images (double precision)
    if islogical(inputImg) || isinteger(inputImg)
        % Convert to double for accurate calculation
        img_double = double(inputImg); 
    else
        img_double = inputImg; % Assume it's already double/single
    end

    % --- Convolution ---
    % Use the 'same' option to keep the output size identical to the input
    outputGx = conv2(img_double, Gx, 'same');
    outputGy = conv2(img_double, Gy, 'same');

    % --- Gradient Magnitude Calculation ---
    % Calculate the magnitude M = sqrt(Gx^2 + Gy^2)
    outputMatrix = sqrt((outputGx .^ 2) + (outputGy .^ 2));

end