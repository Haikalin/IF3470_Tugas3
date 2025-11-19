function [outputMatrix] = LoG(inputImg, sigma)
%LoG applies the LoG (Laplacian of Gaussian) operator for edge detection using 
%   a manually calculated kernel.
%   
%   Inputs:
%       inputImg - The input grayscale image.
%       sigma    - The standard deviation of the Gaussian, controls the scale.
    
    % Convert image to double precision
    img_double = double(inputImg);
    
    % --- 1. Manually Generate the LoG Kernel (Mexican Hat) ---
    
    % Determine the kernel size (N x N) based on 3 standard deviations
    radius = ceil(3 * sigma); 
    
    % Create coordinate matrices (X and Y) centered at zero
    x_vec = -radius:radius; % x_vec defines the range of coordinates
    [X, Y] = meshgrid(x_vec, x_vec); % Creates square coordinate grid (Y is based on x_vec)
    
    % Pre-calculate required terms
    R_squared = X.^2 + Y.^2;
    sigma_sq = sigma^2;
    sigma_fourth = sigma^4;

    % Apply the LoG mathematical formula
    % K_LoG(x, y, sigma) = [(x^2 + y^2 - 2*sigma^2) / sigma^4] * exp(-(x^2 + y^2) / (2*sigma^2))
    
    exponent_term = exp(-R_squared / (2 * sigma_sq));
    coefficient_term = (R_squared - 2 * sigma_sq) / sigma_fourth;
    
    K_LoG = coefficient_term .* exponent_term;
    
    % --- 2. Convolution ---
    % Perform a single convolution with the manually calculated LoG kernel
    outputMatrix = conv2(img_double, K_LoG, 'same');
    
end