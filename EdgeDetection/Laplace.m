function [outputMatrix] = Laplace(inputImg)
%Laplace applies the raw Laplacian operator (second derivative) for edge detection.
%   This function performs a single convolution with the 8-neighbor
%   (isotropic) kernel. The output matrix represents the second derivative,
%   where edges are located at the zero-crossings.
    
    % Define 8-neighbor (isotropic) Laplacian kernel
    K_Laplace = [1 1 1; 1 -8 1; 1 1 1];
    
    % Convert image to double precision
    img_double = double(inputImg); 
    
    % --- Convolution ---
    % Use the 'same' option to keep the output size identical to the input
    outputMatrix = conv2(img_double, K_Laplace, 'same');
    
end