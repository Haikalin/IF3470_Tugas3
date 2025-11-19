function [edgeMap] = FindZeroCrossings(convolutionMatrix, threshold)
%FindZeroCrossings Detects edges by finding sign changes in a second-derivative matrix.
%   
%   Inputs:
%       convolutionMatrix - The result matrix from Laplace or LoG.
%       threshold         - Optional magnitude threshold (default 0).
    
    if nargin < 2
        threshold = 0; % Default to no threshold
    end

    [R, C] = size(convolutionMatrix);
    edgeMap = false(R, C);
    
    % --- Core Zero-Crossing Detection Logic ---
    
    % Check Horizontal Zero-Crossings
    for r = 1:R
        for c = 1:C-1
            p1 = convolutionMatrix(r, c);
            p2 = convolutionMatrix(r, c+1);
            
            % Check for sign change AND check if magnitude difference exceeds threshold
            if (p1 * p2 < 0) && (abs(p1) + abs(p2) > threshold)
                % Mark the pixel with the smaller magnitude (closer to zero)
                if abs(p1) < abs(p2)
                    edgeMap(r, c) = true;
                else
                    edgeMap(r, c+1) = true;
                end
            end
        end
    end

    % Check Vertical Zero-Crossings
    for r = 1:R-1
        for c = 1:C
            p1 = convolutionMatrix(r, c);
            p2 = convolutionMatrix(r+1, c);
            
            % Check for sign change AND check if magnitude difference exceeds threshold
            if (p1 * p2 < 0) && (abs(p1) + abs(p2) > threshold)
                % Mark the pixel with the smaller magnitude
                if abs(p1) < abs(p2)
                    edgeMap(r, c) = true;
                else
                    edgeMap(r+1, c) = true;
                end
            end
        end
    end
end