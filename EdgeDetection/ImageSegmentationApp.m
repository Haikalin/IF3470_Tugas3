function ImageSegmentationApp()
% IMAGESEGMENTATIONAPP A simple GUI for comparing image segmentation techniques.
    % --- 1. GUI Setup ---
    
    % Ensure the existing figures are cleared for a clean run
    close all force; 
    
    % Create the main figure window, setting units to NORMALIZED for responsiveness
    fig = figure('Name', 'Simple Image Segmentation Viewer (3-Axis)', ...
                 'NumberTitle', 'off', ...
                 'Units', 'normalized', ... % CRITICAL FOR RESPONSIVENESS
                 'Position', [0.1 0.1 0.8 0.8], ... % Start size (80% of screen)
                 'ToolBar', 'none', ...
                 'MenuBar', 'none');
    
    % Store application data in the figure handle for easy access
    appData = struct();
    appData.OriginalImage = [];
    appData.GrayImage = [];
    appData.SegmentationMethods = {'Original', 'Laplace', 'LoG', 'Sobel', 'Prewitt', 'Roberts', 'Canny'}; 
    set(fig, 'UserData', appData);
    
    % --- 2. UI Controls (Menu, Buttons, and Sigma Input) ---
    controlPanel = uipanel(fig, 'Title', 'Controls', ...
                                'Units', 'normalized', ...
                                'Position', [0.1 0.05 0.8 0.1]); 
    
    X_START = 0.19;
    SPACE = 0.01; % 1% normalized spacing
    
    % 1. Select Technique Label
    uicontrol(controlPanel, 'Style', 'text', ...
                          'String', 'Select Technique:', ...
                          'Units', 'normalized', ...
                          'Position', [X_START 0.3 0.12 0.5], ...
                          'HorizontalAlignment', 'left');
    X_POS = X_START + 0.12 + SPACE;
                          
    % 2. Method Menu
    appData.MethodMenu = uicontrol(controlPanel, 'Style', 'popupmenu', ...
                                      'String', appData.SegmentationMethods, ...
                                      'Value', 1, ...
                                      'Units', 'normalized', ...
                                      'Position', [X_POS 0.3 0.15 0.5], ...
                                      'Callback', @(src, event) updateDisplay(fig));
    X_POS = X_POS + 0.15 + SPACE;
                                  
    % 3. Sigma Label (Initially Hidden)
    appData.SigmaLabel = uicontrol(controlPanel, 'Style', 'text', ...
                          'String', 'LoG/Canny Sigma:', ...
                          'Units', 'normalized', ...
                          'Position', [X_POS 0.3 0.15 0.5], ...
                          'HorizontalAlignment', 'left', ...
                          'Visible', 'off'); 
    X_POS = X_POS + 0.15 + SPACE;
                      
    % 4. Sigma Input (Initially Hidden)
    appData.SigmaInput = uicontrol(controlPanel, 'Style', 'edit', ...
                                   'String', '2.0', ...
                                   'Units', 'normalized', ...
                                   'Position', [X_POS 0.3 0.05 0.5], ...
                                   'Callback', @(src, event) updateDisplay(fig), ...
                                   'Visible', 'off'); 
    X_POS = X_POS + 0.05 + SPACE*2; 
                                  
    % 5. Load Image Button
    uicontrol(controlPanel, 'Style', 'pushbutton', ...
                          'String', 'Load New Image', ...
                          'Units', 'normalized', ...
                          'Position', [X_POS 0.3 0.15 0.5], ...
                          'Callback', @(src, event) loadImage(fig));
    
    % --- 3. Axes for Image Display (Three Axes) ---
    
    % 1. Original Image (Left)
    appData.AxOriginal = axes(fig, 'Position', [0.03 0.20 0.30 0.75]);
    title(appData.AxOriginal, '1. Original Image');
    
    % 2. Grayscale/Magnitude Result (Middle)
    appData.AxGrayscaleResult = axes(fig, 'Position', [0.35 0.20 0.30 0.75]);
    title(appData.AxGrayscaleResult, '2. Grayscale Result');
    
    % 3. Color Segmented/Masked Image (Right)
    appData.AxColorMasked = axes(fig, 'Position', [0.67 0.20 0.30 0.75]);
    title(appData.AxColorMasked, '3. Color Masked Area');
    
    % Update UserData with handles
    set(fig, 'UserData', appData);
    
    % --- 4. Initialization ---
    loadDefaultImage(fig);
    
end

function loadImage(fig)
% Load a new image file specified by the user
    [filename, pathname] = uigetfile({'*.jpg;*.png;*.tif', 'Image Files (*.jpg, *.png, *.tif)'}, 'Select an Image');
    
    if filename == 0
        % User cancelled the selection
        return;
    end
    
    fullFilePath = fullfile(pathname, filename);
    img = imread(fullFilePath);
    
    appData = get(fig, 'UserData');
    
    % Convert to grayscale if it's RGB
    if size(img, 3) == 3
        appData.GrayImage = im2double(rgb2gray(img));
        appData.OriginalImage = img; % Keep RGB for original display
    else
        appData.GrayImage = im2double(img);
        appData.OriginalImage = img;
    end
    
    % Update figure data and refresh display
    set(fig, 'UserData', appData);
    updateDisplay(fig);
    
end

function loadDefaultImage(fig)
% Load a built-in image ('peppers') for quick startup
    % Load the image (it's RGB)
    img = imread('peppers.png');
    
    % Convert to double and grayscale
    appData = get(fig, 'UserData');
    appData.OriginalImage = img;
    appData.GrayImage = im2double(rgb2gray(img));
    
    % Update figure data and refresh display
    set(fig, 'UserData', appData);
    updateDisplay(fig);
end

function colorMasked = createColorMaskedImage(rgbImg, binaryMask)
% createColorMaskedImage Returns the RGB image where non-masked areas are black.
% binaryMask is TRUE for the object(s) to keep, FALSE for the background to set to black.
    
    % Ensure binaryMask is logical
    binaryMask = logical(binaryMask);
    
    % Initialize black image of same class as input
    colorMasked = zeros(size(rgbImg), 'like', rgbImg);
    
    % Apply mask to each channel
    for k = 1:size(rgbImg, 3)
        channel = rgbImg(:, :, k);
        % Set background pixels to 0 (black)
        channel(~binaryMask) = 0;
        colorMasked(:, :, k) = channel;
    end
end

function updateDisplay(fig)
    
    appData = get(fig, 'UserData');
    imgGray = appData.GrayImage;
    if isempty(imgGray), return; end
    
    sigma = str2double(get(appData.SigmaInput, 'String'));
    if isnan(sigma) || sigma <= 0, sigma = 1.0; end
    selectedIdx = get(appData.MethodMenu, 'Value');
    selectedMethod = appData.SegmentationMethods{selectedIdx};
    
    % Toggle UI visibility
    isLineMethod = any(strcmp(selectedMethod, {'LoG', 'Canny'}));
    visState = 'off'; if isLineMethod, visState = 'on'; end
    set(appData.SigmaLabel, 'Visible', visState); 
    set(appData.SigmaInput, 'Visible', visState);
    
    % --- 1. Edge Detection (The Outline) ---
    switch selectedMethod
        case 'Original', rawResult = imgGray;
        case 'Laplace', rawResult = Laplace(imgGray); 
        case 'LoG', rawResult = LoG(imgGray, sigma); 
        case 'Canny', rawResult = edge(imgGray, 'canny', [], sigma); 
        case {'Sobel', 'Prewitt', 'Roberts'}, rawResult = feval(selectedMethod, imgGray);
        otherwise, rawResult = imgGray;
    end
    
    % --- 2. Process Mask (The Separation Logic) ---
    if strcmp(selectedMethod, 'Original')
        binaryMask = true(size(imgGray));
        grayscaleResult = imgGray;
        
    % --- SPECIAL HANDLING FOR SOBEL / PREWITT ---
    elseif any(strcmp(selectedMethod, {'Sobel', 'Prewitt'}))
        
        % 1. Normalize
        normResult = rawResult / max(rawResult(:));
        
        % 2. Adaptive Threshold (Otsu's Method)
        % Automatic threshold is usually better for Sobel than hardcoded 0.1
        level = graythresh(normResult); 
        % Add a small offset to reduce noise (optional tuning)
        mask = imbinarize(normResult, level * 0.7); 
        
        % 3. CONNECT GAPS (The Critical Step)
        % Instead of thinning, we DILATE (thicken) to close broken edges
        se_bridge = strel('disk', 3);
        mask_thick = imdilate(mask, se_bridge);
        
        % 4. Fill Holes
        mask_filled = imfill(mask_thick, 'holes');
        
        % 5. Restore Shape (Erosion)
        % We thickened the object to fill it, now we slim it back down
        binaryMask = imerode(mask_filled, se_bridge);
        
        % For display purposes in the middle axis, show the thick edges
        grayscaleResult = mask; 

    % --- HANDLING FOR CANNY / LOG / ROBERTS / LAPLACE ---
    else
        if isLineMethod && islogical(rawResult) % Canny
            edges = rawResult;
        elseif isLineMethod % Laplace & LoG
            edges = FindZeroCrossings(rawResult);
        else % Roberts
            % Roberts is very sensitive to noise, keep high threshold
            edges = imbinarize(rawResult / max(rawResult(:)), 0.15); 
            edges = bwmorph(edges, 'thin', Inf); % Thinning is okay for Roberts
            edges = bwmorph(edges, 'spur', 2);
        end
        
        % Standard Morphology for sharp edge detectors
        se = strel('disk', 2); 
        closedEdges = imclose(edges, se); 
        binaryMask = imfill(closedEdges, 'holes');
        
        grayscaleResult = edges;
    end

    % --- 3. INTELLIGENT FILTERING (Common for all) ---
    % Clean up small noise blobs for the final mask
    if ~strcmp(selectedMethod, 'Original')
        cc = bwconncomp(binaryMask);
        stats = regionprops(cc, 'Area');
        
        if isempty(stats)
            binaryMask = false(size(imgGray));
        else
            allAreas = [stats.Area];
            maxArea = max(allAreas);
            % Keep objects that are at least 10% of the largest object
            relativeThreshold = 0.10 * maxArea; 
            
            cleanMask = false(size(imgGray));
            for k = 1:length(stats)
                if stats(k).Area >= relativeThreshold
                    cleanMask(cc.PixelIdxList{k}) = true;
                end
            end
            binaryMask = cleanMask;
        end
    end
    
    % --- 3. Apply Color ---
    if size(appData.OriginalImage, 3) == 3
        colorResult = createColorMaskedImage(appData.OriginalImage, binaryMask);
    else
        colorResult = double(appData.OriginalImage) .* double(binaryMask);
    end
    
    % --- 4. Display ---
    imshow(appData.OriginalImage, 'Parent', appData.AxOriginal);
    title(appData.AxOriginal, 'Original');
    
    imshow(grayscaleResult, [], 'Parent', appData.AxGrayscaleResult);
    title(appData.AxGrayscaleResult, ['Edges: ' selectedMethod]);
    
    imshow(colorResult, [], 'Parent', appData.AxColorMasked);
    title(appData.AxColorMasked, 'Segmented Objects');
end