%List of Operations:

% Exp 1A: Grayscale, Resize, Rotate, Flip, Brightness Adjustment

% Exp 1B: Negation, Bitwise AND, Bitwise OR, Add, Subtract, Multiply,
% Divide, Quantization, Sampling, Adjacency

% Exp 2A: Contrast Stretching, Logarithmic Transformation, Power Law
% transformation with gamma correction, bit level slicing with
% reconstruction, histogram equalisation, intensity level slicing,
% thresholding

%Exp 2B: Filters: Median, Mean, Laplacian, Gaussian, High pass, Low pass

% Exp 3: Sobel, Prewitt and Robert Filters

% Exp 4: Hough transform, Erosion, Dilation, Opening and Closing, Boundary
% Extraction, Hit and Miss Transform

% Exp 5: Lossless Compression, Lossy Compression, ROI based Adaptive
% Compression

%Exp 6: Video Processing Operations: Frame rate, Resolution, Bitrate, Duration,
% File format / Codec, Frame sequence, Brightness, Contrast, Saturation, Hue, Sharpness,
% Color model (RGB, HSV, YCbCr, etc.), Noise level, Gamma, Frame timing, Temporal resolution, 
% Motion vectors / Optical flow, Scene transitions, Sample rate (audio), Bit depth (audio),
% Volume, Audio-video synchronization, GOP structure (Group of Pictures), Quantization level, 
% Chroma subsampling, Cropping, Resizing, Rotation / Flipping, Stabilization,
% Overlay / Watermarking, Color grading, Frame interpolation

%Exp 7: Video Stabilisation, Activity Recognition, Block matching, Object
%Tracking

%% ===================================================================
%% MATLAB IMAGE & VIDEO PROCESSING CHEATSHEET
%% ===================================================================

%% EXPERIMENT 1A: Basic Image Operations
% -----------------------------------------------------------------

% Read Image
img = imread('image.jpg');

% Grayscale Conversion
gray_img = rgb2gray(img);
gray_img = im2gray(img); % Alternative

% Resize
resized = imresize(img, 0.5);           % Scale by factor
resized = imresize(img, [rows cols]);   % Specify size

% Rotate
rotated = imrotate(img, 45);            % Rotate 45 degrees
rotated = imrotate(img, 90, 'crop');    % Rotate and crop

% Flip
flipped_ud = flipud(img);               % Vertical flip
flipped_lr = fliplr(img);               % Horizontal flip

% Brightness Adjustment
bright_img = img + 50;                  % Increase brightness
bright_img = imadjust(img, [], [], 1.5); % Gamma adjustment


%% EXPERIMENT 1B: Arithmetic & Logic Operations
% -----------------------------------------------------------------

% Negation (Complement)
neg_img = imcomplement(img);
neg_img = 255 - img;                    % For uint8

% Bitwise Operations
and_img = bitand(img1, img2);
or_img = bitor(img1, img2);
xor_img = bitxor(img1, img2);
not_img = bitcmp(img);

% Arithmetic Operations
add_img = imadd(img1, img2);
sub_img = imsubtract(img1, img2);
mul_img = immultiply(img1, img2);
div_img = imdivide(img1, img2);

% Quantization
quant_levels = 8;
quant_img = round(double(img) / (256/quant_levels)) * (256/quant_levels);

% Sampling (Downsampling)
sampled = img(1:2:end, 1:2:end);        % Every 2nd pixel

% Adjacency (4-connectivity and 8-connectivity)
conn4 = [0 1 0; 1 1 1; 0 1 0];
conn8 = ones(3,3);


%% EXPERIMENT 2A: Image Enhancement
% -----------------------------------------------------------------

% Contrast Stretching
stretched = imadjust(img, stretchlim(img), []);

% Logarithmic Transformation
c = 1;
log_img = c * log(1 + double(img));
log_img = uint8(255 * mat2gray(log_img));

% Power Law (Gamma) Transformation
gamma = 2.2;
gamma_img = imadjust(img, [], [], gamma);
% Or manual: gamma_img = 255 * ((double(img)/255).^gamma);

% Bit Plane Slicing
bit_plane = bitget(img, 7);             % 7th bit plane
reconstructed = bitset(zeros(size(img)), 7, bit_plane);

% Histogram Equalization
eq_img = histeq(img);
eq_img = histeq(img, 256);              % With 256 bins

% Intensity Level Slicing
lower = 50; upper = 150;
sliced = (img >= lower) & (img <= upper);
sliced_preserve = img .* uint8(sliced); % With background

% Thresholding
level = graythresh(img);                % Otsu's method
bw = imbinarize(img, level);
bw = im2bw(img, 0.5);                   % Manual threshold


%% EXPERIMENT 2B: Filtering
% -----------------------------------------------------------------

% Median Filter
med_filtered = medfilt2(img);
med_filtered = medfilt2(img, [3 3]);    % 3x3 window

% Mean Filter (Average)
h_mean = fspecial('average', [3 3]);
mean_filtered = imfilter(img, h_mean);

% Gaussian Filter
h_gauss = fspecial('gaussian', [5 5], 1.5); % Size, sigma
gauss_filtered = imfilter(img, h_gauss);
gauss_filtered = imgaussfilt(img, 2);   % Sigma = 2

% Laplacian Filter
h_lap = fspecial('laplacian', 0.2);     % Alpha = 0.2
lap_filtered = imfilter(img, h_lap);

% High Pass Filter
h_highpass = [-1 -1 -1; -1 8 -1; -1 -1 -1];
high_filtered = imfilter(img, h_highpass);

% Low Pass Filter
h_lowpass = ones(3,3) / 9;
low_filtered = imfilter(img, h_lowpass);


%% EXPERIMENT 3: Edge Detection
% -----------------------------------------------------------------

% Sobel Filter
sobel_h = fspecial('sobel');            % Horizontal
sobel_v = sobel_h';                     % Vertical
edges_sobel = edge(img, 'sobel');

% Prewitt Filter
prewitt_h = [-1 0 1; -1 0 1; -1 0 1];
prewitt_v = prewitt_h';
edges_prewitt = edge(img, 'prewitt');

% Roberts Filter
roberts_h = [1 0; 0 -1];
roberts_v = [0 1; -1 0];
edges_roberts = edge(img, 'roberts');

% Canny Edge Detection (bonus)
edges_canny = edge(img, 'canny');


%% EXPERIMENT 4: Morphological Operations & Hough Transform
% -----------------------------------------------------------------

% Create Structuring Element
se = strel('disk', 5);
se = strel('square', 3);
se = strel('line', 10, 45);

% Erosion
eroded = imerode(img, se);

% Dilation
dilated = imdilate(img, se);

% Opening (Erosion followed by Dilation)
opened = imopen(img, se);

% Closing (Dilation followed by Erosion)
closed = imclose(img, se);

% Boundary Extraction
boundary = img - imerode(img, se);

% Hit-or-Miss Transform
hmt = bwhitmiss(img, se1, se2);

% Hough Transform (for line detection)
bw = edge(img, 'canny');
[H, theta, rho] = hough(bw);
peaks = houghpeaks(H, 5);
lines = houghlines(bw, theta, rho, peaks);


%% EXPERIMENT 5: Image Compression
% -----------------------------------------------------------------

% Lossless Compression (save as PNG)
imwrite(img, 'output.png');             % PNG is lossless

% Lossy Compression (JPEG with quality)
imwrite(img, 'output.jpg', 'Quality', 75); % Quality: 0-100

% ROI-Based Adaptive Compression
roi_mask = false(size(img));
roi_mask(100:200, 100:200) = true;      % Define ROI
roi_img = img;
roi_img(~roi_mask) = roi_img(~roi_mask) / 2; % Reduce quality outside ROI
imwrite(roi_img, 'output_roi.jpg', 'Quality', 90);


%% EXPERIMENT 6: Video Processing Basics
% -----------------------------------------------------------------

% Read Video
v = VideoReader('video.mp4');

% Get Video Properties
fps = v.FrameRate;                      % Frame rate
duration = v.Duration;                  % Duration in seconds
width = v.Width;                        % Width
height = v.Height;                      % Height
num_frames = v.NumFrames;               % Total frames

% Read Frames
frame = readFrame(v);                   % Read one frame
all_frames = read(v, [1 Inf]);         % Read all frames

% Create Video Writer
w = VideoWriter('output.mp4', 'MPEG-4');
w.FrameRate = 30;
w.Quality = 95;
open(w);

% Write Frames
writeVideo(w, frame);

% Close Video Writer
close(w);

% Adjust Frame Brightness
bright_frame = frame + 30;

% Adjust Contrast
contrast_frame = imadjust(frame);

% Adjust Saturation (HSV)
hsv = rgb2hsv(frame);
hsv(:,:,2) = hsv(:,:,2) * 1.5;         % Increase saturation
saturated = hsv2rgb(hsv);

% Adjust Hue
hsv(:,:,1) = hsv(:,:,1) + 0.1;         % Shift hue

% Color Model Conversions
hsv_frame = rgb2hsv(frame);
ycbcr_frame = rgb2ycbcr(frame);
lab_frame = rgb2lab(frame);
gray_frame = rgb2gray(frame);

% Add Noise
noisy = imnoise(frame, 'gaussian', 0, 0.01);
noisy = imnoise(frame, 'salt & pepper', 0.05);

% Gamma Correction
gamma_frame = imadjust(frame, [], [], 2.2);

% Cropping
cropped = imcrop(frame, [x y width height]);

% Overlay/Watermark
watermark = imread('logo.png');
blended = imfuse(frame, watermark, 'blend');

% Frame Interpolation
interp_frame = imresize(frame, 2, 'bicubic');


%% EXPERIMENT 7: Advanced Video Processing
% -----------------------------------------------------------------

% Video Stabilization
stabilizer = VideoStabilizer('Method', 'ECC');
reset(stabilizer);
while hasFrame(v)
    frame = readFrame(v);
    stabilized = step(stabilizer, frame);
end

% Optical Flow (Motion Vectors)
opticFlow = opticalFlowHS;
flow = estimateFlow(opticFlow, gray_frame);
% Or use opticalFlowFarneback, opticalFlowLK

% Block Matching
blockSize = [16 16];
mbType = 'Full';
motion_vectors = motionEstTSS(ref_frame, curr_frame, blockSize);

% Object Tracking - KLT Tracker
tracker = vision.PointTracker('MaxBidirectionalError', 2);
initialize(tracker, points, frame);
[points, validity] = step(tracker, next_frame);

% Object Tracking - Blob Analysis
detector = vision.ForegroundDetector('NumGaussians', 3);
blob = vision.BlobAnalysis('BoundingBoxOutputPort', true);
mask = step(detector, frame);
bbox = step(blob, mask);

% Activity Recognition (using detectPeopleACF or similar)
peopleDetector = peopleDetectorACF('caltech');
[bboxes, scores] = detect(peopleDetector, frame);

% Scene Change Detection
scene_change = sum(abs(frame1(:) - frame2(:))) > threshold;


%% UTILITY FUNCTIONS
% -----------------------------------------------------------------

% Display Image
imshow(img);
imshow(img, []);                        % Auto scale

% Display Multiple Images
imshowpair(img1, img2, 'montage');
montage({img1, img2, img3});

% Display Histogram
imhist(img);
histogram(img(:));

% Save Image
imwrite(img, 'output.jpg');

% Image Information
info = imfinfo('image.jpg');

% Convert Data Types
double_img = im2double(img);
uint8_img = im2uint8(img);
single_img = im2single(img);

%% ===================================================================
%% END OF CHEATSHEET
%% ===================================================================