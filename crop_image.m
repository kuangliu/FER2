function cropped = crop_image(X, h, w, varargin)
%CROP_IMAGE Crop the given volumn X to generate more training data
% Inputs:
%   - X: input volumn, could be
%       - 2D volumn: gray image [H,W]
%       - 3D volumn: RGB image [H,W,C], C=3
%       - 4D volumn: N images [H,W,C,N]
%   - h, w: cropped size [h, w]
%   - varargin: crop times, if n==0, we return every possible crop
% Output:
%   - cropped: cropped volumn sized [h, w, C, n*N]

[H,W,C,N] = size(X);

deltaX = W-w;
deltaY = H-h;
assert(deltaX>=0 && deltaY>=0);

if nargin == 3 || isempty(varargin)
    % crop every possible position
    n = (deltaX+1)*(deltaY+1);
    cropped = zeros(h,w,C,n*N,'like',X);
    fprintf('you choose iteration crop mode, you will get %dX more data.\n', n);
    i = 1;
    for x = 1:deltaX+1
        for y = 1:deltaY+1
            cropped(:,:,:,(i-1)*N+1:i*N) = X(y:y+h-1, x:x+w-1, :, :);
            i = i + 1;
        end
    end
else
    % random crop
    n = varargin{1};
    cropped = zeros(h,w,C,n*N,'like',X);
    for i = 1:n
        x = randi(deltaX+1);
        y = randi(deltaY+1);
        cropped(:,:,:,(i-1)*N+1:i*N) = X(y:y+h-1, x:x+w-1, :, :);
        
        % figure; imshow(cropped(:,:,:,i));
    end
end
