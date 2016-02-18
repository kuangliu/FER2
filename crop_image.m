function sub_imgs = crop_image(img, h, w, n)
%CROP_IMAGE Crop the given image to generate more training dataset
% Inputs:
%   - img: given image sized [H, W, C]
%   - h, w: cropped size [h, w]
%   - n: cropped number
% Output:
%   - sub_imgs: cropped images sized [h, w, C, n]

assert(ndims(img)==3);

[H, W, C] = size(img);

deltaX = W-w;
deltaY = H-h;
assert(deltaX>=0 && deltaY>=0);

sub_imgs = zeros(h,w,C,n,'like',img);
for i = 1:n
    x = randi(deltaX);
    y = randi(deltaY);
    sub_imgs(:,:,:,i) = img(y:y+h-1, x:x+w-1, :); 
    % figure; imshow(sub_imgs(:,:,:,i));
end

