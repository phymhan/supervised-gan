src = '../data/unsup/fake_single';

gap = 4;
width = 3;
center = [512 512];

dd = dir(fullfile(src, '*.png'));

for i = 1:numel(dd)
    A = imread(fullfile(src, dd(i).name));
    m = A(:,:,1);
    m = bwselect(~imdilate(m, strel('disk', gap)), center(2), center(1));
    m = imdilate(m, strel('disk', width));
    m = imdilate(edge(m), strel('disk', 4));
    A(:,:,1) = m * 255;
    imwrite(A, fullfile(src, dd(i).name));
    fprintf('--> %s\n', dd(i).name);
end
