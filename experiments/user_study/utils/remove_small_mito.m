src = '../data/dsgan/fake_single';

dd = dir(fullfile(src, '*.png'));

for i = 1:numel(dd)
    A = imread(fullfile(src, dd(i).name));
    m = A(:,:,2);
    if nnz(m) < 64
        delete(fullfile(src, dd(i).name));
    end
    fprintf('--> %s\n', dd(i).name);
end
