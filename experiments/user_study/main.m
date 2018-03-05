function main(mode, celltype, dataset, seed)
global sz num_total num_train data_mode dataset_ celltype_
sz = [512 512];
num_total = nan;
num_train = 10; % number of images used to train user
if ~exist('seed', 'var') || isempty(seed), seed = 0; end
if ~exist('celltype', 'var') || isempty(celltype), celltype = ''; end
if ~exist('dataset', 'var') || isempty(dataset), dataset = 'sgan'; end
if ~exist('mode', 'var') || isempty(mode), mode = 'x'; end
if ~isempty(celltype) && celltype(1) ~= '_', celltype = ['_' celltype]; end
data_mode = mode;
dataset_ = dataset;
celltype_ = celltype;
if strcmpi(celltype_, '_single')
    sz = [1024 1024];
end
rng(seed);
if ~exist(['data/user/' dataset_ celltype_ '/' data_mode], 'dir')
    mkdir(['data/user/' dataset_ celltype_ '/' data_mode]);
end

%% real and fake samples
real_sample = sample_patches('real', 4, sz, 1, 'train');
fake_sample = sample_patches('fake', 4, sz, 1, 'train');
real_sample = cat(2, real_sample{:});
fake_sample = cat(2, fake_sample{:});

%% figure
global h_train
hf = figure('name', 'main');
hf.Position = [568   226   1000   600];
ha1 = axes('position', [0.05 0.8 0.4 0.2], 'title', 'real samples', ...
    'buttonDownFcn', 'disp(''yes'')');
ha2 = axes('position', [0.55 0.8 0.4 0.2], 'title', 'fake samples');
uicontrol('style', 'text', 'string', 'Real', 'units', 'normalized', ...
    'position', [0. 0.8 0.05 0.1]);
uicontrol('style', 'text', 'string', 'Fake', 'units', 'normalized', ...
    'position', [0.5 0.8 0.05 0.1]);
h_train = cell(1, 2);
h_train{1} = imshow(real_sample, 'parent', ha1);
h_train{2} = imshow(fake_sample, 'parent', ha2);

global h_image h_check h_txt num_test
global user_data started
started = 0;
num_test = 0;
h_axes = cell(1, 9);
h_check = cell(1, 9);
h_image = cell(1, 9);
pos = { ...
    [0.15 0.55 0.2 0.2], ...
    [0.15 0.30 0.2 0.2], ...
    [0.15 0.05 0.2 0.2], ...
    [0.40 0.55 0.2 0.2], ...
    [0.40 0.30 0.2 0.2], ...
    [0.40 0.05 0.2 0.2], ...
    [0.65 0.55 0.2 0.2], ...
    [0.65 0.30 0.2 0.2], ...
    [0.65 0.05 0.2 0.2], ...
    };
if strcmpi(data_mode, 'xy')
    sz_ = [sz(1) sz(2)+sz(2) 3];
else
    sz_ = [sz(1:2) 3];
end
for i = 1:9
    h_axes{i} = axes('parent', hf, 'position', pos{i});
    h_image{i} = imshow(zeros(sz_, 'uint8'), 'parent', h_axes{i});
    h_check{i} = uicontrol('parent', hf, 'style', 'checkbox', 'units', 'normalized', ...
        'position', pos{i}+[-0.025 0.1-0.0125 0.0125-0.2 0.0125-0.2]);
end

uicontrol('style', 'pushbutton', 'string', 'Start', 'callback', @start_fcn, ...
    'units', 'normalized', 'position', [0.05 0.05 0.05 0.05]);
uicontrol('style', 'pushbutton', 'string', 'Next', 'callback', @next_fcn, ...
    'units', 'normalized', 'position', [0.05 0.15 0.05 0.05], 'units', 'normalized');
uicontrol('style', 'pushbutton', 'string', 'Save', 'callback', @save_fcn, ...
    'units', 'normalized', 'position', [0.05 0.25 0.05 0.05], 'units', 'normalized');
h_txt = uicontrol('style', 'text', 'string', 'finished: 0', 'units', 'normalized', ...
    'position', [0.05 0.35 0.05 0.05]);

user_data.id = 'xyz';
user_data.num_total = 0;
user_data.num_correct = 0;
user_data.vector_gt = [];
user_data.vector_gs = [];
user_data.expert = 0;

fprintf('check the ones you think are FAKE.\n')

x = inputdlg({'Andrew ID:', 'Expert? (0/1)'}, 'User Info', [1 10; 1 10], {'outlier', '1'});
user_data.id = x{1};
user_data.expert = str2num(x{2});

%% sample patches
function Is = sample_patches(label, N, sz, add_boarder, set)
global data_mode num_train dataset_ celltype_
Is = cell(1, N);
if strcmpi(label, 'real')
    src = ['./data/real/real' celltype_];
else
    src = ['./data/' dataset_ '/' 'fake' celltype_];
end
dd = dir(fullfile(src, sprintf('*_label.png')));
names = {dd.name};
num_total = numel(names);
if strcmpi(set, 'train')
    ids = 1:num_train;
else
    ids = num_train+1:num_total;
end
ids = datasample(ids, N);
names = names(ids);
names = cellfun(@(s) s(1:4), names, 'UniformOutput', false);

for i = 1:N
    if strcmpi(data_mode, 'x')
        A = imread(fullfile(src, [names{i} '_image.png']));
        if size(A, 3) == 1, A = repmat(A, [1 1 3]); end
        if rand < 0.5
            A = fliplr(A);
        end
        A = rot90(A, randi([0 3], 1));
        sz_A = [size(A,1) size(A,2)];
        y = randi([0 sz_A(1)-sz(1)], 1);
        x = randi([0 sz_A(2)-sz(2)], 1);
        if add_boarder
            Is{i} = padarray(A(1+y:sz(1)+y,1+x:sz(2)+x,:), [2 2 0], 255);
        else
            Is{i} = A(1+y:sz(1)+y,1+x:sz(2)+x,:);
        end
    elseif strcmpi(data_mode, 'y')
        A = imread(fullfile(src, [names{i} '_label.png']));
        if size(A, 3) == 1, A = repmat(A, [1 1 3]); end
        if rand < 0.5
            A = fliplr(A);
        end
        A = rot90(A, randi([0 3], 1));
        sz_A = [size(A,1) size(A,2)];
        y = randi([0 sz_A(1)-sz(1)], 1);
        x = randi([0 sz_A(2)-sz(2)], 1);
        if add_boarder
            Is{i} = padarray(A(1+y:sz(1)+y,1+x:sz(2)+x,:), [2 2 0], 255);
        else
            Is{i} = A(1+y:sz(1)+y,1+x:sz(2)+x,:);
        end
    else
        A = imread(fullfile(src, [names{i} '_label.png']));
        B = imread(fullfile(src, [names{i} '_image.png']));
        if size(A, 3) == 1, A = repmat(A, [1 1 3]); end
        if size(B, 3) == 1, B = repmat(B, [1 1 3]); end
        if rand < 0.5
            A = fliplr(A);
            B = fliplr(B);
        end
        r = randi([0 3], 1);
        A = rot90(A, r);
        B = rot90(B, r);
        sz_A = [size(A,1) size(A,2)];
        y = randi([0 sz_A(1)-sz(1)], 1);
        x = randi([0 sz_A(2)-sz(2)], 1);
        A_ = A(1+y:sz(1)+y,1+x:sz(2)+x,:);
        B_ = B(1+y:sz(1)+y,1+x:sz(2)+x,:);
        if add_boarder
            Is{i} = padarray(cat(2, A_, B_), [2 2 0], 255);
        else
            Is{i} = cat(2, A_, B_);
        end
    end
end

%% refresh
function refresh(gt, sz)
% gt: 0 real, 1 fake
global h_check h_image
for i = 1:9
    if gt(i)
        label = 'fake';
    else
        label = 'real';
    end
    I = sample_patches(label, 1, sz, 0, 'test');
    h_image{i}.CData = I{1};
    h_check{i}.Value = 0;
end

global h_train data_mode
if strcmp(data_mode, 'xy')
    ns = 3;
else
    ns = 4;
end
real_sample = sample_patches('real', ns, sz, 1, 'train');
fake_sample = sample_patches('fake', ns, sz, 1, 'train');
real_sample = cat(2, real_sample{:});
fake_sample = cat(2, fake_sample{:});
h_train{1}.CData = real_sample;
h_train{2}.CData = fake_sample;

%% start
function start_fcn(varargin)
global curr_gt sz started
if started
    return
else
    started = 1;
end
fprintf('start\n')
gt = randi([0 1], 1, 9);
curr_gt = gt;
refresh(gt, sz)

%% save
function save_fcn(varargin)
global user_data data_mode dataset_ celltype_
% save(fullfile(sprintf('%s.mat', user_data.id)), '-struct', 'user_data');
save(fullfile('./', 'data', 'user', [dataset_ celltype_], data_mode, ...
    sprintf('%s.mat', user_data.id)), '-struct', 'user_data');
% fprintf('Finished.\nThank you %s! Your accuracy is: %d/%d %.2f%%\n', ...
%     user_data.id, user_data.num_correct, user_data.num_total, ...
%     user_data.num_correct/user_data.num_total*100);
fprintf('Your accuracy is: %d/%d %.2f%%\n', ...
    user_data.num_correct, user_data.num_total, ...
    user_data.num_correct/user_data.num_total*100);

%% next
function next_fcn(varargin)
% eval current
global curr_gt h_check user_data h_txt num_test
pred = zeros(1, 9);
for i = 1:9
    pred(i) = h_check{i}.Value;
end
user_data.num_total = user_data.num_total + 9;
user_data.num_correct = user_data.num_correct + nnz(pred==curr_gt);
user_data.vector_gt = [user_data.vector_gt curr_gt];
user_data.vector_gs = [user_data.vector_gs pred];
num_test = num_test + 1;
h_txt.String = sprintf('finished: %d', num_test);

% next
global sz
gt = randi([0 1], 1, 9);
curr_gt = gt;
refresh(gt, sz)

fprintf('finished %d\n', num_test);
save_fcn() % auto save
