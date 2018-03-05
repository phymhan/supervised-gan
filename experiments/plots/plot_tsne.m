clear
load feat.mat

tags = {'real (train)', 'real (val)', 'FCGAN', 'non-MS', 'non-FC'};

n_group = numel(tags);

X = cat(1, feat{:});
Xr = X(label == 1,:);
mu = mean(Xr, 1);
sigma = std(Xr, 0, 1);
X = bsxfun(@rdivide, bsxfun(@minus, X, mu), sigma);
X_labels = label';

idx = randperm(size(X, 1));
X = X(idx,:);
X_labels = X_labels(idx,:);

rng(0);
Y = tsne(X);
figure;
hold on
ids = (1:n_group);
for i = ids
    idx = X_labels == i;
    idx = find(idx); idx = idx(randperm(numel(idx), min(numel(idx), 100)));
    % scatter(mappedX(idx,1), mappedX(idx,2), styles{i});
    if i == 1
        scatter(Y(idx,1), Y(idx,2), 'r', 'filled', 'markerfacealpha', 0.8);
    elseif i == 2
        scatter(Y(idx,1), Y(idx,2), 'b', 'filled', 'markerfacealpha', 0.8);
    else
        scatter(Y(idx,1), Y(idx,2), 'filled', 'markerfacealpha', 0.8);
    end
    % scatter3(mappedX(idx,1), mappedX(idx,2), mappedX(idx,3), styles{i});
end
% title('t-SNE');
hl = legend(tags(ids), 'Location', 'Southwest');
% set(hl, 'color', 'none')
grid on;
box on;
set(gcf, 'Position', [1038         267         450         425])
set(gcf, 'color', [1 1 1]);
h = gca;
h.XLim = [-50 50];
h.YLim = [-25 25];
h.XTick = -50:20:50;
h.YTick = -25:10:25;
