clear
load feat_2.mat

tags = {'Non-Param', 'real (val)', 'Joint', 'SGAN', 'DSGAN', 'Unsup', 'Param'};

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

%% t-SNE fill
rng(0);

Y = tsne(X);
% mappedX = tsne(X, [], 2, 30, 30);
% mappedX = X(:, [31 38]);

% gscatter(mappedX(:,1), mappedX(:,2), X_labels);

figure;
hold on
ids = [1 5 4 3 6 7];
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
hl = legend(tags(ids), 'Location', 'Southeast');
% set(hl, 'color', 'none')
grid on;
box on;
set(gcf, 'Position', [1038         267         400         380])
set(gcf, 'color', [1 1 1]);
h = gca;
% h.XLim = [-50 50];
% h.YLim = [-40 40];
h.XTick = -50:25:50;
h.YTick = -40:20:40;
% h.XLim = [-40 40];
% h.YLim = [-50 30];
% h.XTick = -40:20:40;
% h.YTick = -50:20:30;
