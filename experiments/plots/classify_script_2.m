%% load and parse data
clear
load feat_2.mat
datasets = {'train', 'val', 'joint', 'dsgan_detach', 'dsgan', 'unsup', 'param'};
% n_test = 50; % nnz(label == 2);

opt = '-s 0 -c 1';

labelMapping = [1, 1, 2, 3, 4, 5, 6];
tags = {'Trivial', 'Joint', 'SGAN', 'DSGAN', 'Unsup', 'Param'};
y = labelMapping(label)';

X = cat(1, feat{:});
Xr = X(label == 1,:);
mu = mean(Xr, 1);
sigma = std(Xr, 0, 1);
X = bsxfun(@rdivide, bsxfun(@minus, X, mu), sigma);

n_group = numel(tags);

% sample 50 or 100 points for each class
accs = cell(1, n_group);

for ii = 1:100 %%%%%%%%%%%%%%%%%%%%%%%%%
    rng(ii-1);
    X_train = cell(1, n_group);
    X_test = cell(1, n_group);
    y_train = cell(1, n_group);
    y_test = cell(1, n_group);
    for j = 1:n_group
        idx = y == j;
        if nnz(idx) < 200
            n_train = ceil(nnz(idx)*0.6);
        else
            n_train = 100;
        end
        if j == 1
            id2 = 1:n_train;
        else
            id2 = datasample(1:nnz(idx), n_train, 'replace', false);
        end
        idx2 = false(1, nnz(idx)); idx2(id2) = true;
        tmpx = double(X(idx,:));
        tmpy = y(idx,:);
        X_train{j} = tmpx(idx2,:);
        X_test{j} = tmpx(~idx2,:);
        y_train{j} = tmpy(idx2);
        y_test{j} = tmpy(~idx2);
    end
    X_train = cat(1, X_train{:});
    X_test = cat(1, X_test{:});
    y_train = cat(1, y_train{:});
    y_test = cat(1, y_test{:});
    
    models = cell(1, n_group);
    % sample 100 points for each class
    
    for j = 2:n_group
        % train
        idx = y_train == 1 | y_train == j;
        xx = X_train(idx,:);
        yy = y_train(idx);
        models{j} = fitcsvm(xx, yy);
        
        % test
        % idx1 = find(y_test == 1);
        idx2 = find(y_test == j);
        if numel(idx2) < 100
            n_test = numel(idx2);
        else
            n_test = 100;
        end
        %     n_test = numel(idx2);
        % idx1 = datasample(idx1, n_test, 'replace', false);
        idx2 = datasample(idx2, n_test, 'replace', false);
        % idx = [idx1; idx2];
        idx = idx2;
        xx = X_test(idx,:);
        yy = y_test(idx);
        [pred, ~] = predict(models{j}, xx);
        % accs(j) = nnz(pred == yy) / (2 * n_test);
        accs{j}(ii) = nnz(pred == 1) / n_test;
        fprintf('[%s] %.2f\n', tags{j}, accs{j}(end) * 100);
    end
    
end

acc_mean = cellfun(@mean, accs(1:end));
acc_err = cellfun(@std, accs(1:end));

%% re-order
acc_mean = acc_mean([1 4 3 2 5 6]);
acc_err = acc_err([1 4 3 2 5 6]);
tags = tags([1 4 3 2 5 6]);

%%
figure
hold on
bar(acc_mean)
errorbar(acc_mean, acc_err, 'r.');

for j = 2:n_group
    text(j, acc_mean(j)+acc_err(j)+0.02, sprintf('%.0f%%', acc_mean(j)*100), 'HorizontalAlignment', 'center', 'fontsize', 10);
end

grid on;
box on;
ylabel('Realisticness');
h = gca;
h.XLim = [1.5 n_group+0.5];
h.YLim = [0 0.5];
h.XTickLabel = tags(2:end);
% h.XTickLabelRotation = 45;
% set(gcf, 'Position', [1014         318         294         304]);
h.XTickLabelRotation = 37.5;
% set(gcf, 'Position', [1313         362         227         237]);
set(gcf, 'color', [1 1 1]);
set(gca, 'Units', 'pixels');
set(gca, 'Position', [33.3700+30   50.0984+30  192.9750  176.6016])
