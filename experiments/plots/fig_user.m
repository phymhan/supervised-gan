%% one plot (x) with expert and non-export

src = '../user_study/data/user/sgan';
mode = 'x';

dd = dir(fullfile(src, mode, '*.mat'));
N = numel(dd);
acc = zeros(1, N);
expert = false(1, N);
for i = 1:numel(dd)
    s = load(fullfile(src, mode, dd(i).name));
    acc(i) = s.num_correct/s.num_total;
    expert(i) = s.expert ~= 0;
end

% accs = {acc(~expert), acc(expert)};
% tags = {'Non-expert', 'Expert'};
accs = {acc(expert), acc(~expert)};
tags = {'Expert', 'Non-Expert'};
n_group = 2;

figure
hold on
acc_mean = cellfun(@mean, accs);
acc_err = cellfun(@std, accs);
bar(acc_mean)
errorbar(acc_mean, acc_err, 'r.');

for j = 1:n_group
    if acc_mean(j) > 0.7
        text(j, acc_mean(j)-acc_err(j)-0.04, sprintf('%.0f%%', acc_mean(j)*100), 'HorizontalAlignment', 'center', 'fontsize', 10);
    else
        text(j, acc_mean(j)+acc_err(j)+0.04, sprintf('%.0f%%', acc_mean(j)*100), 'HorizontalAlignment', 'center', 'fontsize', 10);
    end
end

grid on;
box on;
ylabel('Accuracy');
h = gca;
h.XLim = [0.5 n_group+0.5];
% h.YLim = [0 0.8];
h.YLim = [0.3 0.9];
h.YTick = 0.3:0.1:0.9;
h.XTickLabel = tags;
% h.XTickLabelRotation = 45;
% set(gcf, 'Position', [1014         318         294         304]);
h.XTickLabelRotation = 37.5;
% set(gcf, 'Position', [1313         362         227         237]);
set(gcf, 'color', [1 1 1]);
set(gca, 'Units', 'pixels');
set(gca, 'Position', [33.3700+20   50.0984+30  192.9750-75*1.5  176.6016])
