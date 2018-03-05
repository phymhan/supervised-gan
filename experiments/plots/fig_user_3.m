
src = '../user_study/data/user';

dataset = {'sgan_single', 'param_single'};
acc1_mean = zeros(1, 3);
acc1_err = zeros(1, 3);
acc2_mean = zeros(1, 3);
acc2_err = zeros(1, 3);
for j = 1:numel(dataset)
    dd = dir(fullfile(src, dataset{j}, 'y', '*.mat'));
    N = numel(dd);
    acc = zeros(1, N);
    expert = false(1, N);
    for i = 1:numel(dd)
        s = load(fullfile(src, dataset{j}, 'y', dd(i).name));
        acc(i) = s.num_correct/s.num_total;
        expert(i) = s.expert ~= 0;
    end
    acc1 = acc(expert);
    acc1_mean(j) = mean(acc1);
    acc1_err(j) = std(acc1);
    acc2 = acc(~expert);
    acc2_mean(j) = mean(acc2);
    acc2_err(j) = std(acc2);
end
acc_mean = [acc1_mean', acc2_mean'];
acc_err = [acc1_err', acc2_err'];

tags = {'SGAN', 'Parametric'};
n_group = 2;

figure
hold on
bar(acc1_mean)
errorbar(acc1_mean, acc1_err, 'r.');

for j = 1:n_group
    if acc1_mean(j) > 0.7
        text(j, acc1_mean(j)-acc1_err(j)-0.04, sprintf('%.0f%%', acc1_mean(j)*100), 'HorizontalAlignment', 'center', 'fontsize', 10);
    else
        text(j, acc1_mean(j)+acc1_err(j)+0.04, sprintf('%.0f%%', acc1_mean(j)*100), 'HorizontalAlignment', 'center', 'fontsize', 10);
    end
end

grid on;
box on;
ylabel('Accuracy');
h = gca;
h.XLim = [0.5 n_group+0.5];
% h.YLim = [0 0.8];
h.YLim = [0.5 1];
h.YTick = 0.5:0.1:1;
h.XTickLabel = tags;
% h.XTickLabelRotation = 45;
% set(gcf, 'Position', [1014         318         294         304]);
h.XTickLabelRotation = 37.5;
% set(gcf, 'Position', [1313         362         227         237]);
set(gcf, 'color', [1 1 1]);
set(gca, 'Units', 'pixels');
set(gca, 'Position', [33.3700+20   50.0984+30  192.9750-75*1.5  176.6016])
