function [] = plot_eq()
%% Plot OMP 2th, 4th, 8th, default

results_OMP_2th = csvread('./csv_data/OMP_exec_time_2th.csv')/1000;

results_OMP_4th = csvread('./csv_data/OMP_exec_time_4th.csv')/1000;

results_OMP_8th = csvread('./csv_data/OMP_exec_time_8th.csv')/1000;

results_OMP_def = csvread('./csv_data/OMP_exec_time_default.csv')/1000;

figure(1);
%x_axis = log10([50,500,10000,50000,100000,150000]);
x_axis = [0.01,0.03,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1];
plot(x_axis,results_OMP_2th,'.-','MarkerSize',20,'LineWidth',1);
title('OMP results')
xticks(x_axis);
xticklabels({'0.01','0.03','0.05','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1'});
xlabel('Pixel');
ylabel('Time (s)');
hold on;
plot(x_axis,results_OMP_4th,'.-','MarkerSize',20,'LineWidth',1);
plot(x_axis,results_OMP_8th,'.-','MarkerSize',20,'LineWidth',1);
plot(x_axis,results_OMP_def,'.-','MarkerSize',20,'LineWidth',1);
legend({'2 thread','4 thread','8 thread','Default'},'FontSize',12);

%% Plot CUDA

results_CUDA = csvread('./csv_data/CUDA_exec_time.csv')/1000;
figure(2);
x_axis = [0.01,0.03,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1];
plot(x_axis,results_CUDA,'.-','MarkerSize',20,'LineWidth',1);
title('CUDA results')
xticks(x_axis);
xticklabels({'0.01','0.03','0.05','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1'});
xlabel('Pixel');
ylabel('Time (s)');
legend({'Cuda'},'FontSize',12);

% Sequential, OMP default, CUDA

results_seq = csvread('./csv_data/sequential_exec_time.csv')/1000;
figure(3);
x_axis = [0.01,0.03,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1];
plot(x_axis,results_seq,'.-','MarkerSize',20,'LineWidth',1);
title('Global results')
xticks(x_axis);
xticklabels({'0.01','0.03','0.05','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1'});
xlabel('Pixel');
ylabel('Time (s)');
hold on;
plot(x_axis,results_OMP_def,'.-','MarkerSize',20,'LineWidth',1);
plot(x_axis,results_CUDA,'.-','MarkerSize',20,'LineWidth',1);
legend({'Sequential','OMP default','CUDA'},'FontSize',12);

% Sequential, OMP default, CUDA per piccoli valori
x_axis2 = [0.01,0.03,0.05,0.1,0.2];
figure(4);
plot(x_axis2,[results_seq(1),results_seq(2),results_seq(3),results_seq(4),results_seq(5)],'.-','MarkerSize',20,'LineWidth',1);
title('Global results for little pixel')
xticks(x_axis2);
xticklabels({'0.01','0.03','0.05','0.1','0.2'});
xlabel('Pixel');
ylabel('Time (s)');
hold on;
plot(x_axis2,[results_OMP_def(1),results_OMP_def(2),results_OMP_def(3),results_OMP_def(4),results_OMP_def(5)],'.-','MarkerSize',20,'LineWidth',1);
plot(x_axis2,[results_CUDA(1),results_CUDA(2),results_CUDA(3),results_CUDA(4),results_CUDA(5)],'.-','MarkerSize',20,'LineWidth',1);
legend({'Sequential','OMP default','CUDA'},'FontSize',12);

% Speed Up

speed_OMP_def = results_seq./results_OMP_def;
speed_CUDA = results_seq./results_CUDA;

figure(5);
plot(x_axis,speed_OMP_def,'.-','MarkerSize',20,'LineWidth',2);
title('Speed Up')
xticks(x_axis);
xticklabels({'0.01','0.03','0.05','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1'});
xlabel('Pixel');
ylabel('Speed Up');
hold on;
plot(x_axis,speed_CUDA,'.-','MarkerSize',20,'LineWidth',2);

legend({'OMP default','CUDA'},'FontSize',12);

end

