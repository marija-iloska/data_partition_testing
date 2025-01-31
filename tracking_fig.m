% Tracking figure
clear all
close all
clc

% Load data
load dy100_dx400_M100track.mat

dx = length(x(:,1));


% Plot trajectory of random state
j = datasample(1:dx, 1);

close all
t0 = 1; tT = 30;

sz = 1.8;

mg = [237, 110, 152]/256;
%bg = [122, 186, 126]/256;

bg = [39, 163, 151]/256;


plot(x(j,t0:tT), 'k', 'LineWidth',2)
hold on
plot(x_grp(j,t0:tT), 'Color', 'm', 'LineStyle','--', 'LineWidth',sz)
hold on
plot(x_mpf(j,t0:tT), 'Color', bg, 'LineStyle','-.', 'LineWidth',sz)
set(gca, 'FontSize',15)
xlabel('Time', 'FontSize',20)
ylabel('State', 'FontSize',20)
legend('True State', 'RP-TPF', 'TWRP-MPF', 'FontSize', 20)