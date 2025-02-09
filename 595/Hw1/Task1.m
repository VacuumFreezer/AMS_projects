%% AMS595 Hw1 Estimating Pi Task1

%% Experiment of for loop
%Conduct the experiment for certain number of points.
pi_for = for_loop(1e7);
disp(pi_for*4)


%% Plot the relation between relative error and number of points
benchmark= linspace(2,7,20); % Benchmark nops are linear power of 10
error_list = zeros(size(benchmark));
for i = 1:length(benchmark)
    cnt = fix(10^benchmark(i));
    sample = zeros(1,10); % Take mean of 10 separate experiments
    for j = 1:length(sample)
        rel_error = (4*for_loop(cnt) - pi) / pi;
        sample(j) = rel_error;
    end
    rel = mean(sample);
    error_list(i) = rel;
end
figure;
plot(benchmark, error_list, '-o');
xlabel('Number of random points, 10^k');
ylabel('Relative error');
legend('For loop');
title('The relation of relative error and number of points');
grid on;
%saveas(gcf, '/MATLAB Drive/figs/relative_error.png')

%% Plot the relation between runtime and relative error
T_list = zeros(size(benchmark));
error_list2 = zeros(size(benchmark));
for i = 1:length(benchmark)
    tic;
    cnt = fix(10^benchmark(i));
    sample = zeros(1,10); % Take mean of 10 separate experiments
    for j = 1:length(sample)
        rel_error = (4*for_loop(cnt) - pi) / pi;
        sample(j) = rel_error;
    end
    rel = mean(sample);
    error_list2(i) = rel;
    T_list(i) = toc / 10;
end
lgT_list = log10(T_list); % Use logarithmic of time for explicity
figure;
scatter(lgT_list, error_list2); % Since the runtime measurment fluctuates 
% while nop is small, we choose scatter to show the relation
xlabel('Runtime(log(s))');
ylabel('Relative error');
title('The comparasion of runtime and precision');
grid on;
%saveas(gcf, '/MATLAB Drive/figs/runtime.png')


%% 
%nop = number of points
function prob = for_loop(nop)
    in_cnt = 0;
    prob = 0;
    for cnt = 1:nop
        %poi represent the random point generated in the square
        poi = rand(2,1);
        %Those fall in to the circle contribute to the count
        if poi(1)^2+poi(2)^2 < 1.0
            in_cnt = in_cnt + 1;
            prob = in_cnt / cnt;
        else
            continue
        end
    end
end
