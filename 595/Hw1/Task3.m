%% AMS595 Hw1 Estimating Pi Task3
%% Experiment of while loop
[pi_while, tol, points] = grap_while(100, 1e-2); % User defined precision equals 1e-2
disp(tol);

%% Plot the graphical display
figure;
hold on;
axis equal;

rectangle('Position',[0 0 1 1],'EdgeColor','black','LineWidth', 1); % Draw the square
colors = zeros(size(points,1), 3); 
% If the third entry is 0, the point fall in the circle, red color (R=1, G=0, B=0)
% Or it's blue (R=0, G=0, B=1)
colors(points(:,3) == 0, :) = repmat([1, 0, 0], sum(points(:,3) == 0), 1); 
colors(points(:,3) == 1, :) = repmat([0, 0, 1], sum(points(:,3) == 1), 1); 

scatter(points(:,1), points(:,2), 1, colors,'filled')
xlim([0 1]);
ylim([0 1]);

title(sprintf('Graphical display of the Monte Carlo method \n Pi = %.2f',pi_while*4));
hold off;

%saveas(gcf, '/MATLAB Drive/figs/graphical_display.png')
export('Pi_est.mlx', format='pdf');

%% function similar to task2 except it storage every point
function [ratio, total_cnt, points] = grap_while(steps, epslion) 
    ratio = 0;
    in_cnt = 0;
    total_cnt = 0;
    no_update = 0;
    ratio_list = zeros(1,1e8); % pre-allocation for the results
    points = []; 
    while no_update <= steps % Exit loop when continuous no update steps reach the uppe bound 
        poi = rand(1,2);
        total_cnt = total_cnt + 1;
        if total_cnt > 1e8
            error('List size too small');
        end
        if poi(1)^2+poi(2)^2 < 1.0
            in_cnt = in_cnt + 1;
            poi(end+1) = 0; % Color tag for inner points
        else
            poi(end+1) = 1; % Color tag for outer points
        end
        points = [points;poi]; % Storage the points in a matrix of 3 columns
        ratio = in_cnt / total_cnt;
        ratio_list(total_cnt) = ratio;
        if bin_test(ratio_list, total_cnt, 100000, epslion)
            no_update = no_update + 1;
        else
            no_update = 0;
        end
    end
end