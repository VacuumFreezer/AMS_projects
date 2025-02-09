%% AMS595 Hw1 Estimating Pi Task2
%% Experiment of while loop
[pi_while, tol] = exp_while(100, 1e-2); % User defined precision equals 1e-2
disp(pi_while*4);
disp(tol);

%%
% Test whether the current update is below the bound of precision
% If below the bound, output true, else output false
function p = bin_test(data, end_i, bin_len, epslion)
      if bin_len >= end_i
          p = false;
      else
          % Take the former bin_len updates as a bin, then calculate the
          % range
          range = max(data((end_i-bin_len):end_i)) - min(data((end_i-bin_len):end_i));
          if range < epslion
              p = true;
          else
              p = false;
          end
      end
  end

%%
% steps: the upper bound of numbers of continuous no_update steps.
% epsilon: the expected precision of the program
% Output the final estimation of pi and the steps needed to reach the
% precision
function [ratio, total_cnt] = exp_while(steps, epslion) 
    ratio = 0;
    in_cnt = 0;
    total_cnt = 0;
    no_update = 0;
    ratio_list = zeros(1,1e8); % pre-allocation for the results and set the upper bound of nop
    while no_update <= steps % Exit loop when continuous no update steps reach the uppe bound 
        poi = rand(2,1);
        total_cnt = total_cnt + 1;
        if total_cnt > 1e8
            error('List size too small');
        end
        if poi(1)^2+poi(2)^2 < 1.0
            in_cnt = in_cnt + 1;
        end
        ratio = in_cnt / total_cnt;
        ratio_list(total_cnt) = ratio;
        if bin_test(ratio_list, total_cnt, 100000, epslion)
            no_update = no_update + 1;
        else
            no_update = 0;
        end
    end
end