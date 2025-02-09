% Run the results script
results;
MAXDEGREE = size(err_svd,1)+2;

range = 3:MAXDEGREE;
semilogy(range, err_svd, 'k-x', range, err_eig, 'r-o');
title('Errors');
xlabel('degree');
ylabel('error');
legend('dgesvd', 'dsyev', 'Location','NorthWest');
print -depsc errors.eps

figure(2);
semilogy(range, times_svd, 'k-x', range, times_eig, 'r-o');
title('Execution Times');
xlabel('degree');
ylabel('times');
legend('dgesvd', 'dsyev', 'Location','NorthWest');
print -depsc times.eps
