close all 
clear

N = 100;
l = 1.5;
sigma2_f = 1;
k = @(x1, x2) sigma2_f*exp(-(x1-x2)^2/(2*l));
t1 = 5;
x = linspace(-t1,t1,N);

for i = 1:N
    for j = 1:N
        K(i,j) = k(x(i),x(j));
    end
end

n = 15;
col = jet(n);
figure(1)
h = fill([-5, -5, 5, 5], [-1.96, 1.96, 1.96, -1.96], 'b');
set(h,'facealpha',.07)
hold on
for i = 1:n
    f = mvnrnd(zeros(N,1), K);
    plot(x, f, 'color', col(i,:), 'LineWidth', 1)
    hold on
end
ylim([-3,3])
title('Gaussian Process Priors using the SE kernel, $$l = 0.5, \sigma^2_f  = 1$$', 'Interpreter', 'Latex')
xlabel('x')
ylabel('f')
hold off

%%
a = 0.3;
sigma_n = 0.2;
eta = mvnrnd(zeros(N,1), sigma_n^2*eye(N));
t = 2;
for i = 1:N
    if x(i) < -t1+t
        ytrue(i) = x(i);
        yobs(i) = x(i) + eta(i);
        xobs(i) = x(i);
        xtrue(i) = x(i);
    elseif x(i) > t1-t
        ytrue(i) = x(i);
        yobs(i) = x(i) + eta(i);
        xobs(i) = x(i);
        xtrue(i) = x(i);
    else
        ytrue(i) = a*x(i).^2+x(i)-a*(t1-t)^2;
        xtrue(i) = x(i);
        xobs(i) = 0;
        yons(i) = 0;
    end
end
yobs(xobs == 0) = [];
xobs(xobs == 0) = [];
figure(2)
plot(xtrue, ytrue, 'b')
hold on
plot(xobs, yobs, 'or')
title('Observed data and true model example')
legend('True model', 'Observed data')
xlabel('x')
ylabel('y')
grid on
hold off

gram = x'*x;
res = (xobs*yobs')/(xobs*xobs');
yhat = res*x;

var_yhat = 1.96*sqrt(1+(x.^2/(xobs*xobs')))*sigma_n;

figure(3)
plot(xobs, yobs, 'or')
hold on
plot(xtrue, ytrue, 'b')
plot(x, yhat, '*m')
plot(x, yhat+var_yhat, '--m', x, yhat-var_yhat, '--m')
xlabel('x')
ylabel('y')
title('First Order Least Squares Fit Example')
legend('Observations', 'True function', 'First order least squares fit', '95% Confidence interval')
grid on


k = @(x1, x2) (x1*x2);
for i = 1:N
    for j = 1:N
        K(i,j) = k(x(i),x(j));
    end
end
Nobs = length(xobs);
for i = 1:Nobs
    for j = 1:Nobs
        Kobs(i,j) = k(xobs(i),xobs(j));
    end
end
for i = 1:N
    for j = 1:Nobs
        Kstar(i,j) = k(x(i),xobs(j));
    end
end
for i = 1:N
    for j = 1:N
        Kstarstar(i,j) = k(x(i),x(j));
    end
end


B = Kobs + sigma_n^2*eye(Nobs);
L = chol(B, 'lower');
alpha = L'\(L\yobs');
fbar = Kstar*alpha;
v = L\Kstar';
var_fbar = Kstarstar - v'*v;


figure(4)
plot(xobs, yobs, 'or')
hold on
plot(xtrue, ytrue, 'b')
plot(x, fbar, '*m')
plot(x, fbar+1.96*sqrt(sigma_n^2 + diag(var_fbar)), '--m')
plot(x, fbar-1.96*sqrt(sigma_n^2 + diag(var_fbar)), '--m')
xlabel('x')
ylabel('y')
title('Gaussian Process Fit Example, dot-product kernel')
legend('Observations', 'True function', 'Gaussian fit dot-product kernel', '95% Confidence interval')
grid on


k = @(x1, x2) sigma2_f*exp(-(x1-x2)^2/(2*l));
for i = 1:N
    for j = 1:N
        K(i,j) = k(x(i),x(j));
    end
end
Nobs = length(xobs);
for i = 1:Nobs
    for j = 1:Nobs
        Kobs(i,j) = k(xobs(i),xobs(j));
    end
end
for i = 1:N
    for j = 1:Nobs
        Kstar(i,j) = k(x(i),xobs(j));
    end
end
for i = 1:N
    for j = 1:N
        Kstarstar(i,j) = k(x(i),x(j));
    end
end


B = Kobs + sigma_n^2*eye(Nobs);
L = chol(B, 'lower');
alpha = L'\(L\yobs');
fbar = Kstar*alpha;
v = L\Kstar';
var_fbar = Kstarstar - v'*v;


figure(5)
plot(xobs, yobs, 'or')
hold on
plot(xtrue, ytrue, 'b')
plot(x, fbar, '*m')
plot(x, fbar+1.96*sqrt(sigma_n^2 + diag(var_fbar)), '--m')
plot(x, fbar-1.96*sqrt(sigma_n^2 + diag(var_fbar)), '--m')
xlabel('x')
ylabel('y')
title('Gaussian Process Fit Example, SE kernel, l = 1.5')
legend('Observations', 'True function', 'Gaussian fit SE kernel', '95% Confidence interval')
grid on


[p, S] = polyfit(xobs,yobs,3);
[yhat, delta] = polyconf(p,x,S);
figure(6)
plot(xobs, yobs, 'or')
hold on
plot(xtrue, ytrue, 'b')
plot(x, yhat, '*m')
plot(x, yhat+delta, '--m', x, yhat-delta, '--m')
xlabel('x')
ylabel('y')
title('Thrird Order Least Squares Fit Example')
legend('Observations', 'True function', 'Third order least squares fit', '95% Confidence interval')
grid on


    
