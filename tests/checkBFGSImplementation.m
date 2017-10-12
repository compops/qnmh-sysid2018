parametersDiff = [-0.02956439, -0.00242051, -0.00969463];
gradientsDiff = [  1.65837436,  19.44478073,  33.56385822];
B = eye(3) * 0.0001;
inverseHessianEstimate = sqrt(B);

term1 = parametersDiff * gradientsDiff';
term2 = parametersDiff * B * parametersDiff';

theta = 0.8 * term2 / (term2 - term1);
r = theta * gradientsDiff + (1.0 - theta) * (B * parametersDiff')';

t = parametersDiff / term2;
u1 = sqrt(term2 / (parametersDiff * r'));
u2 = (B * parametersDiff')';
u = u1 * r + u2;

inverseHessianEstimate = (eye(3) - u' * t) * inverseHessianEstimate

gradient = [0.65667684,  36.54903406,  36.9900761];
inverseHessianEstimateSquared = inverseHessianEstimate * inverseHessianEstimate';
naturalGradient = (inverseHessianEstimateSquared * gradient')'