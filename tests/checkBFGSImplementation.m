parametersDiff = [-0.02956439, -0.00242051, -0.00969463];
gradientsDiff = [  1.65837436,  19.44478073,  33.56385822];
H = eye(3) * 0.0001;

quadraticFormSB = parametersDiff * H * parametersDiff';
curvatureCondition = parametersDiff * gradientsDiff';

term1 = (curvatureCondition + quadraticFormSB) / curvatureCondition^2;
term1 = term1 * parametersDiff' * parametersDiff;

term2 = H * gradientsDiff' * parametersDiff;
term2 = term2 + parametersDiff' * gradientsDiff * H;
term2 = term2 / curvatureCondition;

H = H + term1 - term2
