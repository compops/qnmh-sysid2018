##########################################################################
# Sample the proposal
##########################################################################
def sampleProposal(self,):

    if (self.PMHtype == "PMH0"):
        if (self.nPars == 1):
            self.thp[self.iter, :] = self.th[self.iter - 1, :] + \
                self.stepSize * np.random.normal();
        else:
            self.thp[self.iter, :] = self.th[self.iter - 1, :] + np.random.multivariate_normal(
                np.zeros(self.nPars), self.stepSize**2 * np.eye(self.nPars));

    elif (self.PMHtype == "pPMH0"):
        # Sample the preconditioned PMH0 proposal
        if (self.nPars == 1):
            self.thp[self.iter, :] = self.th[self.iter - 1, :] + \
                self.stepSize * np.random.normal();
        else:
            self.thp[self.iter, :] = self.th[self.iter - 1, :] + np.random.multivariate_normal(
                np.zeros(self.nPars), self.stepSize**2 * self.invHessian);

    elif (self.PMHtype == "PMH1"):
        self.thp[self.iter, :] = self.th[self.iter - 1, :] + 0.5 * self.stepSize**2 * self.gradient[self.iter -
                                                                                                    1, :] + np.random.multivariate_normal(np.zeros(self.nPars), self.stepSize**2 * np.eye(self.nPars));

    elif (self.PMHtype == "bPMH1"):
        #"Binary" PMH1 proposal
        self.thp[self.iter, :] = self.th[self.iter - 1, :] + self.stepSize * np.sign(
            self.gradient[self.iter - 1, :]) + np.random.multivariate_normal(np.zeros(self.nPars), self.stepSize**2 * np.eye(self.nPars));

    elif (self.PMHtype == "pbPMH1"):
        #"Binary" PMH1 proposal
        self.thp[self.iter, :] = self.th[self.iter - 1, :] + self.stepSize * np.dot(self.invHessian, np.sign(
            self.gradient[self.iter - 1, :])) + np.random.multivariate_normal(np.zeros(self.nPars), self.stepSize**2 * self.invHessian);

    elif(self.PMHtype == "pPMH1"):
        # Sample the preconditioned PMH1 proposal
        self.thp[self.iter, :] = self.th[self.iter - 1, :] + 0.5 * self.stepSize**2 * np.dot(
            self.invHessian, self.gradient[self.iter - 1, :]) + np.random.multivariate_normal(np.zeros(self.nPars), self.stepSize**2 * self.invHessian);

    elif (self.PMHtype == "PMH2"):
        self.thp[self.iter, :] = self.th[self.iter - 1, :] + 0.5 * self.stepSize**2 * np.dot(self.gradient[self.iter - 1, :], np.linalg.pinv(
            self.hessian[self.iter - 1, :, :])) + np.random.multivariate_normal(np.zeros(self.nPars), self.stepSize**2 * np.linalg.pinv(self.hessian[self.iter - 1, :, :]));

    elif (self.PMHtype == "qPMH2"):

        if (self.iter > self.memoryLength):
            self.thp[self.iter, :] = self.th[self.iter - self.memoryLength, :] + 0.5 * self.stepSize**2 * np.dot(
                self.gradient[self.iter - self.memoryLength, :], self.hessian[self.iter - self.memoryLength, :, :]) + np.random.multivariate_normal(np.zeros(self.nPars), self.stepSize**2 * self.hessian[self.iter - self.memoryLength, :, :]);
        else:
            # Initial phase, use pPMH0
            self.thp[self.iter, :] = self.th[self.iter - 1, :] + np.random.multivariate_normal(
                np.zeros(self.nPars), self.stepSize**2 * self.hessian[self.iter - 1, :, :]);

##########################################################################
# Calculate Acceptance Probability
##########################################################################
def calculateAcceptanceProbability(self, sm,  thSys, ):

    # Check the "hard prior"
    if (inferenceModel.priorUniform() == 0.0):
        if (self.writeOutPriorWarnings):
            print("The parameters " +
                    str(self.thp[self.iter, :]) + " were proposed.");
        return None

    # Run the smoother to get the ll-estimate, gradient and hessian-estimate
    self.estimateLikelihoodGradientsHessians(sm, thSys)

    # Compute the part in the acceptance probability related to the non-symmetric proposal
    if (self.PMHtype == "PMH0"):
        proposalP = 0
        proposal0 = 0

    elif (self.PMHtype == "pPMH0"):
        proposalP = 0
        proposal0 = 0

    elif (self.PMHtype == "PMH1"):
        proposalP = lognormpdf(self.thp[self.iter, :], self.th[self.iter - 1, :] + 0.5 * self.stepSize **
                                2 * self.gradient[self.iter - 1, :], self.stepSize**2 * np.eye(self.nPars));
        proposal0 = lognormpdf(self.th[self.iter - 1, :],  self.thp[self.iter, :] + 0.5 * self.stepSize **
                                2 * self.gradientp[self.iter, :],     self.stepSize**2 * np.eye(self.nPars));

    elif (self.PMHtype == "bPMH1"):
        #"Binary" PMH1 proposal
        proposalP = lognormpdf(self.thp[self.iter, :], self.th[self.iter - 1, :] + self.stepSize * np.sign(
            self.gradient[self.iter - 1, :]), self.stepSize**2 * np.eye(self.nPars));
        proposal0 = lognormpdf(self.th[self.iter - 1, :],  self.thp[self.iter, :] + self.stepSize * np.sign(
            self.gradientp[self.iter, :]), self.stepSize**2 * np.eye(self.nPars));

    elif (self.PMHtype == "pbPMH1"):
        #"Binary" PMH1 proposal
        proposalP = lognormpdf(self.thp[self.iter, :], self.th[self.iter - 1, :] + self.stepSize * np.dot(
            self.invHessian, np.sign(self.gradient[self.iter - 1, :])), self.stepSize**2 * self.invHessian);
        proposal0 = lognormpdf(self.th[self.iter - 1, :],  self.thp[self.iter, :] + self.stepSize * np.dot(
            self.invHessian, np.sign(self.gradientp[self.iter, :])), self.stepSize**2 * self.invHessian);

    elif (self.PMHtype == "pPMH1"):
        proposalP = lognormpdf(self.thp[self.iter, :], self.th[self.iter - 1, :] + 0.5 * self.stepSize**2 * np.dot(
            self.invHessian, self.gradient[self.iter - 1, :]),  self.stepSize**2 * self.invHessian);
        proposal0 = lognormpdf(self.th[self.iter - 1, :], self.thp[self.iter, :] + 0.5 * self.stepSize**2 * np.dot(
            self.invHessian, self.gradientp[self.iter, :]),  self.stepSize**2 * self.invHessian);

    elif (self.PMHtype == "PMH2"):
        proposalP = lognormpdf(self.thp[self.iter, :], self.th[self.iter - 1, :] + 0.5 * self.stepSize**2 * np.dot(self.gradient[self.iter - 1, :],
                                                                                                                    np.linalg.pinv(self.hessian[self.iter - 1, :, :])), self.stepSize**2 * np.linalg.pinv(self.hessian[self.iter - 1, :, :]));
        proposal0 = lognormpdf(self.th[self.iter - 1, :],  self.thp[self.iter, :] + 0.5 * self.stepSize**2 * np.dot(
            self.gradientp[self.iter, :],   np.linalg.pinv(self.hessianp[self.iter, :, :])), self.stepSize**2 * np.linalg.pinv(self.hessianp[self.iter, :, :]));

    elif (self.PMHtype == "qPMH2"):

        if (self.iter > self.memoryLength):
            proposalP = lognormpdf(self.thp[self.iter, :],                   self.th[self.iter - self.memoryLength, :] + 0.5 * self.stepSize**2 * np.dot(
                self.gradient[self.iter - self.memoryLength, :],  self.hessian[self.iter - self.memoryLength, :, :]), self.stepSize**2 * self.hessian[self.iter - self.memoryLength, :, :]);
            proposal0 = lognormpdf(self.th[self.iter - self.memoryLength, :],  self.thp[self.iter, :] + 0.5 * self.stepSize**2 * np.dot(
                self.gradientp[self.iter, :],                   self.hessianp[self.iter, :, :]),                   self.stepSize**2 * self.hessianp[self.iter, :, :]);
        else:
            # Initial phase, use pPMH0
            proposalP = lognormpdf(self.thp[self.iter, :],   self.th[self.iter - 1, :],
                                    self.stepSize**2 * self.hessian[self.iter - 1, :, :]);
            proposal0 = lognormpdf(
                self.th[self.iter - 1, :],  self.thp[self.iter, :], self.stepSize**2 * self.hessianp[self.iter, :, :]);

    # Compute prior and Jacobian
    self.priorp[self.iter] = inferenceModel.prior()
    self.Jp[self.iter] = inferenceModel.Jacobian()

    # Compute the acceptance probability
    self.aprob[self.iter] = self.flag * np.exp(self.llp[self.iter, :] - self.ll[self.iter - 1, :] + proposal0 - proposalP +
                                                self.priorp[self.iter, :] - self.prior[self.iter - 1, :] + self.Jp[self.iter, :] - self.J[self.iter - 1, :]);

    # Store the proposal calculations
    self.proposalProb[self.iter] = proposal0
    self.proposalProbP[self.iter] = proposalP
    self.llDiff[self.iter] = self.llp[self.iter, :] - \
        self.ll[self.iter - 1, :];

##########################################################################
# Run the SMC algorithm and get the required information
##########################################################################
def estimateLikelihoodGradientsHessians(self, sm, thSys,):

    # Flag if the Hessian is PSD or not.
    self.flag = 1.0

    # PMH0, only run the filter and extract the likelihood estimate
    if (self.PMHtypeN == 0):
        stateEstimator.filter(thSys)

    # PMH1, only run the smoother and extract the likelihood estimate and gradient
    if (self.PMHtypeN == 1):
        stateEstimator.smoother(thSys)
        self.gradientp[self.iter, :] = stateEstimator.gradient;

    # PMH2, only run the smoother and extract the likelihood estimate and gradient
    if (self.PMHtype == "qPMH2"):
        stateEstimator.smoother(thSys)
        self.gradientp[self.iter, :] = stateEstimator.gradient;

        # Note that this is the inverse Hessian
        self.hessianp[self.iter, :, :] = self.lbfgs_hessian_update();

        # Extract the diagonal if needed and regularise if not PSD
        self.checkHessian()

    elif (self.PMHtypeN == 2):
        stateEstimator.smoother(thSys)
        self.gradientp[self.iter, :] = stateEstimator.gradient;
        self.hessianp[self.iter, :, :] = stateEstimator.hessian;

        # Extract the diagonal if needed and regularise if not PSD
        self.checkHessian()

    # Create output
    self.llp[self.iter] = stateEstimator.ll
    self.xp[self.iter, :] = stateEstimator.xtraj;

    return None

##########################################################################
# Extract the diagonal if needed and regularise if not PSD
##########################################################################
def checkHessian(self):

    # Extract the diagonal if it is the only part that we want
    if (self.makeHessianDiagonal):
        self.hessianp[self.iter, :, :] = np.diag(
            np.diag(self.hessianp[self.iter, :, :]));

    # Pre-calculate posterior covariance estimate
    if ((self.makeHessianPSDmethod == "hybrid") & (self.iter >= self.nBurnIn) & (self.empHessian == None)):
        self.empHessian = np.cov(self.th[range(
            self.nBurnIn - self.PSDmethodhybridSamps, self.nBurnIn), ].transpose())

    if ((self.makeHessianPSDmethod == "hybrid2") & (self.iter >= self.nBurnIn) & (self.empHessian == None)):
        self.empHessian = np.cov(self.th[range(
            self.nBurnIn - self.PSDmethodhybridSamps, self.nBurnIn), ].transpose())

    # Check if it is PSD
    if (~isPSD(self.hessianp[self.iter, :, :])):

        eigens = np.linalg.eig(self.hessianp[self.iter, :, :])[0];

        #=================================================================
        # Should we try to make the Hessian PSD using approach 1
        # Mirror the smallest eigenvalue in zero.
        #=================================================================

        # Add a diagonal matrix proportional to the largest negative eigv
        if (self.makeHessianPSDmethod == "regularise"):
            mineigv = np.min(np.linalg.eig(
                self.hessianp[self.iter, :, :])[0])
            self.hessianp[self.iter, :, :] = self.hessianp[self.iter,
                                                            :, :] - 2.0 * mineigv * np.eye(self.nPars)
            print("Iteration: " + str(self.iter) + " has eigenvalues: " +
                    str(eigens) + " mirroring by adding " + str(- 2.0 * mineigv));

        #=================================================================
        # Should we try to make the Hessian PSD using approach 2
        # During burn-in: mirror the smallest eigenvalue in zero.
        # After burn-in:  replace Hessian with covariance matrix from
        #                 the last iterations during the burn-in.
        #=================================================================

        # Add a diagonal matrix proportional to the largest negative eigv during burnin
        if ((self.makeHessianPSDmethod == "hybrid") & (self.iter <= self.nBurnIn)):
            mineigv = np.min(np.linalg.eig(
                self.hessianp[self.iter, :, :])[0])
            self.hessianp[self.iter, :, :] = self.hessianp[self.iter,
                                                            :, :] - 2.0 * mineigv * np.eye(self.nPars)
            print("Iteration: " + str(self.iter) + " has eigenvalues: " +
                    str(eigens) + " mirroring by adding " + str(- 2.0 * mineigv));

        # Replace the Hessian with the posterior covariance matrix after burin
        if ((self.makeHessianPSDmethod == "hybrid") & (self.iter > self.nBurnIn)):
            self.hessianp[self.iter, :, :] = self.empHessian;
            print("Iteration: " + str(self.iter) + " has eigenvalues: " +
                    str(eigens) + " replaced Hessian with pre-computed estimated.");

        #=================================================================
        # Should we try to make the Hessian PSD using approach 3
        # Reject the proposed parameters
        #=================================================================

        # Discard the estimate (make the algorithm reject)
        if (self.makeHessianPSDmethod == "reject"):
            self.flag = 0.0

        #=================================================================
        # Should we try to make the Hessian PSD using approach 4
        # Flip the negative eigenvalues
        #=================================================================

        if (self.makeHessianPSDmethod == "flipEigenvalues"):
            foo = np.linalg.eig(self.hessianp[self.iter, :, :]);
            self.hessianp[self.iter, :, :] = np.dot(
                np.dot(foo[1], np.diag(np.abs(foo[0]))), foo[1]);

        #=================================================================
        # Should we try to make the Hessian PSD using approach 5
        # During burn-in: replace with qPMH2-Hessian initalisation
        # After burn-in:  replace Hessian with covariance matrix from
        #                 the last iterations during the burn-in.
        #=================================================================

        # Add a diagonal matrix proportional to the largest negative eigv during burnin
        if ((self.makeHessianPSDmethod == "hybrid2") & (self.iter <= self.nBurnIn)):
            self.hessianp[self.iter, :, :] = np.eye(
                self.nPars) / self.epsilon;
            print("Iteration: " + str(self.iter) + " has eigenvalues: " +
                    str(eigens) + " replacing with initial Hessian.");

        # Replace the Hessian with the posterior covariance matrix after burin
        if ((self.makeHessianPSDmethod == "hybrid2") & (self.iter > self.nBurnIn)):
            self.hessianp[self.iter, :, :] = self.empHessian;
            print("Iteration: " + str(self.iter) + " has eigenvalues: " +
                    str(eigens) + " replaced Hessian with pre-computed estimated.");

        #=================================================================
        # Check if it did not help
        #=================================================================

        if ~(isPSD(self.hessianp[self.iter, :, :])):
            if (self.makeHessianPSDmethod != "reject"):
                print("pmh: tried to correct for a non PSD Hessian but failed.");
                self.flag = 0.0