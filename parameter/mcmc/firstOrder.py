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