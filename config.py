
def can_build(env, platform):
    return True

def configure(env):
    env.pmlpp_build_old_classes = True
    env.pmlpp_build_tests = True

def get_doc_classes():
    return [
        "MLPPVector",
        "MLPPMatrix",
        "MLPPTensor3",

        "MLPPUtilities",
        "MLPPReg",
        "MLPPActivation",
        "MLPPCost",
        "MLPPTransforms",
        "MLPPStat",
        "MLPPNumericalAnalysis",
        "MLPPHypothesisTesting",
        "MLPPGaussMarkovChecker",
        "MLPPConvolutions",
        "MLPPLinAlg",

        "MLPPHiddenLayer",
        "MLPPOutputLayer",
        "MLPPMultiOutputLayer",

        "MLPPKNN",
        "MLPPKMeans",

        "MLPPMLP",
        "MLPPWGAN",
        "MLPPPCA",
        "MLPPUniLinReg",
        "MLPPOutlierFinder",
        "MLPPProbitReg",
        "MLPPSVC",
        "MLPPSoftmaxReg",
        "MLPPAutoEncoder",
        "MLPPTanhReg",
        "MLPPSoftmaxNet",
        "MLPPMultinomialNB",
        "MLPPMANN",
        "MLPPLogReg",
        "MLPPLinReg",
        "MLPPGaussianNB",
        "MLPPGAN",
        "MLPPExpReg",
        "MLPPDualSVC",
        "MLPPCLogLogReg",
        "MLPPBernoulliNB",
        "MLPPANN",

        "MLPPDataESimple",
        "MLPPDataSimple",
        "MLPPDataComplex",
        "MLPPData",

        "MLPPTests",
        "MLPPMatrixTests",

        "MLPPTestsOld",
    ]

def get_doc_path():
    return "doc_classes"

def get_license_file():
  return "COPYRIGHT.txt"
