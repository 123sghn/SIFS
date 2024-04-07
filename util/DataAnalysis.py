# A class for feature analysis, including clustering, feature normalization, selection, and dimensionality reduction.

#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import os, sys, re, platform

from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)
from PyQt5.QtWidgets import QMessageBox
import numpy as np
import pandas as pd
import copy
from sklearn.cluster import (
    KMeans,
    AffinityPropagation,
    MeanShift,
    estimate_bandwidth,
    DBSCAN,
    AgglomerativeClustering,
    SpectralClustering,
    MiniBatchKMeans,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.manifold import TSNE
from MCL import MarkvCluster
import math
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=RuntimeWarning)

from feature_selection import (
    jAntColonyOptimization,
    jAntColonySystem,
    jAntLionOptimizer,
    jArithmeticOptimizationAlgorithm,
    jArtificialBeeColony,
    jArtificialButterflyOptimization,
    jAtomSearchOptimization,
    jBatAlgorithm,
    jBelugaWhaleOptimization,
    jBinaryDwarfMongooseOptimizer,
    jButterflyOptimizationAlgorithm,
    jChameleonSwarmAlgorithm,
    jCrowSearchAlgorithm,
    jCuckooSearchAlgorithm,
    jDifferentialEvolution,
    jDingoOptimizationAlgorithm,
    jDungBeetleOptimizer,
    jDynamicArithmeticOptimizationAlgorithm,
    jEmperorPenguinOptimizer,
    jEquilibriumOptimizer,
    jFireflyAlgorithm,
    jFlowerPollinationAlgorithm,
    jFruitFlyOptimizationAlgorithm,
    jGeneralizedNormalDistributionOptimization,
    jGeneticAlgorithm,
    jGeneticAlgorithmTour,
    jGravitationalSearchAlgorithm,
    jGreyWolfOptimizer,
    jHarmonySearch,
    jHarrisHawksOptimization,
    jHenryGasSolubilityOptimization,
    jHumanLearningOptimization,
    jHunterPreyOptimization,
    jJayaAlgorithm,
    jMantaRayForagingOptimization,
    jMarinePredatorsAlgorithm,
    jMonarchButterflyOptimization,
    jMothFlameOptimization,
    jMultiVerseOptimizer,
    jParticleSwarmOptimization,
    jPathFinderAlgorithm,
    jPoorAndRichOptimization,
    jSalpSwarmAlgorithm,
    jSandCatSwarmOptimization,
    jSatinBowerBirdOptimization,
    jSimulatedAnnealing,
    jSineCosineAlgorithm,
    jSlimeMouldAlgorithm,
    jSymbioticOrganismsSearch,
    jTreeGrowthAlgorithm,
    jTreeSeedAlgorithm,
    jWeightedSuperpositionAttraction,
    jWhaleOptimizationAlgorithm,
    jWildHorseOptimizer,
)
from losses.jFitnessFunction import jFitnessFunction
from sklearn.linear_model import LogisticRegression


class SIFSData(object):
    def __init__(self, kw):
        self.kw = kw  # dict
        self.dataframe = None  # pandas dataframe
        self.datalabel = None  # sample label
        self.data_sample_purpose = None  # 1-D ndarray, sample used as training dataset (True) or testing dataset(False)
        self.cluster_result = None  # 1-D ndarray
        self.cluster_plot_data = None  # list [cluster1, cluster2, ...] cluster - > 2-D ndarray [x, y, z, ...]
        self.spots = None  # discarded
        self.dimension_reduction_result = None  # 2-D ndarray
        self.error_msg = ""  # string
        self.row = 0  # int
        self.column = 0  # int
        self.feature_selection_result = (
            None  # DataFrame  columns=['SampleName', 'Value']
        )
        self.feature_origin_data = None  # pandas DataFrame
        self.feature_selection_data = None  # pandas DataFrame
        self.feature_normalization_data = None  # pandas DataFrame
        self.loss_data = None  

    def load_data_from_file(self, file):
        if not os.path.exists(file):
            self.error_msg = "data file does not exist."
            return False
        try:
            if file.endswith(".tsv"):
                df = pd.read_csv(file, sep="\t", header=None)
                self.dataframe = df.iloc[:, 1:]
                self.row = self.dataframe.index.size
                self.column = self.dataframe.columns.size
                self.dataframe.index = ["Sample_%s" % i for i in range(self.row)]
                self.dataframe.columns = ["F_%s" % i for i in range(self.column)]
                self.datalabel = np.array(df.iloc[:, 0]).astype(int)
                self.data_sample_purpose = np.array([True] * self.row)
            elif file.endswith(".csv"):
                df = pd.read_csv(file, sep=",", header=None)
                self.dataframe = df.iloc[:, 1:]
                self.row = self.dataframe.index.size
                self.column = self.dataframe.columns.size
                self.dataframe.index = ["Sample_%s" % i for i in range(self.row)]
                self.dataframe.columns = ["F_%s" % i for i in range(self.column)]
                self.datalabel = np.array(df.iloc[:, 0]).astype(int)
                self.data_sample_purpose = np.array([True] * self.row)
            elif file.endswith(".svm"):
                with open(file) as f:
                    record = f.read().strip()
                record = re.sub("\d+:", "", record)
                array = np.array(
                    [[i for i in item.split()] for item in record.split("\n")]
                )
                self.dataframe = pd.DataFrame(array[:, 1:], dtype=float)
                self.row = self.dataframe.index.size
                self.column = self.dataframe.columns.size
                self.dataframe.index = ["Sample_%s" % i for i in range(self.row)]
                self.dataframe.columns = ["F_%s" % i for i in range(self.column)]
                self.datalabel = array[:, 0].astype(int)
                self.data_sample_purpose = np.array([True] * self.row)
            else:
                with open(file) as f:
                    record = f.read().strip().split("@")[-1].split("\n")[1:]
                array = np.array([item.split(",") for item in record])
                self.dataframe = pd.DataFrame(array[:, 0:-1], dtype=float)
                self.row = self.dataframe.index.size
                self.column = self.dataframe.columns.size
                self.dataframe.index = ["Sample_%s" % i for i in range(self.row)]
                self.dataframe.columns = ["F_%s" % i for i in range(self.column)]
                label = []
                for i in array[:, -1]:
                    if i == "yes":
                        label.append(1)
                    else:
                        label.append(0)
                self.datalabel = np.array(label)
                self.data_sample_purpose = np.array([True] * self.row)
        except Exception as e:
            self.error_msg = str(e)
            return False
        return True

    def load_data_from_descriptor(self, objDescriptor):
        data = copy.deepcopy(objDescriptor.encoding_array)
        try:
            columnName = data[0, 2:]
            indexName = data[1:, 0]
            self.dataframe = pd.DataFrame(
                data[1:, 2:].astype(float), index=indexName, columns=columnName
            )
            self.row = self.dataframe.index.size
            self.column = self.dataframe.columns.size
            self.datalabel = data[1:, 1].astype(int)
            self.data_sample_purpose = objDescriptor.sample_purpose
        except Exception as e:
            self.error_msg = str(e)
            return False
        return True

    def load_data_from_dimension_reduction(self, objData):
        try:
            data = copy.deepcopy(objData)
            columnName = [
                "PC%s" % (i + 1)
                for i in range(data.dimension_reduction_result.shape[1])
            ]
            indexName = data.dataframe.index
            self.dataframe = pd.DataFrame(
                data.dimension_reduction_result, index=indexName, columns=columnName
            )
            self.row = self.dataframe.index.size
            self.column = self.dataframe.columns.size
            self.datalabel = data.datalabel
            self.data_sample_purpose = data.data_sample_purpose
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def load_data_from_selection(self, objSelection):
        try:
            self.dataframe = copy.deepcopy(objSelection.feature_selection_data)
            del self.dataframe["Labels"]
            self.datalabel = copy.copy(objSelection.datalabel)
            self.data_sample_purpose = copy.copy(objSelection.data_sample_purpose)
            self.row = self.dataframe.values.shape[0]
            self.column = self.dataframe.values.shape[1]
        except Exception as e:
            self.error_msg = str(e)
            return False
        return True

    def load_data_from_normalization(self, objSelection):
        try:
            self.dataframe = copy.deepcopy(objSelection.feature_normalization_data)
            del self.dataframe["Labels"]
            self.datalabel = copy.copy(objSelection.datalabel)
            self.data_sample_purpose = copy.copy(objSelection.data_sample_purpose)
            self.row = self.dataframe.values.shape[0]
            self.column = self.dataframe.values.shape[1]
        except Exception as e:
            self.error_msg = str(e)
            return False
        return True

    def kmeans(self):
        try:
            nclusters = self.kw["nclusters"]
            if not self.dataframe is None:
                self.cluster_result = KMeans(n_clusters=nclusters).fit_predict(
                    self.dataframe.values
                )
                self.cluster_plot_data, ok = self.t_sne(2)
                return True
            else:
                self.error_msg = "Data is null."
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def MiniBatchKMeans(self):
        try:
            nclusters = self.kw["nclusters"]
            if not self.dataframe is None:
                self.cluster_result = MiniBatchKMeans(n_clusters=nclusters).fit_predict(
                    self.dataframe.values
                )
                self.cluster_plot_data, ok = self.t_sne(2)
                return True
            else:
                self.error_msg = "Data is null."
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def MCL(self):
        try:
            if not self.dataframe is None:
                self.cluster_result = MarkvCluster(
                    self.dataframe.values,
                    int(self.kw["expand_factor"]),
                    self.kw["inflate_factor"],
                    self.kw["multiply_factor"],
                    2000,
                ).cluster_array
                self.cluster_plot_data, ok = self.t_sne(2)
                if ok and len(self.cluster_result) == len(self.dataframe):
                    return True
                else:
                    return False
            else:
                self.error_msg = "Data is null."
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def GM(self):
        try:
            nclusters = self.kw["nclusters"]
            if not self.dataframe is None:
                self.cluster_result = GaussianMixture(
                    n_components=nclusters
                ).fit_predict(self.dataframe.values)
                self.cluster_plot_data, ok = self.t_sne(2)
                return True
            else:
                self.error_msg = "Data is null."
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Agglomerative(self):
        try:
            nclusters = self.kw["nclusters"]
            if not self.dataframe is None:
                self.cluster_result = AgglomerativeClustering(
                    n_clusters=nclusters
                ).fit_predict(self.dataframe.values)
                self.cluster_plot_data, ok = self.t_sne(2)
                return True
            else:
                self.error_msg = "Data is null."
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Spectral(self):
        try:
            nclusters = self.kw["nclusters"]
            if not self.dataframe is None:
                self.cluster_result = SpectralClustering(
                    n_clusters=nclusters
                ).fit_predict(self.dataframe.values)
                self.cluster_plot_data, ok = self.t_sne(2)
                return True
            else:
                self.error_msg = "Data is null."
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def hcluster(self):
        try:
            if not self.dataframe is None:
                disMat = sch.distance.pdist(self.dataframe.values, "euclidean")
                Z = sch.linkage(disMat, method="average")
                self.cluster_result = sch.fcluster(Z, 1, "inconsistent")
                self.cluster_plot_data, ok = self.t_sne(2)
                return True
            else:
                self.error_msg = "Data is null."
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def APC(self):
        try:
            if not self.dataframe is None:
                self.cluster_result = AffinityPropagation().fit_predict(
                    self.dataframe.values
                )
                self.cluster_plot_data, ok = self.t_sne(2)
                return True
            else:
                self.error_msg = "Data is null."
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def meanshift(self):
        try:
            if not self.dataframe is None:
                bandwidth = estimate_bandwidth(self.dataframe)
                try:
                    self.cluster_result = MeanShift(
                        bandwidth=bandwidth, bin_seeding=True
                    ).fit_predict(self.dataframe.values)
                except Exception as e:
                    self.cluster_result = np.zeros(len(self.dataframe))
                self.cluster_plot_data, ok = self.t_sne(2)
                return True
            else:
                self.error_msg = "Data is null."
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def DBSCAN(self):
        try:
            if not self.dataframe is None:
                data = StandardScaler().fit_transform(self.dataframe.values)
                self.cluster_result = DBSCAN().fit_predict(data)
                self.cluster_plot_data, ok = self.t_sne(2)
                return True
            else:
                self.error_msg = "Data is null."
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def export_cluster_text(self):
        try:
            if not self.cluster_result is None:
                text = "SampleName\tCluster\n==============================\n"
                for i, name in enumerate(list(self.dataframe.index)):
                    text += "%s\t%s\n" % (name, self.cluster_result[i])
                return text
        except Exception as e:
            self.error_msg = str(e)
            return None

    """
    def generate_spot_data(self, c_label, rd_data):
        spots = [{}]
        color = {0: 0, 1: 6, 2: 7, 3: 8, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5}
        if len(c_label) != len(rd_data):
            self.error_msg = 'Plot failed.'
        else:
            spots = [
                {'pos': rd_data[i], 'data': 1, 'brush': pg.intColor(color[c_label[i] % 9]), 'symbol': c_label[i] % 12,
                 'size': 15} for i in range(len(c_label))]
        return spots
    """

    def generate_plot_data(self, label, rd_data):
        plot_data = []
        clusters = sorted(set(label))
        for c in clusters:
            plot_data.append([c, rd_data[np.where(label == c)]])
        return plot_data

    def t_sne(self, n_components=2):
        try:
            if not self.dataframe is None:
                if n_components >= self.dataframe.shape[1]:
                    self.error_msg = "The reduced dimension number is out of range."
                    return None, False
                rd_data = TSNE(
                    n_components=n_components, method="exact", learning_rate=100
                ).fit_transform(self.dataframe.values)
                return rd_data, True
            else:
                self.error_msg = "Data is null."
                return None, False
        except Exception as e:
            self.error_msg = str(e)
            return None, False

    def PCA(self, n_components=2):
        try:
            if not self.dataframe is None:
                if n_components >= self.dataframe.shape[1]:
                    self.error_msg = "The reduced dimension number is out of range."
                    return None, False
                rd_data = PCA(n_components=n_components).fit_transform(
                    self.dataframe.values
                )
                return rd_data, True
            else:
                self.error_msg = "Data is null."
                return None, False
        except Exception as e:
            self.error_msg = str(e)
            return None, False

    def LDA(self, n_components=2):
        try:
            if not self.dataframe is None:
                if n_components >= self.dataframe.shape[1]:
                    self.error_msg = "The reduced dimension number is out of range."
                    return None, False
                lda = LatentDirichletAllocation(n_components=n_components).fit(
                    self.dataframe.values, self.datalabel
                )
                rd_data = lda.transform(self.dataframe.values)
                return rd_data, True
            else:
                self.error_msg = "Data is null."
                return None, False
        except Exception as e:
            self.error_msg = str(e)
            return None, False

    def save_data(self, file, analysis_type):
        if analysis_type == "Cluster algorithms":
            if not self.cluster_result is None:
                with open(file, "w", encoding="utf-8") as f:
                    f.write("SampleName\tCluster\n")
                    for i in range(len(self.cluster_result)):
                        f.write(
                            "%s\t%s\n"
                            % (self.dataframe.index[i], self.cluster_result[i])
                        )
            else:
                QMessageBox.critical(
                    self,
                    "Error",
                    "Empty data!",
                    QMessageBox.Ok | QMessageBox.No,
                    QMessageBox.Ok,
                )
        else:
            if not self.dimension_reduction_result is None:
                np.savetxt(
                    file, self.dimension_reduction_result, fmt="%f", delimiter="\t"
                )
            else:
                QMessageBox.critical(
                    self,
                    "Error",
                    "Empty data!",
                    QMessageBox.Ok | QMessageBox.No,
                    QMessageBox.Ok,
                )

    """ feature selection / normalization """

    def Swarm(self, algorithm):
        try:
            labels = self.datalabel  
            data = np.array(self.dataframe.values) 

            if data.shape[0] < 2 or data.shape[1] < 2:
                self.error_msg = "Sample number is two less" 
                return False

            train_indices = np.random.choice(
                len(data), size=int(0.8 * len(data)), replace=False
            )
            x_train = data[train_indices]
            y_train = labels[train_indices]

            test_indices = np.setdiff1d(np.arange(len(data)), train_indices)
            x_test = data[test_indices]
            y_test = labels[test_indices]

            N = self.kw["N"]
            max_Iter = self.kw["max_Iter"]
            alpha = self.kw["alpha"]
            beta = self.kw["beta"]
            thres = self.kw["thres"]
            tau = self.kw["tau"]
            rho = self.kw["rho"]
            eta = self.kw["eta"]

            phi = self.kw["phi"]
            Alpha = self.kw["Alpha"]
            Mu = self.kw["Mu"]
            max_limit = self.kw["max_limit"]
            step_e = self.kw["step_e"]
            ratio = self.kw["ratio"]
            type = self.kw["type"]
            gamma = self.kw["gamma"]
            A_max = self.kw["A_max"]
            r0_max = self.kw["r0_max"]
            nBabysitter = self.kw["nBabysitter"]
            peep = self.kw["peep"]
            c = self.kw["c"]
            p = self.kw["p"]
            AP = self.kw["AP"]
            f1 = self.kw["f1"]
            Pa = self.kw["Pa"]
            CR = self.kw["CR"]
            F = self.kw["F"]
            P = self.kw["P"]
            Q = self.kw["Q"]
            M = self.kw["M"]
            f = self.kw["f"]
            l = self.kw["l"]
            beta0 = self.kw["beta0"]
            theta = self.kw["theta"]
            MR = self.kw["MR"]
            Tour_size = self.kw["Tour_size"]
            G0 = self.kw["G0"]
            num_gas = self.kw["num_gas"]
            K = self.kw["K"]
            L1 = self.kw["L1"]
            L2 = self.kw["L2"]
            L3 = self.kw["L3"]
            c1 = self.kw["c1"]
            c2 = self.kw["c2"]
            PAR = self.kw["PAR"]
            HMCR = self.kw["HMCR"]
            bw = self.kw["bw"]
            pi = self.kw["pi"]
            pr = self.kw["pr"]
            B = self.kw["B"]
            S = self.kw["S"]
            FADs = self.kw["FADs"]
            peri = self.kw["peri"]
            Smax = self.kw["Smax"]
            BAR = self.kw["BAR"]
            num_land1 = self.kw["num_land1"]
            b = self.kw["b"]
            w = self.kw["w"]
            Pmut = self.kw["Pmut"]
            z = self.kw["z"]
            T0 = self.kw["T0"]
            num_tree1 = self.kw["num_tree1"]
            num_tree2 = self.kw["num_tree2"]
            num_tree4 = self.kw["num_tree4"]
            lambda_ = self.kw["lambda_"]
            ST = self.kw["ST"]
            sl = self.kw["sl"]
            lambda_val = self.kw["lambda_val"]

            basic_model = self.kw["basic_model"]
            try:
                models = {
                    "AdaBoost": AdaBoostClassifier(),
                    "Bagging Classifier": BaggingClassifier(),
                    "Bernoulli Naive Bayes": BernoulliNB(),
                    "Calibrated Classifier": CalibratedClassifierCV(),
                    "Decision Tree": DecisionTreeClassifier(),
                    "Extra Trees Classifier": ExtraTreesClassifier(),
                    "Gaussian Process": GaussianProcessClassifier(),
                    "Gradient Boosting": GradientBoostingClassifier(),
                    "Gaussian Naive Bayes": GaussianNB(),
                    "K-Nearest Neighbors": KNeighborsClassifier(),
                    "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
                    "Logistic Regression": LogisticRegression(),
                    "LightGBM": LGBMClassifier(),
                    "MLP Neural Network": MLPClassifier(),
                    "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
                    "Random Forest": RandomForestClassifier(),
                    "XGBoost": XGBClassifier(),
                }
            except ImportError as e:
                print("导入模型时出现错误:", e)
            selected_model = models[basic_model]

            # Create a fitness function object.
            fitness = jFitnessFunction(
                selected_model, num_feat=x_train.shape[1], alpha=0.9, beta=0.1
            )

            algorithm_mapping = {
                "ACO": jAntColonyOptimization.jAntColonyOptimization(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                ),
                "ACS": jAntColonySystem.jAntColonySystem(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                    phi=phi,
                ),
                "ALO": jAntLionOptimizer.jAntLionOptimizer(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                ),
                "AOA": jArithmeticOptimizationAlgorithm.jArithmeticOptimizationAlgorithm(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                    Alpha=Alpha,
                    Mu=Mu,
                ),
                "ABC": jArtificialBeeColony.jArtificialBeeColony(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                    max_limit=max_limit,
                ),
                "ABO": jArtificialButterflyOptimization.jArtificialButterflyOptimization(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                    step_e=step_e,
                    ratio=ratio,
                    type=type,
                ),
                "ASO": jAtomSearchOptimization.jAtomSearchOptimization(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                ),
                "BA": jBatAlgorithm.jBatAlgorithm(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                    gamma=gamma,
                    A_max=A_max,
                    r0_max=r0_max,
                ),
                "BWO": jBelugaWhaleOptimization.jBelugaWhaleOptimization(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                ),
                "BDMO": jBinaryDwarfMongooseOptimizer.jBinaryDwarfMongooseOptimizer(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                    nBabysitter=nBabysitter,
                    peep=peep,
                ),
                "BOA": jButterflyOptimizationAlgorithm.jButterflyOptimizationAlgorithm(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                    c=c,
                    p=p,
                ),
                "CSAO": jChameleonSwarmAlgorithm.jChameleonSwarmAlgorithm(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                ),
                "CSA": jCrowSearchAlgorithm.jCrowSearchAlgorithm(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                    AP=AP,
                    f1=f1,
                ),
                "CS": jCuckooSearchAlgorithm.jCuckooSearchAlgorithm(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                    Pa=Pa,
                ),
                "DE": jDifferentialEvolution.jDifferentialEvolution(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                    CR=CR,
                    F=F,
                ),
                "DOA": jDingoOptimizationAlgorithm.jDingoOptimizationAlgorithm(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                    P=P,
                    Q=Q,
                ),
                "DBO": jDungBeetleOptimizer.jDungBeetleOptimizer(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                ),
                "DAOA": jDynamicArithmeticOptimizationAlgorithm.jDynamicArithmeticOptimizationAlgorithm(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                    Mu=Mu,
                ),
                "EPO": jEmperorPenguinOptimizer.jEmperorPenguinOptimizer(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                    M=M,
                    f=f,
                    l=l,
                ),
                "EO": jEquilibriumOptimizer.jEquilibriumOptimizer(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                ),
                "FA": jFireflyAlgorithm.jFireflyAlgorithm(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                    beta0=beta0,
                    gamma=gamma,
                    theta=theta,
                ),
                "FPA": jFlowerPollinationAlgorithm.jFlowerPollinationAlgorithm(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                    P=P,
                ),
                "FOA": jFruitFlyOptimizationAlgorithm.jFruitFlyOptimizationAlgorithm(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                ),
                "GNDO": jGeneralizedNormalDistributionOptimization.jGeneralizedNormalDistributionOptimization(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                ),
                "GA": jGeneticAlgorithm.jGeneticAlgorithm(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                    CR=CR,
                    MR=MR,
                ),
                "GAT": jGeneticAlgorithmTour.jGeneticAlgorithmTour(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                    CR=CR,
                    MR=MR,
                    Tour_size=Tour_size,
                ),
                "GSA": jGravitationalSearchAlgorithm.jGravitationalSearchAlgorithm(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                    G0=G0,
                ),
                "GWO": jGreyWolfOptimizer.jGreyWolfOptimizer(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                ),
                "HS": jHarmonySearch.jHarmonySearch(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                    PAR=PAR,
                    HMCR=HMCR,
                    bw=bw,
                ),
                "HHO": jHarrisHawksOptimization.jHarrisHawksOptimization(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                ),
                "HGSO": jHenryGasSolubilityOptimization.jHenryGasSolubilityOptimization(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                    num_gas=num_gas,
                    K=K,
                    L1=L1,
                    L2=L2,
                    L3=L3,
                    c1=c1,
                    c2=c2,
                ),
                "HLO": jHumanLearningOptimization.jHumanLearningOptimization(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                    pi=pi,
                    pr=pr,
                ),
                "HPO": jHunterPreyOptimization.jHunterPreyOptimization(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                    B=B,
                ),
                "JA": jJayaAlgorithm.jJayaAlgorithm(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                ),
                "MRFO": jMantaRayForagingOptimization.jMantaRayForagingOptimization(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                    S=S,
                ),
                "MPA": jMarinePredatorsAlgorithm.jMarinePredatorsAlgorithm(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                    P=P,
                    FADs=FADs,
                ),
                "MBO": jMonarchButterflyOptimization.jMonarchButterflyOptimization(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                    peri=peri,
                    p=p,
                    Smax=Smax,
                    BAR=BAR,
                    num_land1=num_land1,
                ),
                "MFO": jMothFlameOptimization.jMothFlameOptimization(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                    b=b,
                ),
                "MVO": jMultiVerseOptimizer.jMultiVerseOptimizer(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                    p=6,
                    type=1,
                ),
                "PSO": jParticleSwarmOptimization.jParticleSwarmOptimization(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                    c1=c1,
                    c2=c2,
                    w=w,
                ),
                "PFA": jPathFinderAlgorithm.jPathFinderAlgorithm(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                ),
                "PRO": jPoorAndRichOptimization.jPoorAndRichOptimization(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                    Pmut=Pmut,
                ),
                "SSA": jSalpSwarmAlgorithm.jSalpSwarmAlgorithm(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                ),
                "SCSO": jSandCatSwarmOptimization.jSandCatSwarmOptimization(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                ),
                "SBO": jSatinBowerBirdOptimization.jSatinBowerBirdOptimization(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                    z=z,
                    MR=MR,
                ),
                "SA": jSimulatedAnnealing.jSimulatedAnnealing(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                    c=c,
                    T0=T0,
                ),
                "SCA": jSineCosineAlgorithm.jSineCosineAlgorithm(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                ),
                "SMA": jSlimeMouldAlgorithm.jSlimeMouldAlgorithm(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                    z=z,
                ),
                "SOS": jSymbioticOrganismsSearch.jSymbioticOrganismsSearch(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                ),
                "TGA": jTreeGrowthAlgorithm.jTreeGrowthAlgorithm(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                    num_tree1=num_tree1,
                    num_tree2=num_tree2,
                    num_tree4=num_tree4,
                    theta=theta,
                    lambda_=lambda_,
                ),
                "TSA": jTreeSeedAlgorithm.jTreeSeedAlgorithm(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                    ST=ST,
                ),
                "WSA": jWeightedSuperpositionAttraction.jWeightedSuperpositionAttraction(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                    sl=sl,
                    phi=phi,
                    lambda_val=lambda_val,
                ),
                "WOA": jWhaleOptimizationAlgorithm.jWhaleOptimizationAlgorithm(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                    b=b,
                ),
                "WHO": jWildHorseOptimizer.jWildHorseOptimizer(
                    N=N,
                    max_Iter=max_Iter,
                    loss_func=fitness,
                    alpha=alpha,
                    beta=beta,
                    thres=thres,
                    tau=tau,
                    rho=rho,
                    eta=eta,
                ),
            }
            FSModel = algorithm_mapping.get(algorithm)
            history = FSModel.optimize(x_train, x_test, y_train, y_test)
            sf_idx = history["sf"]
            self.loss_data = history["c"]

            self.feature_origin_data = self.dataframe.copy()
            self.feature_origin_data.insert(0, "Labels", self.datalabel)
            self.feature_selection_data = self.dataframe[self.dataframe.columns[sf_idx]]
            self.feature_selection_data.insert(0, "Labels", self.datalabel)

            return True 
        except Exception as e:
            self.error_msg = str(e)
            return False

    def CHI2(self):
        try:
            binBox = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            labels = self.datalabel.tolist()
            data = np.array(self.dataframe.values)
            features = self.dataframe.columns

            if data.shape[0] < 5 or data.shape[1] < 2:
                self.error_msg = "Sample number is two less"
                return False

            sampleNumber = len(data)
            labelClass = set(labels)

            myFea = {}
            for i in range(len(features)):
                array = data[:, i]
                newArray = list(pd.cut(array, len(binBox), labels=binBox))
                binBoxClass = set(newArray)
                myObservation = {}
                for j in range(len(labels)):
                    myObservation[str(labels[j]) + str(newArray[j])] = (
                        myObservation.get(str(labels[j]) + str(newArray[j]), 0) + 1
                    )
                myExpect = {}
                for j in labelClass:
                    for k in binBox:
                        myExpect[str(j) + str(k)] = (
                            labels.count(j) * newArray.count(k) / sampleNumber
                        )
                chiValue = 0
                for j in labelClass:
                    for k in binBoxClass:
                        chiValue = (
                            chiValue
                            + pow(
                                (
                                    (myObservation.get(str(j) + str(k), 0))
                                    - myExpect.get(str(j) + str(k), 0)
                                ),
                                2,
                            )
                            / myExpect[str(j) + str(k)]
                        )
                myFea[features[i]] = chiValue
            res = []
            for key in sorted(myFea.items(), key=lambda item: item[1], reverse=True):
                res.append([key[0], "{0:.3f}".format(myFea[key[0]])])
            self.feature_selection_result = pd.DataFrame(
                res, columns=["SampleName", "Values"]
            )
            selected_feature_number = (
                self.kw["feature_number"]
                if self.kw["feature_number"] <= data.shape[1]
                else data.shape[1]
            )
            self.feature_origin_data = self.dataframe.copy()
            self.feature_origin_data.insert(0, "Labels", self.datalabel)
            self.feature_selection_data = self.dataframe.loc[
                :, self.feature_selection_result.SampleName[:selected_feature_number]
            ]
            self.feature_selection_data.insert(0, "Labels", self.datalabel)
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def calProb(self, array):
        myProb = {}
        myClass = set(array)
        for i in myClass:
            myProb[i] = array.count(i) / len(array)
        return myProb

    def jointProb(self, newArray, labels):
        myJointProb = {}
        for i in range(len(labels)):
            myJointProb[str(newArray[i]) + "-" + str(labels[i])] = (
                myJointProb.get(str(newArray[i]) + "-" + str(labels[i]), 0) + 1
            )

        for key in myJointProb:
            myJointProb[key] = myJointProb[key] / len(labels)
        return myJointProb

    def IG(self):
        try:
            binBox = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            labels = self.datalabel.tolist()
            data = np.array(self.dataframe.values)
            features = self.dataframe.columns

            if data.shape[0] < 5 or data.shape[1] < 2:
                self.error_msg = "Sample number is two less"
                return False

            probY = self.calProb(labels)

            myFea = {}
            for i in range(len(features)):
                array = data[:, i]
                newArray = list(pd.cut(array, len(binBox), labels=binBox))
                binBoxClass = set(newArray)

                probX = self.calProb(newArray)
                probXY = self.jointProb(newArray, labels)
                HX = -1 * sum([p * math.log(p, 2) for p in probX.values()])
                HXY = 0
                for y in probY.keys():
                    for x in probX.keys():
                        if str(x) + "-" + str(y) in probXY:
                            HXY = HXY + (
                                probXY[str(x) + "-" + str(y)]
                                * math.log(probXY[str(x) + "-" + str(y)] / probY[y], 2)
                            )
                myFea[features[i]] = HX + HXY

            res = []
            for key in sorted(myFea.items(), key=lambda item: item[1], reverse=True):
                res.append([key[0], "{0:.3f}".format(myFea[key[0]])])
            self.feature_selection_result = pd.DataFrame(
                res, columns=["SampleName", "Values"]
            )
            selected_feature_number = (
                self.kw["feature_number"]
                if self.kw["feature_number"] <= data.shape[1]
                else data.shape[1]
            )
            self.feature_origin_data = self.dataframe.copy()
            self.feature_origin_data.insert(0, "Labels", self.datalabel)
            self.feature_selection_data = self.dataframe.loc[
                :, self.feature_selection_result.SampleName[:selected_feature_number]
            ]
            self.feature_selection_data.insert(0, "Labels", self.datalabel)
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Calculate_Fscore(self, array, labels):
        try:
            array_po = []
            array_ne = []
            for i in range(len(labels)):
                if labels[i] == 1:
                    array_po.append(array[i])
                else:
                    array_ne.append(array[i])
            mean_po = sum(array_po) / len(array_po)
            mean_ne = sum(array_ne) / len(array_ne)
            mean = sum(array) / len(array)
            score_1 = (mean_po - mean) ** 2 + (mean_ne - mean) ** 2
            score_2 = sum([(i - mean_po) ** 2 for i in array_po]) / (len(array_po) - 1)
            score_3 = sum([(i - mean_ne) ** 2 for i in array_ne]) / (len(array_ne) - 1)
            if score_2 + score_3 == 0:
                return 0
            else:
                f_score = score_1 / (score_2 + score_3)
                return f_score
        except Exception as e:
            return 0

    def FScore(self):
        try:
            labels = self.datalabel.tolist()
            data = np.array(self.dataframe.values)
            features = self.dataframe.columns

            if data.shape[0] < 5 or data.shape[1] < 2:
                self.error_msg = "Sample number is two less"
                return False

            myFea = {}
            for i in range(len(features)):
                array = list(data[:, i])
                myFea[features[i]] = self.Calculate_Fscore(array, labels)

            res = []
            for key in sorted(myFea.items(), key=lambda item: item[1], reverse=True):
                res.append([key[0], "{0:.3f}".format(myFea[key[0]])])
            self.feature_selection_result = pd.DataFrame(
                res, columns=["SampleName", "Values"]
            )
            selected_feature_number = (
                self.kw["feature_number"]
                if self.kw["feature_number"] <= data.shape[1]
                else data.shape[1]
            )
            self.feature_origin_data = self.dataframe.copy()
            self.feature_origin_data.insert(0, "Labels", self.datalabel)
            self.feature_selection_data = self.dataframe.loc[
                :, self.feature_selection_result.SampleName[:selected_feature_number]
            ]
            self.feature_selection_data.insert(0, "Labels", self.datalabel)
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def MIC(self):
        try:
            binBox = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            labels = self.datalabel.tolist()
            data = np.array(self.dataframe.values)
            features = self.dataframe.columns

            if data.shape[0] < 5 or data.shape[1] < 2:
                self.error_msg = "Sample number is two less"
                return False

            probY = self.calProb(labels)
            myFea = {}
            for i in range(len(features)):
                array = data[:, i]
                newArray = list(pd.cut(array, len(binBox), labels=binBox))
                binBoxClass = set(newArray)
                probX = self.calProb(newArray)
                probXY = self.jointProb(newArray, labels)
                mic = 0
                for x in probX.keys():
                    for y in probY.keys():
                        if str(x) + "-" + str(y) in probXY:
                            mic = mic + probXY[str(x) + "-" + str(y)] * math.log(
                                probXY[str(x) + "-" + str(y)] / (probX[x] * probY[y]), 2
                            )
                myFea[features[i]] = mic

            res = []
            for key in sorted(myFea.items(), key=lambda item: item[1], reverse=True):
                res.append([key[0], "{0:.3f}".format(myFea[key[0]])])
            self.feature_selection_result = pd.DataFrame(
                res, columns=["SampleName", "Values"]
            )
            selected_feature_number = (
                self.kw["feature_number"]
                if self.kw["feature_number"] <= data.shape[1]
                else data.shape[1]
            )
            self.feature_origin_data = self.dataframe.copy()
            self.feature_origin_data.insert(0, "Labels", self.datalabel)
            self.feature_selection_data = self.dataframe.loc[
                :, self.feature_selection_result.SampleName[:selected_feature_number]
            ]
            self.feature_selection_data.insert(0, "Labels", self.datalabel)
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def multipl(self, a, b):
        sumofab = 0.0
        for i in range(len(a)):
            temp = a[i] * b[i]
            sumofab += temp
        return sumofab

    def corrcoef(self, x, y):
        try:
            n = len(x)
            sum1 = sum(x)
            sum2 = sum(y)
            sumofxy = self.multipl(x, y)
            sumofx2 = sum([pow(i, 2) for i in x])
            sumofy2 = sum([pow(j, 2) for j in y])
            num = sumofxy - (float(sum1) * float(sum2) / n)
            den = math.sqrt(
                (sumofx2 - float(sum1**2) / n) * (sumofy2 - float(sum2**2) / n)
            )
            if den != 0:
                return num / den
            else:
                return 0
        except Exception as e:
            return 0

    def Pearsonr(self):
        try:
            labels = self.datalabel.tolist()
            data = np.array(self.dataframe.values)
            features = self.dataframe.columns

            if data.shape[0] < 5 or data.shape[1] < 2:
                self.error_msg = "Sample number is two less"
                return False

            myFea = {}
            for i in range(len(features)):
                array = list(data[:, i])
                myFea[features[i]] = self.corrcoef(array, labels)

            res = []
            for key in sorted(myFea.items(), key=lambda item: item[1], reverse=True):
                res.append([key[0], "{0:.3f}".format(myFea[key[0]])])
            self.feature_selection_result = pd.DataFrame(
                res, columns=["SampleName", "Values"]
            )
            selected_feature_number = (
                self.kw["feature_number"]
                if self.kw["feature_number"] <= data.shape[1]
                else data.shape[1]
            )
            self.feature_origin_data = self.dataframe.copy()
            self.feature_origin_data.insert(0, "Labels", self.datalabel)
            self.feature_selection_data = self.dataframe.loc[
                :, self.feature_selection_result.SampleName[:selected_feature_number]
            ]
            self.feature_selection_data.insert(0, "Labels", self.datalabel)
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def fill_ndarray(self, t1):
        for i in range(t1.shape[1]):
            temp_col = t1[:, i]
            nan_num = np.count_nonzero(temp_col != temp_col)
            if nan_num != 0:
                temp_not_nan_col = temp_col[temp_col == temp_col]
                temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()
        return t1

    def ZScore(self):
        try:
            backup = self.dataframe.copy()
            self.feature_origin_data = self.dataframe.copy()
            self.feature_origin_data.insert(0, "Labels", self.datalabel)
            data = self.dataframe.values
            std_array = np.std(data, axis=0)
            mean_array = np.mean(data, axis=0)
            for i in range(len(mean_array)):
                if std_array[i] != 0:
                    data[:, i] = (data[:, i] - mean_array[i]) / std_array[i]
                else:
                    data[:, i] = 0
            self.feature_normalization_data = pd.DataFrame(
                data, columns=self.dataframe.columns
            )
            self.feature_normalization_data.insert(0, "Labels", self.datalabel)

            self.dataframe = backup
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def MinMax(self):
        try:
            backup = self.dataframe.copy()
            self.feature_origin_data = self.dataframe.copy()
            self.feature_origin_data.insert(0, "Labe", self.datalabel)
            data = self.dataframe.values
            for i in range(len(data[0])):
                maxValue, minValue = max(data[:, i]), min(data[:, i])
                data[:, i] = (data[:, i] - minValue) / (maxValue - minValue)
            # replace NaN value with mean
            data = self.fill_ndarray(data.T).T
            self.feature_normalization_data = pd.DataFrame(
                data, columns=self.dataframe.columns
            )
            self.feature_normalization_data.insert(0, "Label", self.datalabel)

            self.dataframe = backup
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def save_selected_data(self, file, analysis_type):
        data = None
        if analysis_type == "Feature selection algorithms":
            data = self.feature_selection_data
        else:
            data = self.feature_normalization_data

        if not data is None:
            if file.endswith(".csv"):
                data.to_csv(file, sep=",", header=False, index=False)
            elif file.endswith(".tsv"):
                data.to_csv(file, sep="\t", header=False, index=False)
            elif file.endswith(".tsv1"):
                data.to_csv(file, sep="\t", header=True, index=True)
            elif file.endswith(".svm"):
                with open(file, "w") as f:
                    for line in data.values:
                        f.write("%d" % line[0])
                        for i in range(1, len(line)):
                            f.write("  %d:%s" % (i, line[i]))
                        f.write("\n")
            elif file.endswith(".arff"):
                with open(file, "w") as f:
                    f.write("@relation descriptor\n\n")
                    for i in range(1, data.values.shape[1]):
                        f.write("@attribute f.%d numeric\n" % i)
                    f.write("@attribute play {yes, no}\n\n")
                    f.write("@data\n")
                    for line in data.values:
                        for fea in line[1:]:
                            f.write("%s," % fea)
                        if line[0] == 1:
                            f.write("yes\n")
                        else:
                            f.write("no\n")
            else:
                QMessageBox.critical(
                    self,
                    "Error",
                    "Save file failed!",
                    QMessageBox.Ok | QMessageBox.No,
                    QMessageBox.Ok,
                )
        else:
            QMessageBox.critical(
                self,
                "Error",
                "Empty data!",
                QMessageBox.Ok | QMessageBox.No,
                QMessageBox.Ok,
            )


if __name__ == "__main__":
    kw = {"nclusters": 2}
    data = SIFSData(kw)
    data.load_data_from_file("../data/data.csv")
    data.kmeans()
    plot_data = data.generate_plot_data(data.cluster_result, data.cluster_plot_data)
    print(plot_data)
