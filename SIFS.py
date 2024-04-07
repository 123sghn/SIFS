# GUI for the SIFS module

import sys, os, re
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QFileDialog,
    QLabel,
    QHBoxLayout,
    QGroupBox,
    QTextEdit,
    QVBoxLayout,
    QLineEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QSplitter,
    QTableWidget,
    QTabWidget,
    QTableWidgetItem,
    QInputDialog,
    QMessageBox,
    QFormLayout,
    QGridLayout,
    QRadioButton,
    QHeaderView,
    QAbstractItemView,
    QLabel,
)
from PyQt5.QtGui import QIcon, QFont, QMovie
from PyQt5.QtCore import Qt, pyqtSignal
from sklearn.linear_model import LogisticRegression
from util import DataAnalysis, InputDialog, MachineLearning, TableWidget, PlotWidgets

import numpy as np
import pandas as pd
import threading
import qdarkstyle
import sip
import joblib
import torch
import copy


class SIFS(QTabWidget):
    selection_signal = pyqtSignal()
    clust_signal = pyqtSignal()
    ml_signal = pyqtSignal()
    close_signal = pyqtSignal(str)

    def __init__(self):
        super(SIFS, self).__init__()

        # signal
        self.clust_signal.connect(self.display_data_analysis)
        self.selection_signal.connect(self.display_selection_data)
        self.ml_signal.connect(self.display_ml_data)

        self.gif = QMovie("images/progress_bar.gif")

        """ Cluster Variable """
        self.cluster_file = ""
        self.clust_data = None
        self.clust_analysis_type = ""
        self.clust_selected_algorithm = ""
        self.clust_default_para = {
            "nclusters": 2,
            "n_components": 2,
            "expand_factor": 2,
            "inflate_factor": 2.0,
            "multiply_factor": 2.0,
        }

        self.clust_status = False
        self.clust_symbol = {
            0: "o",
            1: "s",
            2: "t",
            3: "+",
            4: "p",
            5: "t2",
            6: "h",
            7: "t3",
            8: "star",
            9: "t1",
            10: "t2",
        }

        """ Feature Selection Variable  """
        self.selection_file = ""
        self.selection_data = None
        self.selection_analysis_type = ""
        self.selection_selected_algorithm = ""
        self.selection_running_status = False
        self.selection_default_para = {
            "feature_number": 5,
            "N": 60,
            "max_Iter": 30,
            "basic_model": LogisticRegression(),
            "alpha": 0.9,
            "beta": 0.1,
            "thres": 0.5,
            "tau": 1,
            "rho": 0.2,
            "eta": 1,
            "phi": None,
            "Alpha": 5,
            "Mu": None,
            "max_limit": 5,
            "step_e": 0.05,
            "ratio": 0.2,
            "type": None,
            "gamma": None,
            "A_max": 2,
            "r0_max": 1,
            "nBabysitter": 3,
            "peep": 1,
            "c": None,
            "p": None,
            "AP": 0.1,
            "f1": 1.5,
            "Pa": 0.25,
            "CR": None,
            "F": 0.5,
            "P": None,
            "Q": 0.7,
            "M": 2,
            "f": 3,
            "l": 2,
            "beta0": 1,
            "theta": None,
            "MR": None,
            "Tour_size": 3,
            "PAR": 0.05,
            "HMCR": 0.7,
            "bw": 0.2,
            "num_gas": 2,
            "K": 1,
            "L1": 0.005,
            "L2": 100,
            "L3": 0.01,
            "c1": None,
            "c2": None,
            "pi": 0.85,
            "pr": 0.1,
            "B": 0.7,
            "S": 2,
            "FADs": 0.2,
            "peri": 2.0,
            "Smax": 1,
            "BAR": 0.5,
            "num_land1": 4,
            "b": None,
            "w": 0.9,
            "Pmut": 0.06,
            "z": None,
            "T0": 100,
            "ST": 0.1,
            "num_tree1": 3,
            "num_tree2": 3,
            "num_tree4": 3,
            "lambda_": 0.5,
            "sl": 0.035,
            "lambda_val": 0.75,
        }

        """ Machine Learning Variable """
        self.MLData = None
        self.MLAlgorithm = None
        self.fold_num = 5
        self.data_index = {
            "Training_data": None,
            "Testing_data": None,
            "Training_score": None,
            "Testing_score": None,
            "Metrics": None,
            "ROC": None,
            "PRC": None,
            "Model": None,
        }
        self.current_data_index = 0
        self.ml_running_status = False
        self.ml_defatult_para = {
            "FOLD": 5,
            "cpu": 1,
            "auto": False,
            "n_trees": 100,
            "tree_range": (100, 1000, 100),
            "kernel": "rbf",
            "penality": 1.0,
            "gamma": "auto",
            "penalityRange": (1.0, 15.0),
            "gammaRange": (-10.0, 5.0),
            "layer": "32;32",
            "activation": "relu",
            "optimizer": "adam",
            "topKValue": 3,
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "max_depth": -1,
            "learning_rate": 0.01,
            "leaves_range": (20, 100, 10),
            "depth_range": (15, 55, 10),
            "rate_range": (0.01, 0.15, 0.02),
            "booster": "gbtree",
            "n_estimator": 100,
            "colsample_bytree": 0.8,
            "input_channel": 1,
            "input_length": 100,
            "output_channel": 64,
            "padding": 2,
            "kernel_size": 5,
            "dropout": 0.5,
            "epochs": 1000,
            "early_stopping": 100,
            "batch_size": 64,
            "fc_size": 64,
            "rnn_hidden_size": 32,
            "rnn_hidden_layers": 1,
            "rnn_bidirection": False,
            "mlp_input_dim": None,
        }

        # initialize UI
        self.initUI()

    def initUI(self):
        self.setWindowTitle("SIFS")
        self.resize(800, 600)
        self.setWindowState(Qt.WindowMaximized)
        self.setWindowIcon(QIcon("images/logo.ico"))

        """ four QWidget """
        self.tab_selection = QWidget()
        self.tab_cluster = QWidget()
        self.tab_machine = QWidget()
        self.addTab(self.tab_selection, " Feature Normalization/Selection ")
        self.addTab(self.tab_cluster, " Cluster / Dimensionality Reduction ")
        self.addTab(self.tab_machine, " Machine Learning ")

        """ Initialize tab """
        self.setup_tab_selection()
        self.setup_tab_cluster()
        self.setup_tab_machinelearning()

    """ setup tab UI """

    def setup_tab_cluster(self):
        # file
        topGroupBox = QGroupBox("Load data", self)
        topGroupBox.setFont(QFont("Arial", 10))
        topGroupBox.setMinimumHeight(100)
        topGroupBoxLayout = QGridLayout()

        self.clust_file_lineEdit = QLineEdit()
        self.clust_file_lineEdit.setFont(QFont("Arial", 8))
        self.clust_file_button = QPushButton("Open")
        self.clust_file_button.setFont(QFont("Arial", 10))
        self.clust_file_button.clicked.connect(self.data_from_file)
        self.clust_data_lineEdit = QLineEdit()
        self.clust_data_lineEdit.setFont(QFont("Arial", 8))
        self.clust_data_button = QPushButton("Select")
        self.clust_data_button.clicked.connect(self.data_from_descriptor)
        self.clust_label2 = QLabel("Data shape: ")
        topGroupBoxLayout.addWidget(self.clust_file_lineEdit, 0, 0)
        topGroupBoxLayout.addWidget(self.clust_file_button, 0, 1)
        topGroupBoxLayout.addWidget(self.clust_data_lineEdit, 1, 0)
        topGroupBoxLayout.addWidget(self.clust_data_button, 1, 1)
        topGroupBoxLayout.addWidget(self.clust_label2, 2, 0, 1, 2)
        topGroupBox.setLayout(topGroupBoxLayout)

        # tree
        treeGroupBox = QGroupBox("Analysis algorithms", self)
        treeGroupBox.setFont(QFont("Arial", 10))
        treeLayout = QHBoxLayout()
        self.clust_treeWidget = QTreeWidget()
        self.clust_treeWidget.setColumnCount(2)
        self.clust_treeWidget.setMinimumWidth(300)
        self.clust_treeWidget.setColumnWidth(0, 150)
        self.clust_treeWidget.setFont(QFont("Arial", 8))
        self.clust_treeWidget.setHeaderLabels(["Methods", "Definition"])
        self.clusterMethods = QTreeWidgetItem(self.clust_treeWidget)
        self.clusterMethods.setExpanded(True)  # set node expanded
        self.clusterMethods.setText(0, "Cluster algorithms")
        self.clust_treeWidget.clicked.connect(self.clust_tree_clicked)
        kmeans = QTreeWidgetItem(self.clusterMethods)
        kmeans.setText(0, "kmeans")
        kmeans.setText(1, "kmeans clustering")
        minikmeans = QTreeWidgetItem(self.clusterMethods)
        minikmeans.setText(0, "MiniBatchKMeans")
        minikmeans.setText(1, "MiniBatchKMeans clustering")
        gmm = QTreeWidgetItem(self.clusterMethods)
        gmm.setText(0, "GM")
        gmm.setText(1, "Gaussian mixture clustering")
        agg = QTreeWidgetItem(self.clusterMethods)
        agg.setText(0, "Agglomerative")
        agg.setText(1, "Agglomerative clustering")
        spectral = QTreeWidgetItem(self.clusterMethods)
        spectral.setText(0, "Spectral")
        spectral.setText(1, "Spectral clustering")
        mcl = QTreeWidgetItem(self.clusterMethods)
        mcl.setText(0, "MCL")
        mcl.setText(1, "Markov Cluster algorithm")
        hcluster = QTreeWidgetItem(self.clusterMethods)
        hcluster.setText(0, "hcluster")
        hcluster.setText(1, "Hierarchical clustering")
        apc = QTreeWidgetItem(self.clusterMethods)
        apc.setText(0, "APC")
        apc.setText(1, "Affinity Propagation Clustering")
        meanshift = QTreeWidgetItem(self.clusterMethods)
        meanshift.setText(0, "meanshift")
        meanshift.setText(1, "Mean-shift Clustering")
        dbscan = QTreeWidgetItem(self.clusterMethods)
        dbscan.setText(0, "DBSCAN")
        dbscan.setText(1, "DBSCAN Clustering")
        self.dimensionReduction = QTreeWidgetItem(self.clust_treeWidget)
        self.dimensionReduction.setExpanded(True)  # set node expanded
        self.dimensionReduction.setText(0, "Dimensionality reduction algorithms")
        pca = QTreeWidgetItem(self.dimensionReduction)
        pca.setText(0, "PCA")
        pca.setText(1, "Principal component analysis")
        tsne = QTreeWidgetItem(self.dimensionReduction)
        tsne.setText(0, "t_SNE")
        tsne.setText(1, "t-distributed Stochastic Neighbor Embedding")
        lda = QTreeWidgetItem(self.dimensionReduction)
        lda.setText(0, "LDA")
        lda.setText(1, "Latent Dirichlet Allocation")
        treeLayout.addWidget(self.clust_treeWidget)
        treeGroupBox.setLayout(treeLayout)

        ## parameter
        paraGroupBox = QGroupBox("Parameters", self)
        paraGroupBox.setMaximumHeight(150)
        paraGroupBox.setFont(QFont("Arial", 10))
        paraLayout = QFormLayout(paraGroupBox)
        self.clust_analysisType_lineEdit = QLineEdit()
        self.clust_analysisType_lineEdit.setFont(QFont("Arial", 8))
        self.clust_analysisType_lineEdit.setEnabled(False)
        paraLayout.addRow("Analysis:", self.clust_analysisType_lineEdit)
        self.cluster_algorithm_lineEdit = QLineEdit()
        self.cluster_algorithm_lineEdit.setFont(QFont("Arial", 8))
        self.cluster_algorithm_lineEdit.setEnabled(False)
        paraLayout.addRow("Algorithm:", self.cluster_algorithm_lineEdit)
        self.clust_para_lineEdit = QLineEdit()
        self.clust_para_lineEdit.setFont(QFont("Arial", 8))
        self.clust_para_lineEdit.setEnabled(False)
        paraLayout.addRow("Parameter(s):", self.clust_para_lineEdit)

        ## start button
        startGroupBox = QGroupBox("Operator", self)
        startGroupBox.setFont(QFont("Arial", 10))
        startLayout = QHBoxLayout(startGroupBox)
        self.clust_start_button = QPushButton("Start")
        self.clust_start_button.clicked.connect(self.run_data_analysis)
        self.clust_start_button.setFont(QFont("Arial", 10))
        self.clust_save_button = QPushButton("Save txt")
        self.clust_save_button.setFont(QFont("Arial", 10))
        self.clust_save_button.clicked.connect(self.save_cluster_rd)
        startLayout.addWidget(self.clust_start_button)
        startLayout.addWidget(self.clust_save_button)

        ### layout
        left_vertical_layout = QVBoxLayout()
        left_vertical_layout.addWidget(topGroupBox)
        left_vertical_layout.addWidget(treeGroupBox)
        left_vertical_layout.addWidget(paraGroupBox)
        left_vertical_layout.addWidget(startGroupBox)

        #### widget
        leftWidget = QWidget()
        leftWidget.setLayout(left_vertical_layout)

        #### view region
        self.clust_tableWidget = QTableWidget()
        self.clust_tableWidget.setFont(QFont("Arial", 8))
        self.clust_diagram = PlotWidgets.ClusteringDiagramMatplotlib()
        clust_diagram_widget = QWidget()
        self.clust_diagram_layout = QVBoxLayout(clust_diagram_widget)
        self.clust_diagram_layout.addWidget(self.clust_diagram)
        clust_tabWidget = QTabWidget()
        clust_tabWidget.addTab(self.clust_tableWidget, " Result ")
        clust_tabWidget.addTab(clust_diagram_widget, " Scatter plot ")

        ##### splitter
        splitter_1 = QSplitter(Qt.Horizontal)
        splitter_1.addWidget(leftWidget)
        splitter_1.addWidget(clust_tabWidget)
        splitter_1.setSizes([100, 1200])

        ##### vertical layout
        vLayout = QVBoxLayout()

        ## status bar
        statusGroupBox = QGroupBox("Status", self)
        statusGroupBox.setFont(QFont("Arial", 10))
        statusLayout = QHBoxLayout(statusGroupBox)
        self.clust_status_label = QLabel("Welcome to SIFS")
        self.clust_progress_bar = QLabel()
        self.clust_progress_bar.setMaximumWidth(230)
        statusLayout.addWidget(self.clust_status_label)
        statusLayout.addWidget(self.clust_progress_bar)

        splitter_2 = QSplitter(Qt.Vertical)
        splitter_2.addWidget(splitter_1)
        splitter_2.addWidget(statusGroupBox)
        splitter_2.setSizes([1000, 100])
        vLayout.addWidget(splitter_2)
        self.tab_cluster.setLayout(vLayout)

    def setup_tab_selection(self):
        # file
        topGroupBox = QGroupBox("Load data", self)
        topGroupBox.setFont(QFont("Arial", 10))
        topGroupBox.setMinimumHeight(100)
        topGroupBoxLayout = QGridLayout()
        self.selection_file_lineEdit = QLineEdit()
        self.selection_file_lineEdit.setFont(QFont("Arial", 8))
        self.selection_file_button = QPushButton("Open")
        self.selection_file_button.clicked.connect(self.data_from_file_s)
        self.selection_file_button.setFont(QFont("Arial", 10))
        self.selection_data_lineEdit = QLineEdit()
        self.selection_data_lineEdit.setFont(QFont("Arial", 8))
        self.selection_data_button = QPushButton("Select")
        self.selection_data_button.clicked.connect(self.data_from_panel_s)
        self.selection_label2 = QLabel("Data shape: ")
        topGroupBoxLayout.addWidget(self.selection_file_lineEdit, 0, 0)
        topGroupBoxLayout.addWidget(self.selection_file_button, 0, 1)
        topGroupBoxLayout.addWidget(self.selection_data_lineEdit, 1, 0)
        topGroupBoxLayout.addWidget(self.selection_data_button, 1, 1)
        topGroupBoxLayout.addWidget(self.selection_label2, 2, 0, 1, 2)
        topGroupBox.setLayout(topGroupBoxLayout)

        # tree
        treeGroupBox = QGroupBox("Analysis algorithms", self)
        treeGroupBox.setFont(QFont("Arial", 10))
        treeLayout = QHBoxLayout()
        self.selection_treeWidget = QTreeWidget()
        self.selection_treeWidget.setColumnCount(2)
        self.selection_treeWidget.setMinimumWidth(300)
        self.selection_treeWidget.setColumnWidth(0, 150)
        self.selection_treeWidget.setFont(QFont("Arial", 8))
        self.selection_treeWidget.setHeaderLabels(["Methods", "Definition"])
        self.selectionMethods = QTreeWidgetItem(self.selection_treeWidget)
        self.selectionMethods.setExpanded(True)  # set node expanded
        self.selectionMethods.setText(0, "Feature selection algorithms")
        self.selection_treeWidget.clicked.connect(self.selection_tree_clicked)

        ACO = QTreeWidgetItem(self.selectionMethods)
        ACO.setText(0, "ACO")
        ACO.setText(1, "Ant Colony Optimization")
        ACS = QTreeWidgetItem(self.selectionMethods)
        ACS.setText(0, "ACS")
        ACS.setText(1, "Ant Colony System")
        ALO = QTreeWidgetItem(self.selectionMethods)
        ALO.setText(0, "ALO")
        ALO.setText(1, "Ant Lion Optimizer")
        AOA = QTreeWidgetItem(self.selectionMethods)
        AOA.setText(0, "AOA")
        AOA.setText(1, "Arithmetic Optimization Algorithm")
        ABC = QTreeWidgetItem(self.selectionMethods)
        ABC.setText(0, "ABC")
        ABC.setText(1, "Artificial Bee Colony")
        ABO = QTreeWidgetItem(self.selectionMethods)
        ABO.setText(0, "ABO")
        ABO.setText(1, "Artificial Butterfly Optimization")
        ASO = QTreeWidgetItem(self.selectionMethods)
        ASO.setText(0, "ASO")
        ASO.setText(1, "Atom Search Optimization")
        BA = QTreeWidgetItem(self.selectionMethods)
        BA.setText(0, "BA")
        BA.setText(1, "Bat Algorithm")
        BWO = QTreeWidgetItem(self.selectionMethods)
        BWO.setText(0, "BWO")
        BWO.setText(1, "Beluga Whale Optimization")
        BDMO = QTreeWidgetItem(self.selectionMethods)
        BDMO.setText(0, "BDMO")
        BDMO.setText(1, "Binary Dwarf Mongoose Optimizer")
        BOA = QTreeWidgetItem(self.selectionMethods)
        BOA.setText(0, "BOA")
        BOA.setText(1, "Butterfly Optimization Algorithm")
        CSAO = QTreeWidgetItem(self.selectionMethods)
        CSAO.setText(0, "CSAO")
        CSAO.setText(1, "Chameleon Swarm Algorithm Optimization")
        CSA = QTreeWidgetItem(self.selectionMethods)
        CSA.setText(0, "CSA")
        CSA.setText(1, "Crow Search Algorithm")
        CS = QTreeWidgetItem(self.selectionMethods)
        CS.setText(0, "CS")
        CS.setText(1, "Cuckoo Search Algorithm")
        DE = QTreeWidgetItem(self.selectionMethods)
        DE.setText(0, "DE")
        DE.setText(1, "Differential Evolution")
        DOA = QTreeWidgetItem(self.selectionMethods)
        DOA.setText(0, "DOA")
        DOA.setText(1, "Dingo Optimization Algorithm")
        DBO = QTreeWidgetItem(self.selectionMethods)
        DBO.setText(0, "DBO")
        DBO.setText(1, "Dung Beetle Optimizer")
        DAOA = QTreeWidgetItem(self.selectionMethods)
        DAOA.setText(0, "DAOA")
        DAOA.setText(1, "Dynamic Arithmetic Optimization Algorithm")
        EPO = QTreeWidgetItem(self.selectionMethods)
        EPO.setText(0, "EPO")
        EPO.setText(1, "Emperor Penguin Optimizer")
        EO = QTreeWidgetItem(self.selectionMethods)
        EO.setText(0, "EO")
        EO.setText(1, "Equilibrium Optimizer")
        FA = QTreeWidgetItem(self.selectionMethods)
        FA.setText(0, "FA")
        FA.setText(1, "Firefly Algorithm")
        FPA = QTreeWidgetItem(self.selectionMethods)
        FPA.setText(0, "FPA")
        FPA.setText(1, "Flower Pollination Algorithm")
        FOA = QTreeWidgetItem(self.selectionMethods)
        FOA.setText(0, "FOA")
        FOA.setText(1, "FruitFly Optimization Algorithm")
        GNDO = QTreeWidgetItem(self.selectionMethods)
        GNDO.setText(0, "GNDO")
        GNDO.setText(1, "Generalized Normal Distribution Optimization")
        GA = QTreeWidgetItem(self.selectionMethods)
        GA.setText(0, "GA")
        GA.setText(1, "Genetic Algorithm")
        GAT = QTreeWidgetItem(self.selectionMethods)
        GAT.setText(0, "GAT")
        GAT.setText(1, "Genetic Algorithm Tour")
        GSA = QTreeWidgetItem(self.selectionMethods)
        GSA.setText(0, "GSA")
        GSA.setText(1, "Gravitational Search Algorithm")
        GWO = QTreeWidgetItem(self.selectionMethods)
        GWO.setText(0, "GWO")
        GWO.setText(1, "Grey Wolf Optimizer")
        HS = QTreeWidgetItem(self.selectionMethods)
        HS.setText(0, "HS")
        HS.setText(1, "Harmony Search")
        HHO = QTreeWidgetItem(self.selectionMethods)
        HHO.setText(0, "HHO")
        HHO.setText(1, "Harris Hawks Optimization")
        HGSO = QTreeWidgetItem(self.selectionMethods)
        HGSO.setText(0, "HGSO")
        HGSO.setText(1, "Henry Gas Solubility Optimization")
        HLO = QTreeWidgetItem(self.selectionMethods)
        HLO.setText(0, "HLO")
        HLO.setText(1, "Human Learning Optimization")
        HPO = QTreeWidgetItem(self.selectionMethods)
        HPO.setText(0, "HPO")
        HPO.setText(1, "Hunter Prey Optimization")
        JA = QTreeWidgetItem(self.selectionMethods)
        JA.setText(0, "JA")
        JA.setText(1, "Jaya Algorithm")
        MRFO = QTreeWidgetItem(self.selectionMethods)
        MRFO.setText(0, "MRFO")
        MRFO.setText(1, "Manta Ray Foraging Optimization")
        MPA = QTreeWidgetItem(self.selectionMethods)
        MPA.setText(0, "MPA")
        MPA.setText(1, "Marine Predators Algorithm")
        MBO = QTreeWidgetItem(self.selectionMethods)
        MBO.setText(0, "MBO")
        MBO.setText(1, "Monarch Butterfly Optimization")
        MFO = QTreeWidgetItem(self.selectionMethods)
        MFO.setText(0, "MFO")
        MFO.setText(1, "Moth Flame Optimization")
        MVO = QTreeWidgetItem(self.selectionMethods)
        MVO.setText(0, "MVO")
        MVO.setText(1, "Multi Verse Optimizer")
        PSO = QTreeWidgetItem(self.selectionMethods)
        PSO.setText(0, "PSO")
        PSO.setText(1, "Particle Swarm Optimization")
        PFA = QTreeWidgetItem(self.selectionMethods)
        PFA.setText(0, "PFA")
        PFA.setText(1, "Path Finder Algorithm")
        PRO = QTreeWidgetItem(self.selectionMethods)
        PRO.setText(0, "PRO")
        PRO.setText(1, "PoorAnd Rich Optimization")
        SSA = QTreeWidgetItem(self.selectionMethods)
        SSA.setText(0, "SSA")
        SSA.setText(1, "Salp Swarm Algorithm")
        SCSO = QTreeWidgetItem(self.selectionMethods)
        SCSO.setText(0, "SCSO")
        SCSO.setText(1, "Sand Cat Swarm Optimization")
        SBO = QTreeWidgetItem(self.selectionMethods)
        SBO.setText(0, "SBO")
        SBO.setText(1, "Satin BowerBird Optimization")
        SA = QTreeWidgetItem(self.selectionMethods)
        SA.setText(0, "SA")
        SA.setText(1, "Simulated Annealing")
        SCA = QTreeWidgetItem(self.selectionMethods)
        SCA.setText(0, "SCA")
        SCA.setText(1, "Sine Cosine Algorithm")
        SMA = QTreeWidgetItem(self.selectionMethods)
        SMA.setText(0, "SMA")
        SMA.setText(1, "Slime Mould Algorithm")
        SOS = QTreeWidgetItem(self.selectionMethods)
        SOS.setText(0, "SOS")
        SOS.setText(1, "Symbiotic Organisms Search")
        TGA = QTreeWidgetItem(self.selectionMethods)
        TGA.setText(0, "TGA")
        TGA.setText(1, "Tree Growth Algorithm")
        TSA = QTreeWidgetItem(self.selectionMethods)
        TSA.setText(0, "TSA")
        TSA.setText(1, "Tree Seed Algorithm")
        WSA = QTreeWidgetItem(self.selectionMethods)
        WSA.setText(0, "WSA")
        WSA.setText(1, "Weighted Superposition Attraction")
        WOA = QTreeWidgetItem(self.selectionMethods)
        WOA.setText(0, "WOA")
        WOA.setText(1, "Whale Optimization Algorithm")
        WHO = QTreeWidgetItem(self.selectionMethods)
        WHO.setText(0, "WHO")
        WHO.setText(1, "Wild Horse Optimizer")

        CHI2 = QTreeWidgetItem(self.selectionMethods)
        CHI2.setText(0, "CHI2")
        CHI2.setText(1, "Chi-Square feature selection")
        IG = QTreeWidgetItem(self.selectionMethods)
        IG.setText(0, "IG")
        IG.setText(1, "Information Gain feature selection")
        FScore = QTreeWidgetItem(self.selectionMethods)
        FScore.setText(0, "FScore")
        FScore.setText(1, "F-score value")
        MIC = QTreeWidgetItem(self.selectionMethods)
        MIC.setText(0, "MIC")
        MIC.setText(1, "Mutual Information feature selection")
        Pearsonr = QTreeWidgetItem(self.selectionMethods)
        Pearsonr.setText(0, "Pearsonr")
        Pearsonr.setText(1, "Pearson Correlation coefficient")

        self.normalizationMethods = QTreeWidgetItem(self.selection_treeWidget)
        self.normalizationMethods.setExpanded(True)  # set node expanded
        self.normalizationMethods.setText(0, "Feature Normalization algorithms")
        ZScore = QTreeWidgetItem(self.normalizationMethods)
        ZScore.setText(0, "ZScore")
        ZScore.setText(1, "ZScore")
        MinMax = QTreeWidgetItem(self.normalizationMethods)
        MinMax.setText(0, "MinMax")
        MinMax.setText(1, "MinMax")
        treeLayout.addWidget(self.selection_treeWidget)
        treeGroupBox.setLayout(treeLayout)

        ## parameter
        paraGroupBox = QGroupBox("Parameters", self)
        paraGroupBox.setMaximumHeight(150)
        paraGroupBox.setFont(QFont("Arial", 10))
        paraLayout = QFormLayout(paraGroupBox)
        self.selection_analysisType_lineEdit = QLineEdit()
        self.selection_analysisType_lineEdit.setFont(QFont("Arial", 8))
        self.selection_analysisType_lineEdit.setEnabled(False)
        paraLayout.addRow("Analysis:", self.selection_analysisType_lineEdit)
        self.selection_algorithm_lineEdit = QLineEdit()
        self.selection_algorithm_lineEdit.setFont(QFont("Arial", 8))
        self.selection_algorithm_lineEdit.setEnabled(False)
        paraLayout.addRow("Algorithm:", self.selection_algorithm_lineEdit)
        self.selection_para_lineEdit = QLineEdit()
        self.selection_para_lineEdit.setFont(QFont("Arial", 8))
        self.selection_para_lineEdit.setEnabled(False)
        paraLayout.addRow("Parameter(s):", self.selection_para_lineEdit)

        ## start button
        startGroupBox = QGroupBox("Operator", self)
        startGroupBox.setFont(QFont("Arial", 10))
        startLayout = QHBoxLayout(startGroupBox)
        self.selection_start_button = QPushButton("Start")
        self.selection_start_button.clicked.connect(self.run_selection)
        self.selection_start_button.setFont(QFont("Arial", 10))
        self.selection_save_button = QPushButton("Save csv")
        self.selection_save_button.setFont(QFont("Arial", 10))
        self.selection_save_button.clicked.connect(
            self.save_selection_normalization_data
        )
        startLayout.addWidget(self.selection_start_button)
        startLayout.addWidget(self.selection_save_button)

        ### layout
        left_vertical_layout = QVBoxLayout()
        left_vertical_layout.addWidget(topGroupBox)
        left_vertical_layout.addWidget(treeGroupBox)
        left_vertical_layout.addWidget(paraGroupBox)
        left_vertical_layout.addWidget(startGroupBox)

        #### widget
        leftWidget = QWidget()
        leftWidget.setLayout(left_vertical_layout)

        #### view region
        selection_viewWidget = QTabWidget()
        self.selection_tableWidget = TableWidget.TableWidgetForSelPanel()
        selection_histWidget = QWidget()
        self.selection_histLayout = QVBoxLayout(selection_histWidget)
        self.selection_hist = PlotWidgets.HistogramWidget()
        self.selection_histLayout.addWidget(self.selection_hist)

        self.selection_tableWidget_origin = TableWidget.TableWidgetForSelPanel()
        selection_histWidget_origin = QWidget()
        self.selection_histLayout_origin = QVBoxLayout(selection_histWidget_origin)
        self.selection_hist_origin = PlotWidgets.HistogramWidget()
        self.selection_histLayout_origin.addWidget(self.selection_hist_origin)

        selection_histWidget_loss_curve = QWidget()
        self.selection_histLayout_loss_curve = QVBoxLayout(
            selection_histWidget_loss_curve
        )
        self.selection_hist_loss_curve = PlotWidgets.LossWidget()
        self.selection_histLayout_loss_curve.addWidget(self.selection_hist_loss_curve)

        selection_viewWidget.addTab(self.selection_tableWidget_origin, " Origin Data ")
        selection_viewWidget.addTab(
            selection_histWidget_origin, " Origin Data distribution "
        )
        selection_viewWidget.addTab(self.selection_tableWidget, " Data ")
        selection_viewWidget.addTab(selection_histWidget, " Data distribution ")
        selection_viewWidget.addTab(selection_histWidget_loss_curve, " Loss curve ")

        ##### splitter
        splitter_1 = QSplitter(Qt.Horizontal)
        splitter_1.addWidget(leftWidget)
        splitter_1.addWidget(selection_viewWidget)
        splitter_1.setSizes([100, 1000])

        ###### vertical layout
        vLayout = QVBoxLayout()

        ## status bar
        statusGroupBox = QGroupBox("Status", self)
        statusGroupBox.setFont(QFont("Arial", 10))
        statusLayout = QHBoxLayout(statusGroupBox)
        self.selection_status_label = QLabel("Welcome to SIFS")
        self.selection_progress_bar = QLabel()
        self.selection_progress_bar.setMaximumWidth(230)
        statusLayout.addWidget(self.selection_status_label)
        statusLayout.addWidget(self.selection_progress_bar)

        splitter_2 = QSplitter(Qt.Vertical)
        splitter_2.addWidget(splitter_1)
        splitter_2.addWidget(statusGroupBox)
        splitter_2.setSizes([1000, 100])
        vLayout.addWidget(splitter_2)
        self.tab_selection.setLayout(vLayout)

    def setup_tab_machinelearning(self):
        # file
        topGroupBox = QGroupBox("Load data", self)
        topGroupBox.setFont(QFont("Arial", 10))
        topGroupBox.setMinimumHeight(100)
        topGroupBoxLayout = QFormLayout()
        trainFileButton = QPushButton("Open")
        trainFileButton.clicked.connect(lambda: self.data_from_file_ml("Training"))
        testFileButton = QPushButton("Open")
        testFileButton.clicked.connect(lambda: self.data_from_file_ml("Testing"))
        selectButton = QPushButton("Select")
        selectButton.clicked.connect(self.data_from_panel)
        topGroupBoxLayout.addRow("Open training file:", trainFileButton)
        topGroupBoxLayout.addRow("Open testing file:", testFileButton)
        topGroupBoxLayout.addRow("Select data:", selectButton)
        topGroupBox.setLayout(topGroupBoxLayout)

        # tree
        treeGroupBox = QGroupBox("Machine learning algorithms", self)
        treeGroupBox.setFont(QFont("Arial", 10))
        treeLayout = QHBoxLayout()
        self.ml_treeWidget = QTreeWidget()
        self.ml_treeWidget.setColumnCount(2)
        self.ml_treeWidget.setMinimumWidth(300)
        self.ml_treeWidget.setColumnWidth(0, 150)
        self.ml_treeWidget.setFont(QFont("Arial", 8))
        self.ml_treeWidget.setHeaderLabels(["Methods", "Definition"])
        self.machineLearningAlgorighms = QTreeWidgetItem(self.ml_treeWidget)
        self.machineLearningAlgorighms.setExpanded(True)  # set node expanded
        self.machineLearningAlgorighms.setText(0, "Machine learning algorithms")
        self.ml_treeWidget.clicked.connect(self.ml_tree_clicked)
        rf = QTreeWidgetItem(self.machineLearningAlgorighms)
        rf.setText(0, "RF")
        rf.setText(1, "Random Forest")
        dtree = QTreeWidgetItem(self.machineLearningAlgorighms)
        dtree.setText(0, "DecisionTree")
        dtree.setText(1, "Decision Tree")
        lightgbm = QTreeWidgetItem(self.machineLearningAlgorighms)
        lightgbm.setText(0, "LightGBM")
        lightgbm.setText(1, "LightGBM")
        svm = QTreeWidgetItem(self.machineLearningAlgorighms)
        svm.setText(0, "SVM")
        svm.setText(1, "Support Verctor Machine")
        mlp = QTreeWidgetItem(self.machineLearningAlgorighms)
        mlp.setText(0, "MLP")
        mlp.setText(1, "Multi-layer Perceptron")
        xgboost = QTreeWidgetItem(self.machineLearningAlgorighms)
        xgboost.setText(0, "XGBoost")
        xgboost.setText(1, "XGBoost")
        knn = QTreeWidgetItem(self.machineLearningAlgorighms)
        knn.setText(0, "KNN")
        knn.setText(1, "K-Nearest Neighbour")
        lr = QTreeWidgetItem(self.machineLearningAlgorighms)
        lr.setText(0, "LR")
        lr.setText(1, "Logistic Regression")
        lda = QTreeWidgetItem(self.machineLearningAlgorighms)
        lda.setText(0, "LDA")
        lda.setText(1, "Linear Discriminant Analysis")
        qda = QTreeWidgetItem(self.machineLearningAlgorighms)
        qda.setText(0, "QDA")
        qda.setText(1, "Quadratic Discriminant Analysis")
        sgd = QTreeWidgetItem(self.machineLearningAlgorighms)
        sgd.setText(0, "SGD")
        sgd.setText(1, "Stochastic Gradient Descent")
        bayes = QTreeWidgetItem(self.machineLearningAlgorighms)
        bayes.setText(0, "NaiveBayes")
        bayes.setText(1, "NaiveBayes")
        bagging = QTreeWidgetItem(self.machineLearningAlgorighms)
        bagging.setText(0, "Bagging")
        bagging.setText(1, "Bagging")
        adaboost = QTreeWidgetItem(self.machineLearningAlgorighms)
        adaboost.setText(0, "AdaBoost")
        adaboost.setText(1, "AdaBoost")
        gbdt = QTreeWidgetItem(self.machineLearningAlgorighms)
        gbdt.setText(0, "GBDT")
        gbdt.setText(1, "Gradient Tree Boosting")
        # deep learning algorighms
        net1 = QTreeWidgetItem(self.machineLearningAlgorighms)
        net1.setText(0, "Net_1_CNN")
        net1.setText(1, "Convolutional Neural Network")
        net2 = QTreeWidgetItem(self.machineLearningAlgorighms)
        net2.setText(0, "Net_2_RNN")
        net2.setText(1, "Recurrent Neural Network")
        net3 = QTreeWidgetItem(self.machineLearningAlgorighms)
        net3.setText(0, "Net_3_BRNN")
        net3.setText(1, "Bidirectional Recurrent Neural Network")
        net4 = QTreeWidgetItem(self.machineLearningAlgorighms)
        net4.setText(0, "Net_4_ABCNN")
        net4.setText(1, "Attention Based Convolutional Neural Network")
        net5 = QTreeWidgetItem(self.machineLearningAlgorighms)
        net5.setText(0, "Net_5_ResNet")
        net5.setText(1, "Deep Residual Network")
        net6 = QTreeWidgetItem(self.machineLearningAlgorighms)
        net6.setText(0, "Net_6_AE")
        net6.setText(1, "AutoEncoder")

        treeLayout.addWidget(self.ml_treeWidget)
        treeGroupBox.setLayout(treeLayout)

        ## parameter
        paraGroupBox = QGroupBox("Parameters", self)
        paraGroupBox.setMaximumHeight(150)
        paraGroupBox.setFont(QFont("Arial", 10))
        paraLayout = QFormLayout(paraGroupBox)
        self.ml_fold_lineEdit = InputDialog.MyLineEdit("5")
        self.ml_fold_lineEdit.setFont(QFont("Arial", 8))
        self.ml_fold_lineEdit.clicked.connect(self.setFold)
        paraLayout.addRow("Cross-Validation:", self.ml_fold_lineEdit)
        self.ml_algorithm_lineEdit = QLineEdit()
        self.ml_algorithm_lineEdit.setFont(QFont("Arial", 8))
        self.ml_algorithm_lineEdit.setEnabled(False)
        paraLayout.addRow("Algorithm:", self.ml_algorithm_lineEdit)
        self.ml_para_lineEdit = QLineEdit()
        self.ml_para_lineEdit.setFont(QFont("Arial", 8))
        self.ml_para_lineEdit.setEnabled(False)
        paraLayout.addRow("Parameter(s):", self.ml_para_lineEdit)

        ## start button
        startGroupBox = QGroupBox("Operator", self)
        startGroupBox.setFont(QFont("Arial", 10))
        startLayout = QHBoxLayout(startGroupBox)
        self.ml_start_button = QPushButton("Start")
        self.ml_start_button.clicked.connect(self.run_train_model)
        self.ml_start_button.setFont(QFont("Arial", 10))
        self.ml_save_button = QPushButton("Save")
        self.ml_save_button.setFont(QFont("Arial", 10))
        self.ml_save_button.clicked.connect(self.save_ml_files)
        startLayout.addWidget(self.ml_start_button)
        startLayout.addWidget(self.ml_save_button)

        ### layout
        left_vertical_layout = QVBoxLayout()
        left_vertical_layout.addWidget(topGroupBox)
        left_vertical_layout.addWidget(treeGroupBox)
        left_vertical_layout.addWidget(paraGroupBox)
        left_vertical_layout.addWidget(startGroupBox)

        #### widget
        leftWidget = QWidget()
        leftWidget.setLayout(left_vertical_layout)

        #### view region
        scoreTabWidget = QTabWidget()
        trainScoreWidget = QWidget()
        testScoreWidget = QWidget()
        scoreTabWidget.addTab(trainScoreWidget, "Training data score")
        train_score_layout = QVBoxLayout(trainScoreWidget)
        self.train_score_tableWidget = QTableWidget()
        self.train_score_tableWidget.setFont(QFont("Arial", 8))
        self.train_score_tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.train_score_tableWidget.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        train_score_layout.addWidget(self.train_score_tableWidget)
        scoreTabWidget.addTab(testScoreWidget, "Testing data score")
        test_score_layout = QVBoxLayout(testScoreWidget)
        self.test_score_tableWidget = QTableWidget()
        self.test_score_tableWidget.setFont(QFont("Arial", 8))
        self.test_score_tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.test_score_tableWidget.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        test_score_layout.addWidget(self.test_score_tableWidget)

        self.metricsTableWidget = QTableWidget()
        self.metricsTableWidget.setFont(QFont("Arial", 8))
        self.metricsTableWidget.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.metricsTableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.metricsTableWidget.resizeRowsToContents()
        splitter_middle = QSplitter(Qt.Vertical)
        splitter_middle.addWidget(scoreTabWidget)
        splitter_middle.addWidget(self.metricsTableWidget)

        self.dataTableWidget = QTableWidget(6, 4)
        self.dataTableWidget.setFont(QFont("Arial", 8))
        self.dataTableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.dataTableWidget.setShowGrid(False)
        self.dataTableWidget.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.dataTableWidget.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self.dataTableWidget.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeToContents
        )
        self.dataTableWidget.setHorizontalHeaderLabels(
            ["Select", "Data", "Shape", "Source"]
        )
        self.dataTableWidget.verticalHeader().setVisible(False)

        self.roc_curve_widget = PlotWidgets.CurveWidget()
        self.prc_curve_widget = PlotWidgets.CurveWidget()
        plotTabWidget = QTabWidget()
        rocWidget = QWidget()
        self.rocLayout = QVBoxLayout(rocWidget)
        self.rocLayout.addWidget(self.roc_curve_widget)
        prcWidget = QWidget()
        self.prcLayout = QHBoxLayout(prcWidget)
        self.prcLayout.addWidget(self.prc_curve_widget)
        plotTabWidget.addTab(rocWidget, "ROC curve")
        plotTabWidget.addTab(prcWidget, "PRC curve")
        splitter_right = QSplitter(Qt.Vertical)
        splitter_right.addWidget(self.dataTableWidget)
        splitter_right.addWidget(plotTabWidget)
        splitter_right.setSizes([100, 300])

        splitter_view = QSplitter(Qt.Horizontal)
        splitter_view.addWidget(splitter_middle)
        splitter_view.addWidget(splitter_right)
        splitter_view.setSizes([100, 200])

        ##### splitter
        splitter_1 = QSplitter(Qt.Horizontal)
        splitter_1.addWidget(leftWidget)
        splitter_1.addWidget(splitter_view)
        splitter_1.setSizes([100, 1200])

        ###### vertical layout
        vLayout = QVBoxLayout()

        ## status bar
        statusGroupBox = QGroupBox("Status", self)
        statusGroupBox.setFont(QFont("Arial", 10))
        statusLayout = QHBoxLayout(statusGroupBox)
        self.ml_status_label = QLabel("Welcome to SIFS")
        self.ml_progress_bar = QLabel()
        self.ml_progress_bar.setMaximumWidth(230)
        statusLayout.addWidget(self.ml_status_label)
        statusLayout.addWidget(self.ml_progress_bar)

        splitter_2 = QSplitter(Qt.Vertical)
        splitter_2.addWidget(splitter_1)
        splitter_2.addWidget(statusGroupBox)
        splitter_2.setSizes([1000, 100])
        vLayout.addWidget(splitter_2)
        self.tab_machine.setLayout(vLayout)

    """ event in tab_cluster """

    def data_from_file(self):
        self.clust_file, ok = QFileDialog.getOpenFileName(
            self,
            "Open",
            "./data",
            "CSV Files (*.csv);;TSV Files (*.tsv);;SVM Files(*.svm);;Weka Files (*.arff)",
        )
        self.clust_file_lineEdit.setText(self.clust_file)
        if ok:
            self.clust_status_label.setText("Open file " + self.clust_file)
            self.clust_data = DataAnalysis.SIFSData(self.clust_default_para)
            ok1 = self.clust_data.load_data_from_file(self.clust_file)
            if ok1:
                self.clust_label2.setText(
                    "Data shape: (%s, %s)"
                    % (self.clust_data.row, self.clust_data.column)
                )
                self.clust_status_label.setText(
                    "Open file %s successfully." % self.clust_file
                )
                self.clust_data_lineEdit.setText("")
            else:
                self.clust_status_label.setText(
                    "Open file %s failed. Error: %s"
                    % (self.clust_file, self.clust_data.error_msg)
                )

    def data_from_descriptor(self):
        data_source, ok = InputDialog.QDataSelection.getValues(
            selection=self.selection_data
        )
        if ok and data_source == "Feature selection data":
            self.clust_data = DataAnalysis.SIFSData(self.clust_default_para)
            ok = self.clust_data.load_data_from_selection(self.selection_data)
            if ok:
                self.clust_label2.setText(
                    "Data shape: (%s, %s)"
                    % (self.clust_data.row, self.clust_data.column)
                )
                self.clust_data_lineEdit.setText("Data From <Selection> panel")
                self.clust_file_lineEdit.setText("")
        if ok and data_source == "Feature normalization data":
            self.clust_data = DataAnalysis.SIFSData(self.clust_default_para)
            ok = self.clust_data.load_data_from_normalization(self.selection_data)
            if ok:
                self.clust_label2.setText(
                    "Data shape: (%s, %s)"
                    % (self.clust_data.row, self.clust_data.column)
                )
                self.clust_data_lineEdit.setText("Data From <Selection> panel")
                self.clust_file_lineEdit.setText("")

    def clust_tree_clicked(self, index):
        item = (
            self.clust_treeWidget.currentItem()
        )  # item = None if currentItem() is disabled
        if item and item.text(0) not in [
            "Cluster algorithms",
            "Dimensionality reduction algorithms",
        ]:
            self.clust_analysis_type = item.parent().text(0)
            self.clust_analysisType_lineEdit.setText(self.clust_analysis_type)
            if item.text(0) in [
                "kmeans",
                "MiniBatchKMeans",
                "GM",
                "Agglomerative",
                "Spectral",
            ]:
                self.clust_selected_algorithm = item.text(0)
                self.cluster_algorithm_lineEdit.setText(self.clust_selected_algorithm)
                num, ok = QInputDialog.getInt(
                    self,
                    "%s setting" % self.clust_selected_algorithm,
                    "Cluster number",
                    2,
                    2,
                    10,
                    1,
                )
                if ok:
                    self.clust_default_para["nclusters"] = num
                    self.clust_para_lineEdit.setText("Cluster number: %s" % num)
            elif item.text(0) in ["MCL"]:
                self.clust_selected_algorithm = item.text(0)
                self.cluster_algorithm_lineEdit.setText(self.clust_selected_algorithm)
                expand, inflate, mult, ok = InputDialog.QMCLInput.getValues()
                if ok:
                    self.clust_default_para["expand_factor"] = expand
                    self.clust_default_para["inflate_factor"] = inflate
                    self.clust_default_para["multiply_factor"] = mult
                    self.clust_para_lineEdit.setText(
                        "Expand: %s; Inflate: %s; Multiply: %s"
                        % (expand, inflate, mult)
                    )
            elif item.text(0) in ["hcluster", "APC", "meanshift", "DBSCAN"]:
                self.clust_selected_algorithm = item.text(0)
                self.cluster_algorithm_lineEdit.setText(self.clust_selected_algorithm)
                self.clust_para_lineEdit.setText("None")
            elif item.text(0) in ["PCA", "t_SNE", "LDA"]:
                self.clust_selected_algorithm = item.text(0)
                self.cluster_algorithm_lineEdit.setText(self.clust_selected_algorithm)
                num, ok = QInputDialog.getInt(
                    self,
                    "%s setting" % self.clust_selected_algorithm,
                    "Reduced number of dimensions",
                    2,
                    2,
                    10000,
                    1,
                )
                if ok:
                    self.clust_default_para["n_components"] = num
                    self.clust_para_lineEdit.setText(
                        "Reduced number of dimensions: %s" % num
                    )

    def run_data_analysis(self):
        if self.clust_selected_algorithm != "" and not self.clust_data is None:
            self.clust_status = False
            self.clust_progress_bar.setMovie(self.gif)
            self.gif.start()
            t = threading.Thread(target=self.data_analysis)
            t.start()
        else:
            if self.clust_data is None:
                QMessageBox.critical(
                    self,
                    "Error",
                    "Empty data!",
                    QMessageBox.Ok | QMessageBox.No,
                    QMessageBox.Ok,
                )
            else:
                QMessageBox.critical(
                    self,
                    "Error",
                    "Please select an analysis algorithm.",
                    QMessageBox.Ok | QMessageBox.No,
                    QMessageBox.Ok,
                )

    def data_analysis(self):
        if self.clust_selected_algorithm != "" and not self.clust_data is None:
            self.clust_start_button.setDisabled(True)
            self.tab_cluster.setDisabled(True)
            self.setTabEnabled(0, False)
            self.setTabEnabled(2, False)
            self.clust_status_label.setText("Calculating ...")
            if self.clust_analysis_type == "Cluster algorithms":
                cmd = "self.clust_data." + self.clust_selected_algorithm + "()"
                try:
                    status = eval(cmd)
                    self.clust_status = status
                except Exception as e:
                    self.clust_data.error_msg = "Clustering failed."
                    status = False
            else:
                if self.clust_selected_algorithm == "t_SNE":
                    algo = "t_sne"
                else:
                    algo = self.clust_selected_algorithm
                # Note: clust_data.dimension_reduction_result used to show RD data in QTableWidget
                cmd = (
                    "self.clust_data."
                    + algo
                    + '(self.clust_default_para["n_components"])'
                )
                try:
                    self.clust_data.dimension_reduction_result, status = eval(cmd)
                    # When ploting, the RD data used in n_components = 2, because when RD data with more than 2-D,
                    self.clust_data.cluster_plot_data, _ = self.clust_data.t_sne(2)
                    self.clust_status = status
                except Exception as e:
                    self.clust_data.error_msg = str(e)
                    self.clust_status = False
            self.clust_start_button.setDisabled(False)
            self.tab_cluster.setDisabled(False)
            self.setTabEnabled(0, True)
            self.setTabEnabled(2, True)
            self.clust_signal.emit()
            self.clust_progress_bar.clear()
        else:
            if self.clust_data is None:
                QMessageBox.critical(
                    self,
                    "Error",
                    "Empty data!",
                    QMessageBox.Ok | QMessageBox.No,
                    QMessageBox.Ok,
                )
            elif self.clust_selected_algorithm == "":
                QMessageBox.critical(
                    self,
                    "Error",
                    "Please select an analysis algorithm.",
                    QMessageBox.Ok | QMessageBox.No,
                    QMessageBox.Ok,
                )
            else:
                QMessageBox.critical(
                    self,
                    "Error",
                    str(self.clust_data.error_msg),
                    QMessageBox.Ok | QMessageBox.No,
                    QMessageBox.Ok,
                )

    def display_data_analysis(self):
        if self.clust_analysis_type == "Cluster algorithms":
            if self.clust_status:
                self.clust_status_label.setText(
                    "%s calculation complete." % self.clust_selected_algorithm
                )
                self.clust_tableWidget.setColumnCount(2)
                self.clust_tableWidget.setRowCount(self.clust_data.row)
                self.clust_tableWidget.setHorizontalHeaderLabels(
                    ["SampleName", "Cluster"]
                )
                for i in range(self.clust_data.row):
                    cell = QTableWidgetItem(self.clust_data.dataframe.index[i])
                    self.clust_tableWidget.setItem(i, 0, cell)
                    cell1 = QTableWidgetItem(str(self.clust_data.cluster_result[i]))
                    self.clust_tableWidget.setItem(i, 1, cell1)
                """ plot with Matplotlib """
                self.clust_diagram_layout.removeWidget(self.clust_diagram)
                sip.delete(self.clust_diagram)
                plot_data = self.clust_data.generate_plot_data(
                    self.clust_data.cluster_result, self.clust_data.cluster_plot_data
                )
                self.clust_diagram = PlotWidgets.ClusteringDiagramMatplotlib()
                self.clust_diagram.init_data("Clustering", plot_data)
                self.clust_diagram_layout.addWidget(self.clust_diagram)
            else:
                self.clust_status_label.setText(str(self.clust_data.error_msg))
                QMessageBox.critical(
                    self,
                    "Calculate failed",
                    "%s" % self.clust_data.error_msg,
                    QMessageBox.Ok | QMessageBox.No,
                    QMessageBox.Ok,
                )
        else:
            if self.clust_status:
                self.clust_status_label.setText(
                    "%s calculation complete." % self.clust_selected_algorithm
                )
                self.clust_tableWidget.setColumnCount(
                    self.clust_default_para["n_components"] + 1
                )
                self.clust_tableWidget.setRowCount(self.clust_data.row)
                self.clust_tableWidget.setHorizontalHeaderLabels(
                    ["SampleName"]
                    + [
                        "PC%s" % i
                        for i in range(1, self.clust_default_para["n_components"] + 1)
                    ]
                )
                for i in range(self.clust_data.row):
                    cell = QTableWidgetItem(self.clust_data.dataframe.index[i])
                    self.clust_tableWidget.setItem(i, 0, cell)
                    for j in range(self.clust_default_para["n_components"]):
                        cell = QTableWidgetItem(
                            str(self.clust_data.dimension_reduction_result[i][j])
                        )
                        self.clust_tableWidget.setItem(i, j + 1, cell)

                """ plot with Matplotlib """
                self.clust_diagram_layout.removeWidget(self.clust_diagram)
                sip.delete(self.clust_diagram)
                plot_data = self.clust_data.generate_plot_data(
                    self.clust_data.datalabel, self.clust_data.cluster_plot_data
                )
                self.clust_diagram = PlotWidgets.ClusteringDiagramMatplotlib()
                self.clust_diagram.init_data("Dimension reduction", plot_data)
                self.clust_diagram_layout.addWidget(self.clust_diagram)
            else:
                self.clust_status_label.setText(self.clust_data.error_msg)
                QMessageBox.critical(
                    self,
                    "Calculate failed",
                    "%s" % self.clust_data.error_msg,
                    QMessageBox.Ok | QMessageBox.No,
                    QMessageBox.Ok,
                )

    def save_cluster_rd(self):
        try:
            if self.clust_analysis_type != "" and (
                not self.clust_data.cluster_result is None
                or not self.clust_data.dimension_reduction_result is None
            ):
                saved_file, ok = QFileDialog.getSaveFileName(
                    self, "Save", "./data", "TXT Files (*.txt)"
                )
                if ok:
                    self.clust_data.save_data(saved_file, self.clust_analysis_type)

            else:
                QMessageBox.critical(
                    self,
                    "Error",
                    "Empty data!",
                    QMessageBox.Ok | QMessageBox.No,
                    QMessageBox.Ok,
                )
        except Exception as e:
            QMessageBox.critical(
                self, "Error", str(e), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok
            )

    """ event in tab_selection """

    def data_from_file_s(self):
        self.selection_file, ok = QFileDialog.getOpenFileName(
            self,
            "Open",
            "./data",
            "CSV Files (*.csv);;TSV Files (*.tsv);;SVM Files(*.svm);;Weka Files (*.arff)",
        )
        self.selection_file_lineEdit.setText(self.selection_file)
        if ok:
            self.selection_status_label.setText("Open file " + self.selection_file)
            self.selection_data = DataAnalysis.SIFSData(self.selection_default_para)
            ok1 = self.selection_data.load_data_from_file(self.selection_file)
            if ok1:
                self.selection_label2.setText(
                    "Data shape: (%s, %s)"
                    % (self.selection_data.row, self.selection_data.column)
                )
                self.selection_status_label.setText(
                    "Open file %s successfully." % self.selection_file
                )
            else:
                self.selection_status_label.setText(
                    "Open file %s failed. Error: %s"
                    % (self.selection_file, self.selection_data.error_msg)
                )

    def data_from_panel_s(self):
        data_source, ok = InputDialog.QDataSelection.getValues(
            reduction=self.clust_data
        )
        if ok and data_source == "Dimensionality reduction data":
            self.selection_data = DataAnalysis.SIFSData(self.selection_default_para)
            ok = self.selection_data.load_data_from_dimension_reduction(self.clust_data)
            if ok:
                self.selection_label2.setText(
                    "Data shape: (%s, %s)"
                    % (self.selection_data.row, self.selection_data.column)
                )
                self.selection_data_lineEdit.setText(
                    "Data From <Dimensionality reduction> panel"
                )
                self.selection_file_lineEdit.setText("")

    def selection_tree_clicked(self, index):
        item = (
            self.selection_treeWidget.currentItem()
        )  # item = None if currentItem() is disabled
        if item and item.text(0) not in [
            "Feature selection algorithms",
            "Feature Normalization algorithms",
        ]:
            self.selection_analysis_type = item.parent().text(0)
            self.selection_analysisType_lineEdit.setText(self.selection_analysis_type)
            if item.text(0) in ["CHI2", "IG", "FScore", "MIC", "Pearsonr"]:
                self.selection_selected_algorithm = item.text(0)
                self.selection_algorithm_lineEdit.setText(
                    self.selection_selected_algorithm
                )
                num, ok = QInputDialog.getInt(
                    self,
                    "%s setting" % self.selection_selected_algorithm,
                    "Selected feature number",
                    5,
                    1,
                    10000,
                    1,
                )
                if ok:
                    self.selection_default_para["feature_number"] = num
                    self.selection_para_lineEdit.setText(
                        "Selected feature number: %s" % num
                    )
            elif item.text(0) in [
                "ACO",
                "ACS",
                "ALO",
                "AOA",
                "ABC",
                "ABO",
                "ASO",
                "BA",
                "BWO",
                "BDMO",
                "BOA",
                "CSAO",
                "CSA",
                "CS",
                "DE",
                "DOA",
                "DBO",
                "DAOA",
                "EPO",
                "EO",
                "FA",
                "FPA",
                "FOA",
                "GNDO",
                "GA",
                "GAT",
                "GSA",
                "GWO",
                "HS",
                "HHO",
                "HGSO",
                "HLO",
                "HPO",
                "JA",
                "MRFO",
                "MPA",
                "MBO",
                "MFO",
                "MVO",
                "PSO",
                "PFA",
                "PRO",
                "SSA",
                "SCSO",
                "SBO",
                "SA",
                "SCA",
                "SMA",
                "SOS",
                "TGA",
                "TSA",
                "WSA",
                "WOA",
                "WHO",
            ]:
                self.selection_selected_algorithm = item.text(0)
                self.selection_algorithm_lineEdit.setText(
                    self.selection_selected_algorithm
                )

                if self.selection_selected_algorithm == "FPA":
                    self.selection_default_para["P"] = 0.8
                elif (
                    self.selection_selected_algorithm == "DOA"
                    or self.selection_selected_algorithm == "MPA"
                ):
                    self.selection_default_para["P"] = 0.5

                if self.selection_selected_algorithm == "AOA":
                    self.selection_default_para["Mu"] = 0.5
                elif self.selection_selected_algorithm == "DAOA":
                    self.selection_default_para["Mu"] = 0.001

                if self.selection_selected_algorithm == "BA":
                    self.selection_default_para["gamma"] = 0.9
                elif self.selection_selected_algorithm == "FA":
                    self.selection_default_para["gamma"] = 1

                if self.selection_selected_algorithm == "DE":
                    self.selection_default_para["CR"] = 0.9
                elif (
                    self.selection_selected_algorithm == "GA"
                    or self.selection_selected_algorithm == "GAT"
                ):
                    self.selection_default_para["CR"] = 0.8

                if (
                    self.selection_selected_algorithm == "GA"
                    or self.selection_selected_algorithm == "GAT"
                ):
                    self.selection_default_para["MR"] = 0.01
                elif self.selection_selected_algorithm == "SBO":
                    self.selection_default_para["MR"] = 0.05

                if self.selection_selected_algorithm == "BOA":
                    self.selection_default_para["p"] = 0.8
                elif self.selection_selected_algorithm == "MBO":
                    self.selection_default_para["p"] = 0.5
                elif self.selection_selected_algorithm == "MVO":
                    self.selection_default_para["p"] = 6.0

                if (
                    self.selection_selected_algorithm == "ABO"
                    or self.selection_selected_algorithm == "MVO"
                ):
                    self.selection_default_para["type"] = 1

                if self.selection_selected_algorithm == "HGSO":
                    self.selection_default_para["c1"] = 0.1
                elif self.selection_selected_algorithm == "PSO":
                    self.selection_default_para["c1"] = 2.0

                if self.selection_selected_algorithm == "HGSO":
                    self.selection_default_para["c2"] = 0.2
                elif self.selection_selected_algorithm == "PSO":
                    self.selection_default_para["c2"] = 2.0

                if self.selection_selected_algorithm == "SBO":
                    self.selection_default_para["z"] = 0.02
                elif self.selection_selected_algorithm == "SMA":
                    self.selection_default_para["z"] = 0.03

                if self.selection_selected_algorithm == "BOA":
                    self.selection_default_para["c"] = 0.01
                elif self.selection_selected_algorithm == "SA":
                    self.selection_default_para["c"] = 0.93

                if self.selection_selected_algorithm == "ACS":
                    self.selection_default_para["phi"] = 0.5
                elif self.selection_selected_algorithm == "WSA":
                    self.selection_default_para["phi"] = 0.001

                if self.selection_selected_algorithm == "FA":
                    self.selection_default_para["theta"] = 0.97
                elif self.selection_selected_algorithm == "TGA":
                    self.selection_default_para["theta"] = 0.8

                if self.selection_selected_algorithm == "MFO":
                    self.selection_default_para["b"] = 1
                elif self.selection_selected_algorithm == "WOA":
                    self.selection_default_para["b"] = 1

                (
                    N,
                    max_Iter,
                    basic_model,
                    alpha,
                    beta,
                    thres,
                    tau,
                    rho,
                    eta,
                    phi,
                    Alpha,
                    Mu,
                    max_limit,
                    step_e,
                    ratio,
                    type,
                    gamma,
                    A_max,
                    r0_max,
                    nBabysitter,
                    peep,
                    c,
                    p,
                    AP,
                    f1,
                    Pa,
                    CR,
                    F,
                    P,
                    Q,
                    M,
                    f,
                    l,
                    beta0,
                    theta,
                    MR,
                    Tour_size,
                    G0,
                    PAR,
                    HMCR,
                    bw,
                    num_gas,
                    K,
                    L1,
                    L2,
                    L3,
                    c1,
                    c2,
                    pi,
                    pr,
                    B,
                    S,
                    FADs,
                    peri,
                    Smax,
                    BAR,
                    num_land1,
                    b,
                    w,
                    Pmut,
                    z,
                    T0,
                    num_tree1,
                    num_tree2,
                    num_tree4,
                    lambda_,
                    ST,
                    sl,
                    lambda_val,
                    ok,
                ) = InputDialog.QSwarmInput(
                    self.selection_selected_algorithm
                ).getValues(
                    self.selection_selected_algorithm
                )
                if ok:
                    self.selection_default_para["N"] = N
                    self.selection_default_para["max_Iter"] = max_Iter
                    self.selection_default_para["basic_model"] = basic_model
                    self.selection_default_para["alpha"] = alpha
                    self.selection_default_para["beta"] = beta
                    self.selection_default_para["thres"] = thres
                    self.selection_default_para["tau"] = tau
                    self.selection_default_para["rho"] = rho
                    self.selection_default_para["eta"] = eta

                    self.selection_default_para["phi"] = phi
                    self.selection_default_para["Alpha"] = Alpha
                    self.selection_default_para["Mu"] = Mu
                    self.selection_default_para["max_limit"] = max_limit
                    self.selection_default_para["step_e"] = step_e
                    self.selection_default_para["ratio"] = ratio
                    self.selection_default_para["type"] = type
                    self.selection_default_para["gamma"] = gamma
                    self.selection_default_para["A_max"] = A_max
                    self.selection_default_para["r0_max"] = r0_max
                    self.selection_default_para["nBabysitter"] = nBabysitter
                    self.selection_default_para["peep"] = peep
                    self.selection_default_para["c"] = c
                    self.selection_default_para["p"] = p
                    self.selection_default_para["AP"] = AP
                    self.selection_default_para["f1"] = f1
                    self.selection_default_para["Pa"] = Pa
                    self.selection_default_para["CR"] = CR
                    self.selection_default_para["F"] = F
                    self.selection_default_para["P"] = P
                    self.selection_default_para["Q"] = Q
                    self.selection_default_para["M"] = M
                    self.selection_default_para["f"] = f
                    self.selection_default_para["l"] = l
                    self.selection_default_para["beta0"] = beta0
                    self.selection_default_para["theta"] = theta
                    self.selection_default_para["MR"] = MR
                    self.selection_default_para["Tour_size"] = Tour_size
                    self.selection_default_para["G0"] = G0
                    self.selection_default_para["PAR"] = PAR
                    self.selection_default_para["HMCR"] = HMCR
                    self.selection_default_para["bw"] = bw
                    self.selection_default_para["num_gas"] = num_gas
                    self.selection_default_para["K"] = K
                    self.selection_default_para["L1"] = L1
                    self.selection_default_para["L2"] = L2
                    self.selection_default_para["L3"] = L3
                    self.selection_default_para["c1"] = c1
                    self.selection_default_para["c2"] = c2
                    self.selection_default_para["pi"] = pi
                    self.selection_default_para["pr"] = pr
                    self.selection_default_para["B"] = B
                    self.selection_default_para["S"] = S
                    self.selection_default_para["FADs"] = FADs
                    self.selection_default_para["peri"] = peri
                    self.selection_default_para["Smax"] = Smax
                    self.selection_default_para["BAR"] = BAR
                    self.selection_default_para["num_land1"] = num_land1
                    self.selection_default_para["b"] = b
                    self.selection_default_para["w"] = w
                    self.selection_default_para["Pmut"] = Pmut
                    self.selection_default_para["z"] = z
                    self.selection_default_para["T0"] = T0
                    self.selection_default_para["num_tree1"] = num_tree1
                    self.selection_default_para["num_tree2"] = num_tree2
                    self.selection_default_para["num_tree4"] = num_tree4
                    self.selection_default_para["lambda_"] = lambda_
                    self.selection_default_para["ST"] = ST
                    self.selection_default_para["sl"] = sl
                    self.selection_default_para["lambda_val"] = lambda_val

                    self.selection_para_lineEdit.setText(
                        "Selected basic model: %s" % basic_model
                    )
            else:
                self.selection_selected_algorithm = item.text(0)
                self.selection_algorithm_lineEdit.setText(
                    self.selection_selected_algorithm
                )
                self.selection_para_lineEdit.setText("None")

    def run_selection(self):
        if self.selection_selected_algorithm != "" and not self.selection_data is None:
            self.selection_running_status = False
            self.selection_progress_bar.setMovie(self.gif)
            self.gif.start()
            t = threading.Thread(target=self.data_analysis_selTab)
            t.start()
        else:
            if self.selection_data is None:
                QMessageBox.critical(
                    self,
                    "Error",
                    "Empty data!",
                    QMessageBox.Ok | QMessageBox.No,
                    QMessageBox.Ok,
                )
            else:
                self.selection_status_label.setText(
                    (
                        "<font color=red>%s failed. Error: %s </font>"
                        % (
                            self.selection_selected_algorithm,
                            self.selection_data.error_msg,
                        )
                    )
                )

    def data_analysis_selTab(self):
        if self.selection_selected_algorithm != "" and not self.selection_data is None:
            self.selection_start_button.setDisabled(True)
            self.tab_selection.setDisabled(True)
            self.setTabEnabled(1, False)
            self.setTabEnabled(2, False)
            if self.selection_analysis_type == "Feature selection algorithms":
                if self.selection_selected_algorithm in [
                    "CHI2",
                    "IG",
                    "FScore",
                    "MIC",
                    "Pearsonr",
                ]:
                    cmd = (
                        "self.selection_data."
                        + self.selection_selected_algorithm
                        + "()"
                    )
                elif self.selection_selected_algorithm in [
                    "ACO",
                    "ACS",
                    "ALO",
                    "AOA",
                    "ABC",
                    "ABO",
                    "ASO",
                    "BA",
                    "BWO",
                    "BDMO",
                    "BOA",
                    "CSAO",
                    "CSA",
                    "CS",
                    "DE",
                    "DOA",
                    "DBO",
                    "DAOA",
                    "EPO",
                    "EO",
                    "FA",
                    "FPA",
                    "FOA",
                    "GNDO",
                    "GA",
                    "GAT",
                    "GSA",
                    "GWO",
                    "HS",
                    "HHO",
                    "HGSO",
                    "HLO",
                    "HPO",
                    "JA",
                    "MRFO",
                    "MPA",
                    "MBO",
                    "MFO",
                    "MVO",
                    "PSO",
                    "PFA",
                    "PRO",
                    "SSA",
                    "SCSO",
                    "SBO",
                    "SA",
                    "SCA",
                    "SMA",
                    "SOS",
                    "TGA",
                    "TSA",
                    "WSA",
                    "WOA",
                    "WHO",
                ]:
                    cmd = f"self.selection_data.Swarm('{self.selection_selected_algorithm}')"
                status = False
                try:
                    status = eval(cmd)
                    self.selection_signal.emit()
                except Exception as e:
                    QMessageBox.critical(
                        self,
                        "Error",
                        "Calculation failed.",
                        QMessageBox.Ok | QMessageBox.No,
                        QMessageBox.Ok,
                    )
                    self.selection_data.error_msg = str(e)
            else:
                cmd = "self.selection_data." + self.selection_selected_algorithm + "()"
                status = False
                try:
                    status = eval(cmd)
                    self.selection_signal.emit()
                except Exception as e:
                    QMessageBox.critical(
                        self,
                        "Error",
                        "Calculate failed.",
                        QMessageBox.Ok | QMessageBox.No,
                        QMessageBox.Ok,
                    )
                    self.selection_data.error_msg = str(e)
            self.selection_running_status = status
            self.selection_start_button.setDisabled(False)
            self.tab_selection.setDisabled(False)
            self.setTabEnabled(1, True)
            self.setTabEnabled(2, True)
            self.selection_progress_bar.clear()
        else:
            if self.selection_data is None:
                QMessageBox.critical(
                    self,
                    "Error",
                    "Empty data!",
                    QMessageBox.Ok | QMessageBox.No,
                    QMessageBox.Ok,
                )
            else:
                self.selection_status_label.setText(
                    (
                        "<font color=red>%s failed. Error: %s </font>"
                        % (
                            self.selection_selected_algorithm,
                            self.selection_data.error_msg,
                        )
                    )
                )

    def display_selection_data(self):
        if (
            self.selection_analysis_type == "Feature selection algorithms"
            and self.selection_running_status
        ):
            self.selection_status_label.setText(
                "%s calculation complete." % self.selection_selected_algorithm
            )

            self.selection_tableWidget_origin.init_data(
                self.selection_data.feature_origin_data.columns,
                self.selection_data.feature_origin_data.values,
            )
            self.selection_tableWidget.init_data(
                self.selection_data.feature_selection_data.columns,
                self.selection_data.feature_selection_data.values,
            )
            # Draw histogram
            self.selection_histLayout_origin.removeWidget(self.selection_hist_origin)
            sip.delete(self.selection_hist_origin)
            data = self.selection_data.feature_origin_data.values
            self.selection_hist_origin = PlotWidgets.HistogramWidget()
            self.selection_hist_origin.init_data("All data", data)
            self.selection_histLayout_origin.addWidget(self.selection_hist_origin)

            self.selection_histLayout.removeWidget(self.selection_hist)
            sip.delete(self.selection_hist)
            data = self.selection_data.feature_selection_data.values
            self.selection_hist = PlotWidgets.HistogramWidget()
            self.selection_hist.init_data("All data", data)
            self.selection_histLayout.addWidget(self.selection_hist)

            # Draw loss curve
            self.selection_histLayout_loss_curve.removeWidget(
                self.selection_hist_loss_curve
            )
            sip.delete(self.selection_hist_loss_curve)
            self.selection_hist_loss_curve = PlotWidgets.LossWidget()
            self.selection_hist_loss_curve.init_data(
                "Loss Change Over Epochs", self.selection_data.loss_data
            )
            self.selection_histLayout_loss_curve.addWidget(
                self.selection_hist_loss_curve
            )

        if (
            self.selection_analysis_type == "Feature Normalization algorithms"
            and self.selection_running_status
        ):
            self.selection_status_label.setText(
                "%s calculation complete." % self.selection_selected_algorithm
            )
            self.selection_tableWidget_origin.init_data(
                self.selection_data.feature_origin_data.columns,
                self.selection_data.feature_origin_data.values,
            )
            self.selection_tableWidget.init_data(
                self.selection_data.feature_normalization_data.columns,
                self.selection_data.feature_normalization_data.values,
            )
            # Draw histogram
            self.selection_histLayout_origin.removeWidget(self.selection_hist_origin)
            sip.delete(self.selection_hist_origin)
            data = self.selection_data.feature_origin_data.values
            self.selection_hist_origin = PlotWidgets.HistogramWidget()
            self.selection_hist_origin.init_data("All data", data)
            self.selection_histLayout_origin.addWidget(self.selection_hist_origin)

            self.selection_histLayout.removeWidget(self.selection_hist)
            sip.delete(self.selection_hist)
            data = self.selection_data.feature_normalization_data.values
            self.selection_hist = PlotWidgets.HistogramWidget()
            self.selection_hist.init_data("All data", data)
            self.selection_histLayout.addWidget(self.selection_hist)

    def save_selection_normalization_data(self):
        try:
            if not self.selection_data is None:
                saved_file, ok = QFileDialog.getSaveFileName(
                    self,
                    "Save",
                    "./data",
                    "CSV Files (*.csv);;TSV Files (*.tsv);;TSV Files with labels (*.tsv1);;SVM Files(*.svm);;Weka Files (*.arff)",
                )
                if ok:
                    self.selection_data.save_selected_data(
                        saved_file, self.selection_analysis_type
                    )
            else:
                QMessageBox.critical(
                    self,
                    "Error",
                    "Empty data!",
                    QMessageBox.Ok | QMessageBox.No,
                    QMessageBox.Ok,
                )
        except Exception as e:
            QMessageBox.critical(
                self, "Error", str(e), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok
            )

    """ event in tab_machinelearning """

    def ml_panel_clear(self):
        try:
            self.MLData = None
            # self.MLData.training_score = None
            # self.MLData.testing_score = None
            # self.MLData.metrics = None

            self.train_score_tableWidget.clear()
            self.test_score_tableWidget.clear()
            self.metricsTableWidget.clear()
            self.dataTableWidget.clear()
            self.dataTableWidget.setHorizontalHeaderLabels(
                ["Select", "Data", "Shape", "Source"]
            )
            self.current_data_index = 0
        except Exception as e:
            pass

    def data_from_file_ml(self, target="Training"):
        if target == "Training":
            self.ml_panel_clear()
            self.data_index["Training_data"] = None
            self.data_index["Testing_data"] = None
        file_name, ok = QFileDialog.getOpenFileName(
            self,
            "Open",
            "./data",
            "CSV Files (*.csv);;TSV Files (*.tsv);;SVM Files(*.svm);;Weka Files (*.arff)",
        )
        if ok:
            if self.MLData is None:
                self.MLData = MachineLearning.SIFSMachineLearning(self.ml_defatult_para)
            ok1 = self.MLData.load_data(file_name, target)
            if ok1:
                index = 0
                if target == "Training":
                    index = 0
                    self.data_index["Training_data"] = index
                    self.training_data_radio = QRadioButton()
                    self.dataTableWidget.setCellWidget(
                        index, 0, self.training_data_radio
                    )
                    self.current_data_index += 1
                    shape = self.MLData.training_dataframe.values.shape
                else:
                    if self.current_data_index == 1:
                        index = 1
                        self.data_index["Testing_data"] = index
                        self.testing_data_radio = QRadioButton()
                        self.dataTableWidget.setCellWidget(
                            index, 0, self.testing_data_radio
                        )
                        self.current_data_index += 1
                        shape = self.MLData.testing_dataframe.values.shape
                    elif self.current_data_index == 2:
                        index = 1
                        shape = self.MLData.testing_dataframe.values.shape
                self.dataTableWidget.setItem(
                    index, 1, QTableWidgetItem("%s data" % target)
                )
                self.dataTableWidget.setItem(index, 2, QTableWidgetItem(str(shape)))
                self.dataTableWidget.setItem(index, 3, QTableWidgetItem(file_name))
                self.dataTableWidget.resizeRowsToContents()

    def data_from_panel(self):
        if self.current_data_index >= 2:
            self.ml_panel_clear()
            self.data_index["Training_data"] = None
            self.data_index["Testing_data"] = None
        data_source, ok = InputDialog.QDataSelection.getValues(
            selection=self.selection_data, reduction=self.clust_data
        )
        data_df, data_label = None, None
        source = None

        if ok and data_source == "Feature selection data":
            data_sample_index = np.where(
                self.selection_data.data_sample_purpose == True
            )[0]
            if len(data_sample_index) != 0:
                data_df = copy.deepcopy(
                    self.selection_data.feature_selection_data.iloc[
                        data_sample_index, 1:
                    ]
                )
                data_label = copy.deepcopy(
                    self.selection_data.datalabel[data_sample_index]
                )
            source = "feature selection"
        if ok and data_source == "Feature normalization data":
            data_sample_index = np.where(
                self.selection_data.data_sample_purpose == True
            )[0]
            if len(data_sample_index) != 0:
                data_df = copy.deepcopy(
                    self.selection_data.feature_normalization_data.iloc[
                        data_sample_index, 1:
                    ]
                )
                data_label = copy.deepcopy(
                    self.selection_data.datalabel[data_sample_index]
                )
            source = "feature selection"
        if ok and data_source == "Dimensionality reduction data":
            data_sample_index = np.where(self.clust_data.data_sample_purpose == True)[0]
            reduction_datafrme = pd.DataFrame(
                copy.deepcopy(self.clust_data.dimension_reduction_result),
                index=copy.deepcopy(self.clust_data.datalabel),
                columns=[
                    "PC%s" % (i + 1)
                    for i in range(self.clust_data.dimension_reduction_result.shape[1])
                ],
            )
            if len(data_sample_index) != 0:
                data_df = reduction_datafrme.iloc[data_sample_index]
                data_label = self.clust_data.datalabel[data_sample_index].copy()
            source = "Dimensionality reduction data"

        if self.MLData is None:
            self.MLData = MachineLearning.SIFSMachineLearning(self.ml_defatult_para)

        if not data_df is None:
            if self.data_index["Training_data"] is None:
                self.MLData.import_training_data(data_df, data_label)
                index = 0
                self.data_index["Training_data"] = index
                self.current_data_index += 1
                self.dataTableWidget.insertRow(index)
                self.training_data_radio = QRadioButton()
                self.dataTableWidget.setCellWidget(index, 0, self.training_data_radio)
                shape = self.MLData.training_dataframe.values.shape
                self.dataTableWidget.setItem(
                    index, 1, QTableWidgetItem("Training data")
                )
                self.dataTableWidget.setItem(index, 2, QTableWidgetItem(str(shape)))
                self.dataTableWidget.setItem(
                    index, 3, QTableWidgetItem("Data from <%s> panel" % source)
                )
            else:
                self.MLData.import_testing_data(data_df, data_label)
                index = 1
                self.data_index["Testing_data"] = index
                self.current_data_index += 1
                self.dataTableWidget.insertRow(index)
                self.testing_data_radio = QRadioButton()
                self.dataTableWidget.setCellWidget(index, 0, self.testing_data_radio)
                shape = self.MLData.testing_dataframe.values.shape
                self.dataTableWidget.setItem(index, 1, QTableWidgetItem("Testing data"))
                self.dataTableWidget.setItem(index, 2, QTableWidgetItem(str(shape)))
                self.dataTableWidget.setItem(
                    index, 3, QTableWidgetItem("Data from <%s> panel" % source)
                )
            self.dataTableWidget.resizeRowsToContents()

    def setFold(self):
        fold, ok = QInputDialog.getInt(
            self, "Fold number", "Setting K-fold cross-validation", 5, 2, 100, 1
        )
        if ok:
            self.ml_fold_lineEdit.setText(str(fold))
            self.ml_defatult_para["FOLD"] = fold

    def ml_tree_clicked(self, index):
        item = self.ml_treeWidget.currentItem()
        if item.text(0) in ["RF"]:
            self.MLAlgorithm = item.text(0)
            self.ml_algorithm_lineEdit.setText(self.MLAlgorithm)
            num, range, cpu, auto, ok = InputDialog.QRandomForestInput.getValues()
            if ok:
                self.ml_defatult_para["n_trees"] = num
                self.ml_defatult_para["tree_range"] = range
                self.ml_defatult_para["auto"] = auto
                self.ml_defatult_para["cpu"] = cpu
                if auto:
                    self.ml_para_lineEdit.setText("Tree range: %s" % str(range))
                else:
                    self.ml_para_lineEdit.setText("n_trees: %s" % num)
        elif item.text(0) in ["SVM"]:
            self.MLAlgorithm = item.text(0)
            self.ml_algorithm_lineEdit.setText(self.MLAlgorithm)
            kernel, penality, gamma, auto, penalityRange, gammaRange, ok = (
                InputDialog.QSupportVectorMachineInput.getValues()
            )
            if ok:
                self.ml_defatult_para["kernel"] = kernel
                self.ml_defatult_para["penality"] = penality
                self.ml_defatult_para["gamma"] = gamma
                self.ml_defatult_para["auto"] = auto
                self.ml_defatult_para["penalityRange"] = penalityRange
                self.ml_defatult_para["gammaRange"] = gammaRange
                if auto:
                    self.ml_para_lineEdit.setText(
                        "kernel: %s; Auto-Optimization" % kernel
                    )
                else:
                    self.ml_para_lineEdit.setText(
                        "kernel: %s; Penality=%s, Gamma=%s" % (kernel, penality, gamma)
                    )
        elif item.text(0) in ["MLP"]:
            self.MLAlgorithm = item.text(0)
            self.ml_algorithm_lineEdit.setText(self.MLAlgorithm)
            layer, epochs, activation, optimizer, ok = (
                InputDialog.QMultiLayerPerceptronInput.getValues()
            )
            if ok:
                self.ml_defatult_para["layer"] = layer
                self.ml_defatult_para["epochs"] = epochs
                self.ml_defatult_para["activation"] = activation
                self.ml_defatult_para["optimizer"] = optimizer
                self.ml_para_lineEdit.setText(
                    "Layer: %s; Epochs: %s; Activation: %s; Optimizer: %s"
                    % (layer, epochs, activation, optimizer)
                )
        elif item.text(0) in [
            "LR",
            "SGD",
            "DecisionTree",
            "NaiveBayes",
            "AdaBoost",
            "GBDT",
            "LDA",
            "QDA",
        ]:
            self.MLAlgorithm = item.text(0)
            self.ml_algorithm_lineEdit.setText(self.MLAlgorithm)
            self.ml_para_lineEdit.setText("None")
        elif item.text(0) in ["KNN"]:
            self.MLAlgorithm = item.text(0)
            self.ml_algorithm_lineEdit.setText(self.MLAlgorithm)
            topKValue, ok = InputDialog.QKNeighborsInput.getValues()
            if ok:
                self.ml_defatult_para["topKValue"] = topKValue
                self.ml_para_lineEdit.setText("KNN top K value: %s" % topKValue)
        elif item.text(0) in ["LightGBM"]:
            self.MLAlgorithm = item.text(0)
            self.ml_algorithm_lineEdit.setText(self.MLAlgorithm)
            (
                type,
                leaves,
                depth,
                rate,
                leavesRange,
                depthRange,
                rateRange,
                threads,
                auto,
                ok,
            ) = InputDialog.QLightGBMInput.getValues()
            if ok:
                self.ml_defatult_para["boosting_type"] = type
                self.ml_defatult_para["num_leaves"] = leaves
                self.ml_defatult_para["max_depth"] = depth
                self.ml_defatult_para["learning_rate"] = rate
                self.ml_defatult_para["auto"] = auto
                self.ml_defatult_para["leaves_range"] = leavesRange
                self.ml_defatult_para["depth_range"] = depthRange
                self.ml_defatult_para["rate_range"] = rateRange
                self.ml_defatult_para["cpu"] = threads
                if auto:
                    self.ml_para_lineEdit.setText("Parameter auto optimization")
                else:
                    self.ml_para_lineEdit.setText(
                        "Boosting type: %s; Leaves number: %s; Max depth: %s; Learning rate: %s"
                        % (type, leaves, depth, rate)
                    )
        elif item.text(0) in ["XGBoost"]:
            self.MLAlgorithm = item.text(0)
            self.ml_algorithm_lineEdit.setText(self.MLAlgorithm)
            (
                booster,
                maxdepth,
                rate,
                estimator,
                colsample,
                depthRange,
                rateRange,
                threads,
                auto,
                ok,
            ) = InputDialog.QXGBoostInput.getValues()
            self.ml_defatult_para["booster"] = booster
            self.ml_defatult_para["max_depth"] = maxdepth
            self.ml_defatult_para["learning_rate"] = rate
            self.ml_defatult_para["n_estimator"] = estimator
            self.ml_defatult_para["colsample_bytree"] = colsample
            self.ml_defatult_para["depth_range"] = depthRange
            self.ml_defatult_para["rate_range"] = rateRange
            self.ml_defatult_para["cpu"] = threads
            self.ml_defatult_para["auto"] = auto
            if auto:
                self.ml_para_lineEdit.setText("Parameter auto optimization")
            else:
                self.ml_para_lineEdit.setText(
                    "Booster: %s; Maxdepth: %s; Learning rate: %s"
                    % (booster, maxdepth, rate)
                )
        elif item.text(0) in ["Bagging"]:
            self.MLAlgorithm = item.text(0)
            self.ml_algorithm_lineEdit.setText(self.MLAlgorithm)
            n_estimators, threads, ok = InputDialog.QBaggingInput.getValues()
            if ok:
                self.ml_defatult_para["n_estimator"] = n_estimators
                self.ml_defatult_para["cpu"] = threads
                self.ml_para_lineEdit.setText("n_estimators: %s" % n_estimators)
        elif item.text(0) in ["Net_1_CNN"]:
            self.MLAlgorithm = item.text(0)
            self.ml_algorithm_lineEdit.setText(self.MLAlgorithm)
            if not self.MLData is None:
                (
                    input_channel,
                    input_length,
                    output_channel,
                    padding,
                    kernel_size,
                    dropout,
                    learning_rate,
                    epochs,
                    early_stopping,
                    batch_size,
                    fc_size,
                    ok,
                ) = InputDialog.QNetInput_1.getValues(
                    self.MLData.training_dataframe.values.shape[1]
                )
                if ok:
                    self.ml_defatult_para["input_channel"] = input_channel
                    self.ml_defatult_para["input_length"] = input_length
                    self.ml_defatult_para["output_channel"] = output_channel
                    self.ml_defatult_para["padding"] = padding
                    self.ml_defatult_para["kernel_size"] = kernel_size
                    self.ml_defatult_para["dropout"] = dropout
                    self.ml_defatult_para["learning_rate"] = learning_rate
                    self.ml_defatult_para["epochs"] = epochs
                    self.ml_defatult_para["early_stopping"] = early_stopping
                    self.ml_defatult_para["batch_size"] = batch_size
                    self.ml_defatult_para["fc_size"] = fc_size
                    self.ml_para_lineEdit.setText(
                        "Input channel=%s; Input_length=%s; Output_channel=%s; Padding=%s; Kernel_size=%s; Dropout=%s; lr=%s; Epochs=%s; Early_stopping=%s; Batch_size=%s"
                        % (
                            input_channel,
                            input_length,
                            output_channel,
                            padding,
                            kernel_size,
                            dropout,
                            learning_rate,
                            epochs,
                            early_stopping,
                            batch_size,
                        )
                    )
            else:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "Please input training data at first!",
                    QMessageBox.Ok | QMessageBox.No,
                    QMessageBox.Ok,
                )
        elif item.text(0) in ["Net_2_RNN"]:
            self.MLAlgorithm = item.text(0)
            self.ml_algorithm_lineEdit.setText(self.MLAlgorithm)
            if not self.MLData is None:
                (
                    input_channel,
                    input_length,
                    hidden_size,
                    num_layers,
                    dropout,
                    learning_rate,
                    epochs,
                    early_stopping,
                    batch_size,
                    fc_size,
                    ok,
                ) = InputDialog.QNetInput_2.getValues(
                    self.MLData.training_dataframe.values.shape[1]
                )
                if ok:
                    self.ml_defatult_para["input_channel"] = input_channel
                    self.ml_defatult_para["input_length"] = input_length
                    self.ml_defatult_para["rnn_hidden_size"] = hidden_size
                    self.ml_defatult_para["rnn_hidden_layers"] = num_layers
                    self.ml_defatult_para["rnn_bidirection"] = False
                    self.ml_defatult_para["dropout"] = dropout
                    self.ml_defatult_para["learning_rate"] = learning_rate
                    self.ml_defatult_para["epochs"] = epochs
                    self.ml_defatult_para["early_stopping"] = early_stopping
                    self.ml_defatult_para["batch_size"] = batch_size
                    self.ml_defatult_para["rnn_bidirectional"] = False
                    self.ml_para_lineEdit.setText(
                        "Input size=%s; Input_length=%s; Hidden_size=%s; Num_hidden_layers=%s; Dropout=%s; lr=%s; Epochs=%s; Early_stopping=%s; Batch_size=%s"
                        % (
                            input_channel,
                            input_length,
                            hidden_size,
                            num_layers,
                            dropout,
                            learning_rate,
                            epochs,
                            early_stopping,
                            batch_size,
                        )
                    )
            else:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "Please input training data at first!",
                    QMessageBox.Ok | QMessageBox.No,
                    QMessageBox.Ok,
                )
        elif item.text(0) in ["Net_3_BRNN"]:
            self.MLAlgorithm = item.text(0)
            self.ml_algorithm_lineEdit.setText(self.MLAlgorithm)
            if not self.MLData is None:
                (
                    input_channel,
                    input_length,
                    hidden_size,
                    num_layers,
                    dropout,
                    learning_rate,
                    epochs,
                    early_stopping,
                    batch_size,
                    fc_size,
                    ok,
                ) = InputDialog.QNetInput_2.getValues(
                    self.MLData.training_dataframe.values.shape[1]
                )
                if ok:
                    self.ml_defatult_para["input_channel"] = input_channel
                    self.ml_defatult_para["input_length"] = input_length
                    self.ml_defatult_para["rnn_hidden_size"] = hidden_size
                    self.ml_defatult_para["rnn_hidden_layers"] = num_layers
                    self.ml_defatult_para["rnn_bidirection"] = False
                    self.ml_defatult_para["dropout"] = dropout
                    self.ml_defatult_para["learning_rate"] = learning_rate
                    self.ml_defatult_para["epochs"] = epochs
                    self.ml_defatult_para["early_stopping"] = early_stopping
                    self.ml_defatult_para["batch_size"] = batch_size
                    self.ml_defatult_para["rnn_bidirectional"] = True
                    self.ml_para_lineEdit.setText(
                        "Input size=%s; Input_length=%s; Hidden_size=%s; Num_hidden_layers=%s; Dropout=%s; lr=%s; Epochs=%s; Early_stopping=%s; Batch_size=%s"
                        % (
                            input_channel,
                            input_length,
                            hidden_size,
                            num_layers,
                            dropout,
                            learning_rate,
                            epochs,
                            early_stopping,
                            batch_size,
                        )
                    )
            else:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "Please input training data at first!",
                    QMessageBox.Ok | QMessageBox.No,
                    QMessageBox.Ok,
                )
        elif item.text(0) in ["Net_4_ABCNN"]:
            self.MLAlgorithm = item.text(0)
            self.ml_algorithm_lineEdit.setText(self.MLAlgorithm)
            if not self.MLData is None:
                (
                    input_channel,
                    input_length,
                    dropout,
                    learning_rate,
                    epochs,
                    early_stopping,
                    batch_size,
                    ok,
                ) = InputDialog.QNetInput_4.getValues(
                    self.MLData.training_dataframe.values.shape[1]
                )
                if ok:
                    self.ml_defatult_para["input_channel"] = input_channel
                    self.ml_defatult_para["input_length"] = input_length
                    self.ml_defatult_para["dropout"] = dropout
                    self.ml_defatult_para["learning_rate"] = learning_rate
                    self.ml_defatult_para["epochs"] = epochs
                    self.ml_defatult_para["early_stopping"] = early_stopping
                    self.ml_defatult_para["batch_size"] = batch_size
                    self.ml_para_lineEdit.setText(
                        "Input size=%s; Input_length=%s; Dropout=%s; lr=%s; Epochs=%s; Early_stopping=%s; Batch_size=%s"
                        % (
                            input_channel,
                            input_length,
                            dropout,
                            learning_rate,
                            epochs,
                            early_stopping,
                            batch_size,
                        )
                    )
            else:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "Please input training data at first!",
                    QMessageBox.Ok | QMessageBox.No,
                    QMessageBox.Ok,
                )
        elif item.text(0) in ["Net_5_ResNet"]:
            self.MLAlgorithm = item.text(0)
            self.ml_algorithm_lineEdit.setText(self.MLAlgorithm)
            if not self.MLData is None:
                (
                    input_channel,
                    input_length,
                    learning_rate,
                    epochs,
                    early_stopping,
                    batch_size,
                    ok,
                ) = InputDialog.QNetInput_5.getValues(
                    self.MLData.training_dataframe.values.shape[1]
                )
                if ok:
                    self.ml_defatult_para["input_channel"] = input_channel
                    self.ml_defatult_para["input_length"] = input_length
                    self.ml_defatult_para["learning_rate"] = learning_rate
                    self.ml_defatult_para["epochs"] = epochs
                    self.ml_defatult_para["early_stopping"] = early_stopping
                    self.ml_defatult_para["batch_size"] = batch_size
                    self.ml_para_lineEdit.setText(
                        "Input size=%s; Input_length=%s; lr=%s; Epochs=%s; Early_stopping=%s; Batch_size=%s"
                        % (
                            input_channel,
                            input_length,
                            learning_rate,
                            epochs,
                            early_stopping,
                            batch_size,
                        )
                    )
            else:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "Please input training data at first!",
                    QMessageBox.Ok | QMessageBox.No,
                    QMessageBox.Ok,
                )
        elif item.text(0) in ["Net_6_AE"]:
            self.MLAlgorithm = item.text(0)
            self.ml_algorithm_lineEdit.setText(self.MLAlgorithm)
            if not self.MLData is None:
                (
                    input_dim,
                    dropout,
                    learning_rate,
                    epochs,
                    early_stopping,
                    batch_size,
                    ok,
                ) = InputDialog.QNetInput_6.getValues(
                    self.MLData.training_dataframe.values.shape[1]
                )
                if ok:
                    self.ml_defatult_para["mlp_input_dim"] = input_dim
                    self.ml_defatult_para["dropout"] = dropout
                    self.ml_defatult_para["learning_rate"] = learning_rate
                    self.ml_defatult_para["epochs"] = epochs
                    self.ml_defatult_para["early_stopping"] = early_stopping
                    self.ml_defatult_para["batch_size"] = batch_size
                    self.ml_para_lineEdit.setText(
                        "Input dimension=%s; Dropout=%s; lr=%s; Epochs=%s; Early_stopping=%s; Batch_size=%s"
                        % (
                            input_dim,
                            dropout,
                            learning_rate,
                            epochs,
                            early_stopping,
                            batch_size,
                        )
                    )
            else:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "Please input training data at first!",
                    QMessageBox.Ok | QMessageBox.No,
                    QMessageBox.Ok,
                )

    def run_train_model(self):
        if not self.MLAlgorithm is None and not self.MLData is None:
            self.ml_running_status = False
            self.ml_progress_bar.setMovie(self.gif)
            self.gif.start()
            t = threading.Thread(target=self.train_model)
            t.start()
        else:
            QMessageBox.critical(
                self,
                "Error",
                "Please load data or specify an algorithm.",
                QMessageBox.Ok | QMessageBox.No,
                QMessageBox.Ok,
            )

    def train_model(self):
        try:
            if not self.MLAlgorithm is None and not self.MLData is None:
                self.ml_status_label.setText("Training model ... ")
                self.tab_machine.setDisabled(True)
                self.setTabEnabled(0, False)
                self.setTabEnabled(1, False)
                if self.MLAlgorithm == "RF":
                    ok = self.MLData.RandomForest()
                elif self.MLAlgorithm == "SVM":
                    ok = self.MLData.SupportVectorMachine()
                elif self.MLAlgorithm == "MLP":
                    ok = self.MLData.MultiLayerPerceptron()
                elif self.MLAlgorithm == "LR":
                    ok = self.MLData.LogisticRegressionClassifier()
                elif self.MLAlgorithm == "LDA":
                    ok = self.MLData.LDAClassifier()
                elif self.MLAlgorithm == "QDA":
                    ok = self.MLData.QDAClassifier()
                elif self.MLAlgorithm == "KNN":
                    ok = self.MLData.KNeighbors()
                elif self.MLAlgorithm == "LightGBM":
                    ok = self.MLData.LightGBMClassifier()
                elif self.MLAlgorithm == "XGBoost":
                    ok = self.MLData.XGBoostClassifier()
                elif self.MLAlgorithm == "SGD":
                    ok = self.MLData.StochasticGradientDescentClassifier()
                elif self.MLAlgorithm == "DecisionTree":
                    ok = self.MLData.DecisionTree()
                elif self.MLAlgorithm == "NaiveBayes":
                    ok = self.MLData.GaussianNBClassifier()
                elif self.MLAlgorithm == "AdaBoost":
                    ok = self.MLData.AdaBoost()
                elif self.MLAlgorithm == "Bagging":
                    ok = self.MLData.Bagging()
                elif self.MLAlgorithm == "GBDT":
                    ok = self.MLData.GBDTClassifier()
                elif self.MLAlgorithm == "Net_1_CNN":
                    ok = self.MLData.run_networks(1)
                elif self.MLAlgorithm == "Net_2_RNN":
                    ok = self.MLData.run_networks(2)
                elif self.MLAlgorithm == "Net_3_BRNN":
                    ok = self.MLData.run_networks(3)
                elif self.MLAlgorithm == "Net_4_ABCNN":
                    ok = self.MLData.run_networks(4)
                elif self.MLAlgorithm == "Net_5_ResNet":
                    ok = self.MLData.run_networks(5)
                elif self.MLAlgorithm == "Net_6_AE":
                    ok = self.MLData.run_networks(6)

                self.ml_running_status = ok
                self.ml_status_label.setText("Training model complete.")
                self.ml_signal.emit()
                self.tab_machine.setDisabled(False)
                self.setTabEnabled(0, True)
                self.setTabEnabled(1, True)
            else:
                QMessageBox.critical(
                    self,
                    "Error",
                    "Please load data or specify an algorithm.",
                    QMessageBox.Ok | QMessageBox.No,
                    QMessageBox.Ok,
                )
        except Exception as e:
            QMessageBox.critical(
                self, "Error", str(e), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok
            )

    def display_ml_data(self):
        if self.ml_running_status:
            if not self.MLData.message is None:
                self.ml_status_label.setText(self.MLData.message)
            # display predicton score
            if not self.MLData.training_score is None:
                data = self.MLData.training_score.values
                self.train_score_tableWidget.setRowCount(data.shape[0])
                self.train_score_tableWidget.setColumnCount(data.shape[1])
                self.train_score_tableWidget.setHorizontalHeaderLabels(
                    self.MLData.training_score.columns
                )
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        cell = QTableWidgetItem(str(round(data[i][j], 4)))
                        self.train_score_tableWidget.setItem(i, j, cell)
                if self.data_index["Training_score"] is None:
                    # index = self.current_data_index
                    index = 2
                    self.data_index["Training_score"] = index
                    self.dataTableWidget.insertRow(index)
                    self.current_data_index += 1
                else:
                    # index = self.data_index['Training_score']
                    index = 2
                self.training_score_radio = QRadioButton()
                self.dataTableWidget.setCellWidget(index, 0, self.training_score_radio)
                self.dataTableWidget.setItem(
                    index, 1, QTableWidgetItem("Training score")
                )
                self.dataTableWidget.setItem(
                    index, 2, QTableWidgetItem(str(data.shape))
                )
                self.dataTableWidget.setItem(
                    index, 3, QTableWidgetItem("%s model" % self.MLAlgorithm)
                )
            if not self.MLData.testing_score is None:
                data = self.MLData.testing_score.values
                self.test_score_tableWidget.setRowCount(data.shape[0])
                self.test_score_tableWidget.setColumnCount(data.shape[1])
                self.test_score_tableWidget.setHorizontalHeaderLabels(
                    self.MLData.training_score.columns
                )
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        if j == 0:
                            cellData = data[i][j]
                        else:
                            cellData = str(round(data[i][j], 4))
                        cell = QTableWidgetItem(cellData)
                        self.test_score_tableWidget.setItem(i, j, cell)
                if self.data_index["Testing_score"] is None:
                    # index = self.current_data_index
                    index = 3
                    self.data_index["Testing_score"] = index
                    self.dataTableWidget.insertRow(index)
                    self.current_data_index += 1
                else:
                    # index = self.data_index['Testing_score']
                    index = 3
                self.testing_score_radio = QRadioButton()
                self.dataTableWidget.setCellWidget(index, 0, self.testing_score_radio)
                self.dataTableWidget.setItem(
                    index, 1, QTableWidgetItem("Testing score")
                )
                self.dataTableWidget.setItem(
                    index, 2, QTableWidgetItem(str(data.shape))
                )
                self.dataTableWidget.setItem(
                    index, 3, QTableWidgetItem("%s model" % self.MLAlgorithm)
                )

            # display evaluation metrics
            data = self.MLData.metrics.values
            self.metricsTableWidget.setRowCount(data.shape[0])
            self.metricsTableWidget.setColumnCount(data.shape[1])
            self.metricsTableWidget.setHorizontalHeaderLabels(
                [
                    "Sn (%)",
                    "Sp (%)",
                    "Pre (%)",
                    "Acc (%)",
                    "MCC",
                    "F1",
                    "AUROC",
                    "AUPRC",
                ]
            )
            self.metricsTableWidget.setVerticalHeaderLabels(self.MLData.metrics.index)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    cell = QTableWidgetItem(str(data[i][j]))
                    self.metricsTableWidget.setItem(i, j, cell)
            if self.data_index["Metrics"] is None:
                # index = self.current_data_index
                index = 4
                self.data_index["Metrics"] = index
                self.dataTableWidget.insertRow(index)
                self.current_data_index += 1
            else:
                # index = self.data_index['Metrics']
                index = 4
            self.metrics_radio = QRadioButton()
            self.dataTableWidget.setCellWidget(index, 0, self.metrics_radio)
            self.dataTableWidget.setItem(
                index, 1, QTableWidgetItem("Evaluation metrics")
            )
            self.dataTableWidget.setItem(index, 2, QTableWidgetItem(str(data.shape)))
            self.dataTableWidget.setItem(
                index, 3, QTableWidgetItem("%s model" % self.MLAlgorithm)
            )

            # display model
            if self.data_index["Model"] is None:
                # index = self.current_data_index
                index = 5
                self.data_index["Model"] = index
                self.dataTableWidget.insertRow(index)
                self.current_data_index += 1
            else:
                # index = self.data_index['Model']
                index = 5
            self.model_radio = QRadioButton()
            self.dataTableWidget.setCellWidget(index, 0, self.model_radio)
            self.dataTableWidget.setItem(index, 1, QTableWidgetItem("Models"))
            self.dataTableWidget.setItem(index, 2, QTableWidgetItem(str(self.fold_num)))
            self.dataTableWidget.setItem(
                index, 3, QTableWidgetItem("%s model" % self.MLAlgorithm)
            )

            # plot ROC
            try:
                # Draw ROC curve
                if not self.MLData.aucData is None:
                    self.rocLayout.removeWidget(self.roc_curve_widget)
                    sip.delete(self.roc_curve_widget)
                    self.roc_curve_widget = PlotWidgets.CurveWidget()
                    self.roc_curve_widget.init_data(
                        0,
                        "ROC curve",
                        self.MLData.aucData,
                        self.MLData.meanAucData,
                        self.MLData.indepAucData,
                    )
                    self.rocLayout.addWidget(self.roc_curve_widget)
                # plot PRC
                if not self.MLData.prcData is None:
                    self.prcLayout.removeWidget(self.prc_curve_widget)
                    sip.delete(self.prc_curve_widget)
                    self.prc_curve_widget = PlotWidgets.CurveWidget()
                    self.prc_curve_widget.init_data(
                        1,
                        "PRC curve",
                        self.MLData.prcData,
                        self.MLData.meanPrcData,
                        self.MLData.indepPrcData,
                    )
                    self.prcLayout.addWidget(self.prc_curve_widget)
            except Exception as e:
                self.ml_status_label.setText(str(e))
        else:
            QMessageBox.critical(
                self,
                "Error",
                str(self.MLData.error_msg),
                QMessageBox.Ok | QMessageBox.No,
                QMessageBox.Ok,
            )
        self.ml_progress_bar.clear()

    def save_ml_files_orig_with_bugs(self):
        if "training_data_radio" in dir(self) and self.training_data_radio.isChecked():
            saved_file, ok = QFileDialog.getSaveFileName(
                self,
                "Save",
                "./data",
                "CSV Files (*.csv);;TSV Files (*.tsv);;SVM Files(*.svm);;Weka Files (*.arff)",
            )
            if ok:
                ok1 = self.MLData.save_coder(saved_file, "training")
                if not ok1:
                    QMessageBox.critical(
                        self,
                        "Error",
                        str(self.MLData.error_msg),
                        QMessageBox.Ok | QMessageBox.No,
                        QMessageBox.Ok,
                    )
        elif "testing_data_radio" in dir(self) and self.testing_data_radio.isChecked():
            saved_file, ok = QFileDialog.getSaveFileName(
                self,
                "Save",
                "./data",
                "CSV Files (*.csv);;TSV Files (*.tsv);;SVM Files(*.svm);;Weka Files (*.arff)",
            )
            if ok:
                ok1 = self.MLData.save_coder(saved_file, "testing")
                if not ok1:
                    QMessageBox.critical(
                        self,
                        "Error",
                        str(self.MLData.error_msg),
                        QMessageBox.Ok | QMessageBox.No,
                        QMessageBox.Ok,
                    )
        elif (
            "training_score_radio" in dir(self)
            and self.training_score_radio.isChecked()
        ):
            save_file, ok = QFileDialog.getSaveFileName(
                self, "Save", "./data", "TSV Files (*.tsv)"
            )
            if ok:
                ok1 = self.MLData.save_prediction_score(save_file, "training")
                if not ok1:
                    QMessageBox.critical(
                        self,
                        "Error",
                        str(self.MLData.error_msg),
                        QMessageBox.Ok | QMessageBox.No,
                        QMessageBox.Ok,
                    )
        elif (
            "testing_score_radio" in dir(self) and self.testing_score_radio.isChecked()
        ):
            save_file, ok = QFileDialog.getSaveFileName(
                self, "Save", "./data", "TSV Files (*.tsv)"
            )
            if ok:
                ok1 = self.MLData.save_prediction_score(save_file, "testing")
                if not ok1:
                    QMessageBox.critical(
                        self,
                        "Error",
                        str(self.MLData.error_msg),
                        QMessageBox.Ok | QMessageBox.No,
                        QMessageBox.Ok,
                    )
        elif "metrics_radio" in dir(self) and self.metrics_radio.isChecked():
            save_file, ok = QFileDialog.getSaveFileName(
                self, "Save", "./data", "TSV Files (*.tsv)"
            )
            if ok:
                ok1 = self.MLData.save_metrics(save_file)
                if not ok1:
                    QMessageBox.critical(
                        self,
                        "Error",
                        str(self.MLData.error_msg),
                        QMessageBox.Ok | QMessageBox.No,
                        QMessageBox.Ok,
                    )
        elif "model_radio" in dir(self) and self.model_radio.isChecked():
            save_directory = QFileDialog.getExistingDirectory(self, "Save", "./data")
            if self.MLData.best_model is not None:
                for i, model in enumerate(self.MLData.best_model):
                    model_name = "%s/%s_model_%s.pkl" % (
                        save_directory,
                        self.MLData.algorithm,
                        i + 1,
                    )
                    if self.MLData.algorithm in [
                        "RF",
                        "SVM",
                        "MLP",
                        "LR",
                        "KNN",
                        "LightGBM",
                        "XGBoost",
                        "SGD",
                        "DecisionTree",
                        "Bayes",
                        "AdaBoost",
                        "Bagging",
                        "GBDT",
                        "LDA",
                        "QDA",
                    ]:
                        joblib.dump(model, model_name)
                    else:
                        torch.save(model, model_name)
                QMessageBox.information(
                    self,
                    "Model saved",
                    "The models have been saved to directory %s" % save_directory,
                    QMessageBox.Ok | QMessageBox.No,
                    QMessageBox.Ok,
                )
        else:
            QMessageBox.critical(
                self,
                "Error",
                "Please select which data to save.",
                QMessageBox.Ok | QMessageBox.No,
                QMessageBox.Ok,
            )

    def save_ml_files(self):
        tag = 0
        try:
            if self.training_data_radio.isChecked():
                tag = 1
                saved_file, ok = QFileDialog.getSaveFileName(
                    self,
                    "Save",
                    "./data",
                    "CSV Files (*.csv);;TSV Files (*.tsv);;SVM Files(*.svm);;Weka Files (*.arff)",
                )
                if ok:
                    ok1 = self.MLData.save_coder(saved_file, "training")
                    if not ok1:
                        QMessageBox.critical(
                            self,
                            "Error",
                            str(self.MLData.error_msg),
                            QMessageBox.Ok | QMessageBox.No,
                            QMessageBox.Ok,
                        )
        except Exception as e:
            pass

        try:
            if self.testing_data_radio.isChecked():
                tag = 1
                saved_file, ok = QFileDialog.getSaveFileName(
                    self,
                    "Save",
                    "./data",
                    "CSV Files (*.csv);;TSV Files (*.tsv);;SVM Files(*.svm);;Weka Files (*.arff)",
                )
                if ok:
                    ok1 = self.MLData.save_coder(saved_file, "testing")
                    if not ok1:
                        QMessageBox.critical(
                            self,
                            "Error",
                            str(self.MLData.error_msg),
                            QMessageBox.Ok | QMessageBox.No,
                            QMessageBox.Ok,
                        )
        except Exception as e:
            pass

        try:
            if self.training_score_radio.isChecked():
                tag = 1
                save_file, ok = QFileDialog.getSaveFileName(
                    self, "Save", "./data", "TSV Files (*.tsv)"
                )
                if ok:
                    ok1 = self.MLData.save_prediction_score(save_file, "training")
                    if not ok1:
                        QMessageBox.critical(
                            self,
                            "Error",
                            str(self.MLData.error_msg),
                            QMessageBox.Ok | QMessageBox.No,
                            QMessageBox.Ok,
                        )
        except Exception as e:
            pass

        try:
            if self.testing_score_radio.isChecked():
                tag = 1
                save_file, ok = QFileDialog.getSaveFileName(
                    self, "Save", "./data", "TSV Files (*.tsv)"
                )
                if ok:
                    ok1 = self.MLData.save_prediction_score(save_file, "testing")
                    if not ok1:
                        QMessageBox.critical(
                            self,
                            "Error",
                            str(self.MLData.error_msg),
                            QMessageBox.Ok | QMessageBox.No,
                            QMessageBox.Ok,
                        )
        except Exception as e:
            pass

        try:
            if self.metrics_radio.isChecked():
                tag = 1
                save_file, ok = QFileDialog.getSaveFileName(
                    self, "Save", "./data", "TSV Files (*.tsv)"
                )
                if ok:
                    ok1 = self.MLData.save_metrics(save_file)
                    if not ok1:
                        QMessageBox.critical(
                            self,
                            "Error",
                            str(self.MLData.error_msg),
                            QMessageBox.Ok | QMessageBox.No,
                            QMessageBox.Ok,
                        )
        except Exception as e:
            pass

        try:
            if self.model_radio.isChecked():
                tag = 1
                save_directory = QFileDialog.getExistingDirectory(
                    self, "Save", "./models"
                )
                if os.path.exists(save_directory):
                    if self.MLData.best_model is not None:
                        for i, model in enumerate(self.MLData.best_model):
                            model_name = "%s/%s_model_%s.pkl" % (
                                save_directory,
                                self.MLData.algorithm,
                                i + 1,
                            )
                            if self.MLData.algorithm in [
                                "RF",
                                "SVM",
                                "MLP",
                                "LR",
                                "KNN",
                                "LightGBM",
                                "XGBoost",
                                "SGD",
                                "DecisionTree",
                                "Bayes",
                                "AdaBoost",
                                "Bagging",
                                "GBDT",
                                "LDA",
                                "QDA",
                            ]:
                                joblib.dump(model, model_name)
                            else:
                                torch.save(model, model_name)
                        QMessageBox.information(
                            self,
                            "Model saved",
                            "The models have been saved to directory %s"
                            % save_directory,
                            QMessageBox.Ok | QMessageBox.No,
                            QMessageBox.Ok,
                        )
                    else:
                        pass
        except Exception as e:
            pass

        if tag == 0:
            QMessageBox.critical(
                self,
                "Error",
                "Please select which data to save.",
                QMessageBox.Ok | QMessageBox.No,
                QMessageBox.Ok,
            )

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self,
            "Confirm Exit",
            "Are you sure want to quit SIFS?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.close_signal.emit("Basic")
            self.close()
        else:
            if event:
                event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SIFS()
    app.setFont(QFont("Arial", 10))
    window.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window.show()
    sys.exit(app.exec_())
