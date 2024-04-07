# A GUI class for all dialog boxes used for parameter settings.

#!/usr/bin/env python
# _*_ coding: utf-8 _*_
import sys
from PyQt5.QtWidgets import (
    QDialog,
    QFormLayout,
    QLineEdit,
    QDialogButtonBox,
    QInputDialog,
    QMessageBox,
    QComboBox,
    QVBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QTableWidgetSelectionRange,
    QAbstractItemView,
    QGridLayout,
    QTreeWidget,
    QTreeWidgetItem,
    QCheckBox,
    QApplication,
    QFileDialog,
    QColorDialog,
    QPushButton,
)
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import pyqtSignal, Qt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import numpy as np
import pandas as pd
import qdarkstyle
from multiprocessing import cpu_count

class MyLineEdit(QLineEdit):
    clicked = pyqtSignal()

    def mouseReleaseEvent(self, QMouseEvent):
        if QMouseEvent.button() == Qt.LeftButton:
            self.clicked.emit()

class QMCLInput(QDialog):
    def __init__(self):
        super(QMCLInput, self).__init__()
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QIcon("images/logo.ico"))
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowTitle("Markov cluster setting")
        self.setFont(QFont("Arial", 10))
        layout = QFormLayout(self)
        self.expand_factor_lineEdit = MyLineEdit("2")
        self.expand_factor_lineEdit.clicked.connect(self.setExpand)
        self.inflate_factor_lineEdit = MyLineEdit("2.0")
        self.inflate_factor_lineEdit.clicked.connect(self.setInflate)
        self.multiply_factor_lineEdit = MyLineEdit("2.0")
        self.multiply_factor_lineEdit.clicked.connect(self.setMultiply)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow("Expand factor:", self.expand_factor_lineEdit)
        layout.addRow("Inflate factor (float):", self.inflate_factor_lineEdit)
        layout.addRow("Multiply factor (float)", self.multiply_factor_lineEdit)
        layout.addWidget(self.buttons)

    def setExpand(self):
        num, ok = QInputDialog.getInt(
            self, "Expand factor", "Get expand factor", 2, 1, 10, 1
        )
        if ok:
            self.expand_factor_lineEdit.setText(str(num))

    def getExpand(self):
        return (
            int(self.expand_factor_lineEdit.text())
            if self.expand_factor_lineEdit.text() != ""
            else 2
        )

    def setInflate(self):
        num, ok = QInputDialog.getDouble(
            self, "Inflate factor", "Get inflate factor", 2.0, 1.0, 6.0
        )
        if ok:
            self.inflate_factor_lineEdit.setText(str(num))

    def getInflate(self):
        return (
            float(self.inflate_factor_lineEdit.text())
            if self.inflate_factor_lineEdit.text() != ""
            else 2.0
        )

    def setMultiply(self):
        num, ok = QInputDialog.getDouble(
            self, "Inflate factor", "Get inflate factor", 2.0, 1.0, 6.0
        )
        if ok:
            self.multiply_factor_lineEdit.setText(str(num))

    def getMultiply(self):
        return (
            float(self.multiply_factor_lineEdit.text())
            if self.multiply_factor_lineEdit.text() != ""
            else 2.0
        )

    @staticmethod
    def getValues():
        dialog = QMCLInput()
        result = dialog.exec_()
        expand = dialog.getExpand()
        inflate = dialog.getInflate()
        multiply = dialog.getMultiply()
        return expand, inflate, multiply, result == QDialog.Accepted


class QDataSelection(QDialog):
    def __init__(
        self, descriptor=None, selection=None, machinelearning=None, reduction=None
    ):
        super(QDataSelection, self).__init__()
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QIcon("images/logo.ico"))
        self.descriptor = descriptor
        self.reduction = reduction
        self.selection = selection
        self.machinelearning = machinelearning
        self.data_source = None
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowTitle("Data selection")
        self.setFont(QFont("Arial", 8))
        self.resize(430, 260)

        layout = QVBoxLayout(self)
        self.dataTreeWidget = QTreeWidget()
        self.dataTreeWidget.setFont(QFont("Arial", 8))
        self.dataTreeWidget.setColumnCount(2)
        self.dataTreeWidget.setColumnWidth(0, 300)
        self.dataTreeWidget.setColumnWidth(1, 100)
        self.dataTreeWidget.clicked.connect(self.treeClicked)
        self.dataTreeWidget.setHeaderLabels(["Source", "Shape"])
        if (
            not self.descriptor is None
            and "encoding_array" in dir(self.descriptor)
            and len(self.descriptor.encoding_array) > 0
        ):
            descriptor = QTreeWidgetItem(self.dataTreeWidget)
            descriptor.setText(0, "Descriptor data")
            descriptor.setText(
                1, "(%s, %s)" % (self.descriptor.row, self.descriptor.column - 2)
            )
        if (
            not self.reduction is None
            and not self.reduction.dimension_reduction_result is None
        ):
            reduction = QTreeWidgetItem(self.dataTreeWidget)
            reduction.setText(0, "Dimensionality reduction data")
            shape = self.reduction.dimension_reduction_result.shape
            reduction.setText(1, "(%s, %s)" % (shape[0], shape[1]))
        if (
            not self.selection is None
            and not self.selection.feature_selection_data is None
        ):
            selection = QTreeWidgetItem(self.dataTreeWidget)
            selection.setText(0, "Feature selection data")
            shape = self.selection.feature_selection_data.values.shape
            selection.setText(1, "(%s, %s)" % (shape[0], shape[1] - 1))
        if (
            not self.selection is None
            and not self.selection.feature_normalization_data is None
        ):
            normalization = QTreeWidgetItem(self.dataTreeWidget)
            normalization.setText(0, "Feature normalization data")
            shape = self.selection.feature_normalization_data.values.shape
            normalization.setText(1, "(%s, %s)" % (shape[0], shape[1] - 1))
        if (
            not self.machinelearning is None
            and not self.machinelearning.training_dataframe is None
        ):
            ml_training_data = QTreeWidgetItem(self.dataTreeWidget)
            ml_training_data.setText(0, "Machine learning training data")
            shape = self.machinelearning.training_dataframe.values.shape
            ml_training_data.setText(1, str(shape))
        if (
            not self.machinelearning is None
            and not self.machinelearning.testing_dataframe is None
        ):
            ml_testing_data = QTreeWidgetItem(self.dataTreeWidget)
            ml_testing_data.setText(0, "Machine learning testing data")
            shape = self.machinelearning.testing_dataframe.values.shape
            ml_testing_data.setText(1, str(shape))
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.dataTreeWidget)
        layout.addWidget(self.buttons)

    def treeClicked(self, index):
        item = self.dataTreeWidget.currentItem()
        if item and item.text(0) in [
            "Descriptor data",
            "Feature selection data",
            "Feature normalization data",
            "Machine learning training data",
            "Machine learning testing data",
            "Dimensionality reduction data",
        ]:
            self.data_source = item.text(0)

    def getDataSource(self):
        return self.data_source

    @staticmethod
    def getValues(
        descriptor=None, selection=None, machinelearning=None, reduction=None
    ):
        dialog = QDataSelection(descriptor, selection, machinelearning, reduction)
        result = dialog.exec_()
        data_source = dialog.getDataSource()
        return data_source, result == QDialog.Accepted


class QSwarmInput(QDialog):
    def __init__(self, algorithm):
        super(QSwarmInput, self).__init__()
        self.algorithm = algorithm
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QIcon("images/logo.ico"))
        self.setWindowTitle(self.algorithm)
        self.setFont(QFont("Arial", 8))
        self.resize(430, 260)
        layout = QFormLayout(self)
        self.N = MyLineEdit("60")
        self.N.clicked.connect(self.set_N)
        self.max_Iter = MyLineEdit("30")
        self.max_Iter.clicked.connect(self.set_max_Iter)
        self.basic_model = MyLineEdit("Logistic Regression")
        self.basic_model.clicked.connect(self.set_basic_model)
        self.alpha = MyLineEdit("0.9")
        self.alpha.clicked.connect(self.set_alpha)
        self.beta = MyLineEdit("0.1")
        self.beta.clicked.connect(self.set_beta)
        self.thres = MyLineEdit("0.5")
        self.thres.clicked.connect(self.set_thres)
        self.tau = MyLineEdit("1")
        self.tau.clicked.connect(self.set_tau)
        self.rho = MyLineEdit("0.2")
        self.rho.clicked.connect(self.set_rho)
        self.eta = MyLineEdit("1")
        self.eta.clicked.connect(self.set_eta)

        # ACS
        if self.algorithm == "ACS":
            self.phi = MyLineEdit("0.5")
            self.phi.clicked.connect(self.set_phi)
        # AOA
        self.Alpha = MyLineEdit("5")
        self.Alpha.clicked.connect(self.set_Alpha)
        if self.algorithm == "AOA":
            self.Mu = MyLineEdit("0.5")
            self.Mu.clicked.connect(self.set_Mu)
        # ABC
        self.max_limit = MyLineEdit("5")
        self.max_limit.clicked.connect(self.set_max_limit)
        # ABO
        self.step_e = MyLineEdit("0.05")
        self.step_e.clicked.connect(self.set_step_e)
        self.ratio = MyLineEdit("0.2")
        self.ratio.clicked.connect(self.set_ratio)
        if self.algorithm == "ABO":
            self.type = MyLineEdit("1")
            self.type.clicked.connect(self.set_type)
        # BA
        if self.algorithm == "BA":
            self.gamma = MyLineEdit("0.9")
            self.gamma.clicked.connect(self.set_gamma)
        self.A_max = MyLineEdit("2")
        self.A_max.clicked.connect(self.set_A_max)
        self.r0_max = MyLineEdit("1")
        self.r0_max.clicked.connect(self.set_r0_max)
        # BDMO
        self.nBabysitter = MyLineEdit("3")
        self.nBabysitter.clicked.connect(self.set_nBabysitter)
        self.peep = MyLineEdit("1.0")
        self.peep.clicked.connect(self.set_peep)
        # BOA
        if self.algorithm == "BOA":
            self.c = MyLineEdit("0.01")
            self.c.clicked.connect(self.set_c)
            self.p = MyLineEdit("0.8")
            self.p.clicked.connect(self.set_p)
        # CSA
        self.AP = MyLineEdit("0.1")
        self.AP.clicked.connect(self.set_AP)
        self.f1 = MyLineEdit("1.5")
        self.f1.clicked.connect(self.set_f1)
        # CS
        self.Pa = MyLineEdit("0.25")
        self.Pa.clicked.connect(self.set_Pa)
        # DE
        if self.algorithm == "DE":
            self.CR = MyLineEdit("0.9")
            self.CR.clicked.connect(self.set_CR)
        self.F = MyLineEdit("0.5")
        self.F.clicked.connect(self.set_F)
        # DOA
        if self.algorithm == "DOA":
            self.P = MyLineEdit("0.5")
            self.P.clicked.connect(self.set_P)
        self.Q = MyLineEdit("0.7")
        self.Q.clicked.connect(self.set_Q)
        # DAOA
        if self.algorithm == "DAOA":
            self.Mu = MyLineEdit("0.001")
            self.Mu.clicked.connect(self.set_Mu)
        # EPO
        self.M = MyLineEdit("2")
        self.M.clicked.connect(self.set_M)
        self.f = MyLineEdit("3")
        self.f.clicked.connect(self.set_f)
        self.l = MyLineEdit("2")
        self.l.clicked.connect(self.set_l)
        # FA
        self.beta0 = MyLineEdit("1")
        self.beta0.clicked.connect(self.set_beta0)
        if self.algorithm == "FA":
            self.gamma = MyLineEdit("1")
            self.gamma.clicked.connect(self.set_gamma)
            self.theta = MyLineEdit("0.97")
            self.theta.clicked.connect(self.set_theta)
        # FPA
        if self.algorithm == "FPA":
            self.P = MyLineEdit("0.8")
            self.P.clicked.connect(self.set_P)
        # GA
        if self.algorithm == "GA":
            self.CR = MyLineEdit("0.8")
            self.CR.clicked.connect(self.set_CR)
            self.MR = MyLineEdit("0.01")
            self.MR.clicked.connect(self.set_MR)
        # GAT
        if self.algorithm == "GAT":
            self.CR = MyLineEdit("0.8")
            self.CR.clicked.connect(self.set_CR)
            self.MR = MyLineEdit("0.01")
            self.MR.clicked.connect(self.set_MR)
        self.Tour_size = MyLineEdit("3")
        self.Tour_size.clicked.connect(self.set_Tour_size)
        # GSA
        self.G0 = MyLineEdit("100")
        self.G0.clicked.connect(self.set_G0)
        # HS
        self.PAR = MyLineEdit("0.05")
        self.PAR.clicked.connect(self.set_PAR)
        self.HMCR = MyLineEdit("0.7")
        self.HMCR.clicked.connect(self.set_HMCR)
        self.bw = MyLineEdit("0.2")
        self.bw.clicked.connect(self.set_bw)
        # HGSO
        self.num_gas = MyLineEdit("2")
        self.num_gas.clicked.connect(self.set_num_gas)
        self.K = MyLineEdit("1")
        self.K.clicked.connect(self.set_K)
        self.L1 = MyLineEdit("0.005")
        self.L1.clicked.connect(self.set_L1)
        self.L2 = MyLineEdit("100")
        self.L2.clicked.connect(self.set_L2)
        self.L3 = MyLineEdit("0.01")
        self.L3.clicked.connect(self.set_L3)
        if self.algorithm == "HGSO":
            self.c1 = MyLineEdit("0.1")
            self.c1.clicked.connect(self.set_c1)
            self.c2 = MyLineEdit("0.2")
            self.c2.clicked.connect(self.set_c2)
        # HLO
        self.pi = MyLineEdit("0.85")
        self.pi.clicked.connect(self.set_pi)
        self.pr = MyLineEdit("0.1")
        self.pr.clicked.connect(self.set_pr)
        # HPO
        self.B = MyLineEdit("0.7")
        self.B.clicked.connect(self.set_B)
        # MRFO
        self.S = MyLineEdit("2")
        self.S.clicked.connect(self.set_S)
        # MPA
        if self.algorithm == "MPA":
            self.P = MyLineEdit("0.5")
            self.P.clicked.connect(self.set_P)
        self.FADs = MyLineEdit("0.2")
        self.FADs.clicked.connect(self.set_FADs)
        # MBO
        self.peri = MyLineEdit("1.2")
        self.peri.clicked.connect(self.set_peri)
        if self.algorithm == "MBO":
            self.p = MyLineEdit("0.5")
            self.p.clicked.connect(self.set_p)
        self.Smax = MyLineEdit("1")
        self.Smax.clicked.connect(self.set_Smax)
        self.BAR = MyLineEdit("0.5")
        self.BAR.clicked.connect(self.set_BAR)
        self.num_land1 = MyLineEdit("4")
        self.num_land1.clicked.connect(self.set_num_land1)
        # MFO
        if self.algorithm == "MFO":
            self.b = MyLineEdit("1")
            self.b.clicked.connect(self.set_b)
        # MVO
        if self.algorithm == "MVO":
            self.p = MyLineEdit("6")
            self.p.clicked.connect(self.set_p)
            self.type = MyLineEdit("1")
            self.type.clicked.connect(self.set_type)
        # PSO
        if self.algorithm == "PSO":
            self.c1 = MyLineEdit("2")
            self.c1.clicked.connect(self.set_c1)
            self.c2 = MyLineEdit("2")
            self.c2.clicked.connect(self.set_c2)
        self.w = MyLineEdit("0.9")
        self.w.clicked.connect(self.set_w)
        # PRO
        self.Pmut = MyLineEdit("0.06")
        self.Pmut.clicked.connect(self.set_Pmut)
        # SBO
        if self.algorithm == "SBO":
            self.z = MyLineEdit("0.02")
            self.z.clicked.connect(self.set_z)
            self.MR = MyLineEdit("0.05")
            self.MR.clicked.connect(self.set_MR)
        # SA
        if self.algorithm == "SA":
            self.c = MyLineEdit("0.93")
            self.c.clicked.connect(self.set_c)
        self.T0 = MyLineEdit("100")
        self.T0.clicked.connect(self.set_T0)
        # SMA
        if self.algorithm == "SMA":
            self.z = MyLineEdit("0.03")
            self.z.clicked.connect(self.set_z)
        # TGA
        self.num_tree1 = MyLineEdit("3")
        self.num_tree1.clicked.connect(self.set_num_tree1)
        self.num_tree2 = MyLineEdit("3")
        self.num_tree2.clicked.connect(self.set_num_tree2)
        self.num_tree4 = MyLineEdit("3")
        self.num_tree4.clicked.connect(self.set_num_tree4)
        if self.algorithm == "TGA":
            self.theta = MyLineEdit("0.8")
            self.theta.clicked.connect(self.set_theta)
        self.lambda_ = MyLineEdit("0.5")
        self.lambda_.clicked.connect(self.set_lambda_)
        # TSA
        self.ST = MyLineEdit("0.1")
        self.ST.clicked.connect(self.set_ST)
        # WSA
        self.sl = MyLineEdit("0.035")
        self.sl.clicked.connect(self.set_sl)
        if self.algorithm == "WSA":
            self.phi = MyLineEdit("0.001")
            self.phi.clicked.connect(self.set_phi)
        self.lambda_val = MyLineEdit("0.75")
        self.lambda_val.clicked.connect(self.set_lambda_val)
        # WOA
        if self.algorithm == "WOA":
            self.b = MyLineEdit("1")
            self.b.clicked.connect(self.set_b)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        layout.addRow("N", self.N)
        layout.addRow("max_Iter", self.max_Iter)
        layout.addRow("basic_model", self.basic_model)
        layout.addRow("alpha", self.alpha)
        layout.addRow("beta", self.beta)
        layout.addRow("thres", self.thres)
        layout.addRow("tau", self.tau)
        layout.addRow("rho", self.rho)
        layout.addRow("eta", self.eta)

        if self.algorithm == "ACS":
            layout.addRow("phi", self.phi)
        if self.algorithm == "AOA":
            layout.addRow("Alpha", self.Alpha)
            layout.addRow("Mu", self.Mu)
        if self.algorithm == "ABC":
            layout.addRow("max_limit", self.max_limit)
        if self.algorithm == "ABO":
            layout.addRow("step_e", self.step_e)
            layout.addRow("ratio", self.ratio)
            layout.addRow("type", self.type)
        if self.algorithm == "BA":
            layout.addRow("gamma", self.gamma)
            layout.addRow("A_max", self.A_max)
            layout.addRow("r0_max", self.r0_max)
        if self.algorithm == "BDMO":
            layout.addRow("nBabysitter", self.nBabysitter)
            layout.addRow("peep", self.peep)
        if self.algorithm == "BOA":
            layout.addRow("c", self.c)
            layout.addRow("p", self.p)
        if self.algorithm == "CSA":
            layout.addRow("AP", self.AP)
            layout.addRow("f1", self.f1)
        if self.algorithm == "CS":
            layout.addRow("Pa", self.Pa)
        if self.algorithm == "DE":
            layout.addRow("CR", self.CR)
            layout.addRow("F", self.F)
        if self.algorithm == "DOA":
            layout.addRow("P", self.P)
            layout.addRow("Q", self.Q)
        if self.algorithm == "DAOA":
            layout.addRow("Mu", self.Mu)
        if self.algorithm == "EPO":
            layout.addRow("M", self.M)
            layout.addRow("f", self.f)
            layout.addRow("l", self.l)
        if self.algorithm == "FA":
            layout.addRow("beta0", self.beta0)
            layout.addRow("gamma", self.gamma)
            layout.addRow("theta", self.theta)
        if self.algorithm == "FPA":
            layout.addRow("P", self.P)
        if self.algorithm == "GA":
            layout.addRow("CR", self.CR)
            layout.addRow("MR", self.MR)
        if self.algorithm == "GAT":
            layout.addRow("CR", self.CR)
            layout.addRow("MR", self.MR)
            layout.addRow("Tour_size", self.Tour_size)
        if self.algorithm == "GSA":
            layout.addRow("G0", self.G0)
        if self.algorithm == "HS":
            layout.addRow("PAR", self.PAR)
            layout.addRow("HMCR", self.HMCR)
            layout.addRow("bw", self.bw)
        if self.algorithm == "HGSO":
            layout.addRow("num_gas", self.num_gas)
            layout.addRow("K", self.K)
            layout.addRow("L1", self.L1)
            layout.addRow("L2", self.L2)
            layout.addRow("L3", self.L3)
            layout.addRow("c1", self.c1)
            layout.addRow("c2", self.c2)
        if self.algorithm == "HLO":
            layout.addRow("pi", self.pi)
            layout.addRow("pr", self.pr)
        if self.algorithm == "HPO":
            layout.addRow("B", self.B)
        if self.algorithm == "MRFO":
            layout.addRow("S", self.S)
        if self.algorithm == "MPA":
            layout.addRow("P", self.P)
            layout.addRow("FADs", self.FADs)
        if self.algorithm == "MBO":
            layout.addRow("peri", self.peri)
            layout.addRow("p", self.p)
            layout.addRow("Smax", self.Smax)
            layout.addRow("BAR", self.BAR)
            layout.addRow("num_land1", self.num_land1)
        if self.algorithm == "MFO":
            layout.addRow("b", self.b)
        if self.algorithm == "MVO":
            layout.addRow("p", self.p)
            layout.addRow("type", self.type)
        if self.algorithm == "PSO":
            layout.addRow("c1", self.c1)
            layout.addRow("c2", self.c2)
            layout.addRow("w", self.w)
        if self.algorithm == "PRO":
            layout.addRow("Pmut", self.Pmut)
        if self.algorithm == "SBO":
            layout.addRow("z", self.z)
            layout.addRow("MR", self.MR)
        if self.algorithm == "SA":
            layout.addRow("c", self.c)
            layout.addRow("T0", self.T0)
        if self.algorithm == "SMA":
            layout.addRow("z", self.z)
        if self.algorithm == "TGA":
            layout.addRow("num_tree1", self.num_tree1)
            layout.addRow("num_tree2", self.num_tree2)
            layout.addRow("num_tree4", self.num_tree4)
            layout.addRow("theta", self.theta)
            layout.addRow("lambda_", self.lambda_)
        if self.algorithm == "TSA":
            layout.addRow("ST", self.ST)
        if self.algorithm == "WSA":
            layout.addRow("sl", self.sl)
            layout.addRow("phi", self.phi)
            layout.addRow("lambda_val", self.lambda_val)
        if self.algorithm == "WOA":
            layout.addRow("b", self.b)

        layout.addRow(self.buttons)

    def set_N(self):
        number, ok = QInputDialog.getInt(
            self, "N", "Set N number", 60, 10, 200, 10
        )
        if ok:
            self.N.setText(str(number))

    def get_N(self):
        try:
            number = int(self.N.text())
            return number
        except Exception as e:
            return 60

    def set_max_Iter(self):
        number, ok = QInputDialog.getInt(
            self, "max_Iter", "Set max_Iter number", 30, 10, 100, 10
        )
        if ok:
            self.max_Iter.setText(str(number))

    def get_max_Iter(self):
        try:
            number = int(self.max_Iter.text())
            return number
        except Exception as e:
            return 30

    def set_basic_model(self):
        models = [
            "AdaBoost",
            "Bagging Classifier",
            "Bernoulli Naive Bayes",
            "Calibrated Classifier",
            "Decision Tree",
            "Extra Trees Classifier",
            "Gaussian Process",
            "Gradient Boosting",
            "Gaussian Naive Bayes",
            "K-Nearest Neighbors",
            "Linear Discriminant Analysis",
            "Logistic Regression",
            "LightGBM",
            "MLP Neural Network",
            "Quadratic Discriminant Analysis",
            "Random Forest",
            "XGBoost",
        ]  
        selected_model, ok = QInputDialog.getItem(
            self, "Select Basic Model", "Select basic_model", models, 0, False
        )
        if ok:
            self.basic_model.setText(selected_model)

    def get_basic_model(self):
        try:
            number = str(self.basic_model.text())
            return number
        except Exception as e:
            return "Logistic Regression" 

    def set_alpha(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set alpha number")
        dialog.setLabelText("Enter a value for alpha:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        dialog.setDoubleRange(0.0, 1.0) 
        dialog.setDoubleDecimals(1)  
        dialog.setDoubleValue(0.9)
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.alpha.setText(str(number))

    def get_alpha(self):
        try:
            number = float(self.alpha.text())
            return number
        except Exception as e:
            return 0.9

    def set_beta(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set beta number")
        dialog.setLabelText("Enter a value for beta:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        dialog.setDoubleRange(0.0, 1.0)  
        dialog.setDoubleDecimals(1)    
        dialog.setDoubleValue(0.1)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.beta.setText(str(number))

    def get_beta(self):
        try:
            number = float(self.beta.text())
            return number
        except Exception as e:
            return 0.1

    def set_thres(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set thres number")
        dialog.setLabelText("Enter a value for thres:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        dialog.setDoubleRange(0.0, 1.0)  
        dialog.setDoubleDecimals(1)    
        dialog.setDoubleValue(0.5)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.thres.setText(str(number))

    def get_thres(self):
        try:
            number = float(self.thres.text())
            return number
        except Exception as e:
            return 0.5

    def set_tau(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set tau number")
        dialog.setLabelText("Enter a value for tau:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        dialog.setDoubleRange(0.0, 1.0)  
        dialog.setDoubleDecimals(1)    
        dialog.setDoubleValue(1.0)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.tau.setText(str(number))

    def get_tau(self):
        try:
            number = float(self.tau.text())
            return number
        except Exception as e:
            return 1

    def set_rho(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set rho number")
        dialog.setLabelText("Enter a value for rho:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        dialog.setDoubleRange(0.0, 1.0)  
        dialog.setDoubleDecimals(1)    
        dialog.setDoubleValue(0.2)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.rho.setText(str(number))

    def get_rho(self):
        try:
            number = float(self.rho.text())
            return number
        except Exception as e:
            return 0.2

    def set_eta(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set eta number")
        dialog.setLabelText("Enter a value for eta:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        dialog.setDoubleRange(0.0, 1.0)  
        dialog.setDoubleDecimals(1)    
        dialog.setDoubleValue(1.0)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.eta.setText(str(number))

    def get_eta(self):
        try:
            number = float(self.eta.text())
            return number
        except Exception as e:
            return 1

    def set_phi(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set phi number")
        dialog.setLabelText("Enter a value for phi:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        if self.algorithm == "ACS":
            dialog.setDoubleRange(0.0, 1.0)  
            dialog.setDoubleDecimals(1)    
            dialog.setDoubleValue(0.5)     
        if self.algorithm == "WSA":
            dialog.setDoubleRange(0.0, 1.0)  
            dialog.setDoubleDecimals(3)    
            dialog.setDoubleValue(0.001)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.phi.setText(str(number))

    def get_phi(self):
        try:
            if self.algorithm == "WSA" or self.algorithm == "ACS":
                number = float(self.phi.text())
                return number
        except Exception as e:
            if self.algorithm == "ACS":
                return 0.5
            elif self.algorithm == "WSA":
                return 0.001

    def set_Alpha(self):
        number, ok = QInputDialog.getInt(self, "Alpha", "Set Alpha number", 5, 1, 10, 1)
        if ok:
            self.Alpha.setText(str(number))

    def get_Alpha(self):
        try:
            if self.algorithm == "AOA":
                number = int(self.Alpha.text())
                return number
        except Exception as e:
            return 5

    def set_Mu(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set Mu number")
        dialog.setLabelText("Enter a value for Mu:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        if self.algorithm == "AOA":
            dialog.setDoubleRange(0.0, 1.0)  
            dialog.setDoubleDecimals(1)    
            dialog.setDoubleValue(0.5)     
        elif self.algorithm == "DAOA":
            dialog.setDoubleRange(0.0, 1.0)  
            dialog.setDoubleDecimals(3)    
            dialog.setDoubleValue(0.001)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.Mu.setText(str(number))

    def get_Mu(self):
        try:
            if self.algorithm == "AOA" or self.algorithm == "DAOA":
                number = float(self.Mu.text())
                return number
        except Exception as e:
            if self.algorithm == "AOA":
                return 0.5
            elif self.algorithm == "DAOA":
                return 0.001

    def set_max_limit(self):
        number, ok = QInputDialog.getInt(
            self, "max_limit", "Set max_limit number", 5, 1, 20, 1
        )
        if ok:
            self.max_limit.setText(str(number))

    def get_max_limit(self):
        try:
            if self.algorithm == "ABC":
                number = int(self.max_limit.text())
                return number
        except Exception as e:
            return 5

    def set_step_e(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set step_e number")
        dialog.setLabelText("Enter a value for step_e:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        dialog.setDoubleRange(0.0, 1.0)  
        dialog.setDoubleDecimals(2)    
        dialog.setDoubleValue(0.05)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.step_e.setText(str(number))

    def get_step_e(self):
        try:
            if self.algorithm == "ABO":
                number = float(self.step_e.text())
                return number
        except Exception as e:
            return 0.05

    def set_ratio(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set ratio number")
        dialog.setLabelText("Enter a value for ratio:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        dialog.setDoubleRange(0.0, 1.0)  
        dialog.setDoubleDecimals(1)    
        dialog.setDoubleValue(0.2)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.ratio.setText(str(number))

    def get_ratio(self):
        try:
            if self.algorithm == "ABO":
                number = float(self.ratio.text())
                return number
        except Exception as e:
            return 0.2

    def set_type(self):
        number, ok = QInputDialog.getInt(self, "type", "Set type number", 1, 1, 2, 1)
        if ok:
            self.type.setText(str(number))

    def get_type(self):
        try:
            if self.algorithm == "ABO" or self.algorithm == "MVO":
                number = int(self.type.text())
                return number
        except Exception as e:
            return 1

    def set_gamma(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set gamma number")
        dialog.setLabelText("Enter a value for gamma:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        if self.algorithm == "BA":
            dialog.setDoubleRange(0.0, 1.0)  
            dialog.setDoubleDecimals(1)    
            dialog.setDoubleValue(0.9)     
        elif self.algorithm == "FA":
            dialog.setDoubleRange(1, 10)  
            dialog.setDoubleDecimals(1)    
            dialog.setDoubleValue(1)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.gamma.setText(str(number))

    def get_gamma(self):
        try:
            if self.algorithm == "BA" or self.algorithm == "FA":
                number = float(self.gamma.text())
                return number
        except Exception as e:
            if self.algorithm == "BA":
                return 0.9
            elif self.algorithm == "FA":
                return 1

    def set_A_max(self):
        number, ok = QInputDialog.getInt(self, "A_max", "Set A_max number", 2, 1, 10, 1)
        if ok:
            self.A_max.setText(str(number))

    def get_A_max(self):
        try:
            if self.algorithm == "BA":
                number = int(self.A_max.text())
                return number
        except Exception as e:
            return 2

    def set_r0_max(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set r0_max number")
        dialog.setLabelText("Enter a value for r0_max:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        dialog.setDoubleRange(0.0, 1.0)  
        dialog.setDoubleDecimals(1)    
        dialog.setDoubleValue(1.0)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.r0_max.setText(str(number))

    def get_r0_max(self):
        try:
            if self.algorithm == "BA":
                number = float(self.r0_max.text())
                return number
        except Exception as e:
            return 1.0

    def set_nBabysitter(self):
        number, ok = QInputDialog.getInt(
            self, "nBabysitter", "Set nBabysitter number", 3, 1, 10, 1
        )
        if ok:
            self.nBabysitter.setText(str(number))

    def get_nBabysitter(self):
        try:
            if self.algorithm == "BDMO":
                number = int(self.nBabysitter.text())
                return number
        except Exception as e:
            return 3

    def set_peep(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set peep number")
        dialog.setLabelText("Enter a value for peep:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        dialog.setDoubleRange(0.0, 1.0)  
        dialog.setDoubleDecimals(1)    
        dialog.setDoubleValue(1.0)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.peep.setText(str(number))

    def get_peep(self):
        try:
            if self.algorithm == "BDMO":
                number = float(self.peep.text())
                return number
        except Exception as e:
            return 1.0

    def set_c(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set c number")
        dialog.setLabelText("Enter a value for c:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        if self.algorithm == "BOA":
            dialog.setDoubleRange(0.0, 1.0)  
            dialog.setDoubleDecimals(2)    
            dialog.setDoubleValue(0.01)     
        if self.algorithm == "SA":
            dialog.setDoubleRange(0.0, 1.0)  
            dialog.setDoubleDecimals(2)    
            dialog.setDoubleValue(0.93)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.c.setText(str(number))

    def get_c(self):
        try:
            if self.algorithm == "BOA" or self.algorithm == "SA":
                number = float(self.c.text())
                return number
        except Exception as e:
            if self.algorithm == "BOA":
                return 0.01
            elif self.algorithm == "SA":
                return 0.93

    def set_p(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set p number")
        dialog.setLabelText("Enter a value for p:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        if self.algorithm == "BOA":
            dialog.setDoubleRange(0.0, 1.0)  
            dialog.setDoubleDecimals(1)    
            dialog.setDoubleValue(0.8)     
        elif self.algorithm == "MBO":
            dialog.setDoubleRange(0.0, 1.0)  
            dialog.setDoubleDecimals(1)    
            dialog.setDoubleValue(0.5)     
        elif self.algorithm == "MVO":
            dialog.setDoubleRange(0.0, 10.0)  
            dialog.setDoubleDecimals(1)    
            dialog.setDoubleValue(6.0)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.p.setText(str(number))

    def get_p(self):
        try:
            if (
                self.algorithm == "BOA"
                or self.algorithm == "MBO"
                or self.algorithm == "MVO"
            ):
                number = float(self.p.text())
                return number
        except Exception as e:
            if self.algorithm == "BOA":
                return 0.8
            elif self.algorithm == "MBO":
                return 0.5
            elif self.algorithm == "MVO":
                return 6.0

    def set_AP(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set AP number")
        dialog.setLabelText("Enter a value for AP:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        dialog.setDoubleRange(0.0, 1.0)  
        dialog.setDoubleDecimals(1)    
        dialog.setDoubleValue(0.1)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.AP.setText(str(number))

    def get_AP(self):
        try:
            if self.algorithm == "CSA":
                number = float(self.AP.text())
                return number
        except Exception as e:
            return 0.1

    def set_f1(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set f1 number")
        dialog.setLabelText("Enter a value for f1:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        dialog.setDoubleRange(0.0, 2.0)  
        dialog.setDoubleDecimals(1)    
        dialog.setDoubleValue(1.5)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.f1.setText(str(number))

    def get_f1(self):
        try:
            if self.algorithm == "CSA":
                number = float(self.f1.text())
                return number
        except Exception as e:
            return 1.5

    def set_Pa(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set Pa number")
        dialog.setLabelText("Enter a value for Pa:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        dialog.setDoubleRange(0.0, 1.0)  
        dialog.setDoubleDecimals(2)    
        dialog.setDoubleValue(0.25)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.Pa.setText(str(number))

    def get_Pa(self):
        try:
            if self.algorithm == "CS":
                number = float(self.Pa.text())
                return number
        except Exception as e:
            return 0.25

    def set_CR(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set CR number")
        dialog.setLabelText("Enter a value for CR:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        if self.algorithm == "DE":
            dialog.setDoubleRange(0.0, 1.0)  
            dialog.setDoubleDecimals(1)    
            dialog.setDoubleValue(0.9)     
        if self.algorithm == "GA" or self.algorithm == "GAT":
            dialog.setDoubleRange(0.0, 1.0)  
            dialog.setDoubleDecimals(1)    
            dialog.setDoubleValue(0.8)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.CR.setText(str(number))

    def get_CR(self):
        try:
            if (
                self.algorithm == "DE"
                or self.algorithm == "GA"
                or self.algorithm == "GAT"
            ):
                number = float(self.CR.text())
                return number
        except Exception as e:
            if self.algorithm == "DE":
                return 0.9
            elif self.algorithm == "GA" or self.algorithm == "GAT":
                return 0.8

    def set_F(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set F number")
        dialog.setLabelText("Enter a value for F:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        dialog.setDoubleRange(0.0, 2.0)  
        dialog.setDoubleDecimals(1)    
        dialog.setDoubleValue(0.5)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.F.setText(str(number))

    def get_F(self):
        try:
            if self.algorithm == "DE":
                number = float(self.F.text())
                return number
        except Exception as e:
            return 0.5

    def set_P(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set P number")
        dialog.setLabelText("Enter a value for P:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        if self.algorithm == "DOA":
            dialog.setDoubleRange(0.0, 1.0)  
            dialog.setDoubleDecimals(1)    
            dialog.setDoubleValue(0.5)     
        elif self.algorithm == "FPA":
            dialog.setDoubleRange(0.0, 1.0)  
            dialog.setDoubleDecimals(1)    
            dialog.setDoubleValue(0.8)     
        elif self.algorithm == "MPA":
            dialog.setDoubleRange(0.0, 1.0)  
            dialog.setDoubleDecimals(1)    
            dialog.setDoubleValue(0.5)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.P.setText(str(number))

    def get_P(self):
        try:
            if (
                self.algorithm == "DOA"
                or self.algorithm == "FPA"
                or self.algorithm == "MPA"
            ):
                number = float(self.P.text())
                return number
        except Exception as e:
            if self.algorithm == "DOA" or self.algorithm == "MPA":
                return 0.5
            elif self.algorithm == "FPA":
                return 0.8

    def set_Q(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set Q number")
        dialog.setLabelText("Enter a value for Q:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        dialog.setDoubleRange(0.0, 1.0)  
        dialog.setDoubleDecimals(1)    
        dialog.setDoubleValue(0.7)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.Q.setText(str(number))

    def get_Q(self):
        try:
            if self.algorithm == "DOA":
                number = float(self.Q.text())
                return number
        except Exception as e:
            return 0.7

    def set_M(self):
        number, ok = QInputDialog.getInt(self, "M", "Set M number", 2, 1, 10, 1)
        if ok:
            self.M.setText(str(number))

    def get_M(self):
        try:
            if self.algorithm == "EPO":
                number = int(self.M.text())
                return number
        except Exception as e:
            return 2

    def set_f(self):
        number, ok = QInputDialog.getInt(self, "f", "Set f number", 3, 1, 10, 1)
        if ok:
            self.f.setText(str(number))

    def get_f(self):
        try:
            if self.algorithm == "EPO":
                number = int(self.f.text())
                return number
        except Exception as e:
            return 3

    def set_l(self):
        number, ok = QInputDialog.getInt(self, "l", "Set l number", 2, 1, 10, 1)
        if ok:
            self.l.setText(str(number))

    def get_l(self):
        try:
            if self.algorithm == "EPO":
                number = int(self.l.text())
                return number
        except Exception as e:
            return 2

    def set_beta0(self):
        number, ok = QInputDialog.getInt(self, "beta0", "Set beta0 number", 1, 1, 10, 1)
        if ok:
            self.beta0.setText(str(number))

    def get_beta0(self):
        try:
            if self.algorithm == "FA":
                number = int(self.beta0.text())
                return number
        except Exception as e:
            return 1

    def set_theta(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set theta number")
        dialog.setLabelText("Enter a value for theta:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        if self.algorithm == "FA":
            dialog.setDoubleRange(0.0, 1.0)  
            dialog.setDoubleDecimals(2)    
            dialog.setDoubleValue(0.97)     
        if self.algorithm == "TGA":
            dialog.setDoubleRange(0.0, 1.0)  
            dialog.setDoubleDecimals(2)    
            dialog.setDoubleValue(0.80)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.theta.setText(str(number))

    def get_theta(self):
        try:
            if self.algorithm == "FA" or self.algorithm == "TGA":
                number = float(self.theta.text())
                return number
        except Exception as e:
            if self.algorithm == "FA":
                return 0.97
            elif self.algorithm == "TGA":
                return 0.8

    def set_MR(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set MR number")
        dialog.setLabelText("Enter a value for MR:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        if self.algorithm == "GA" or self.algorithm == "GAT":
            dialog.setDoubleRange(0.0, 1.0)  
            dialog.setDoubleDecimals(2)    
            dialog.setDoubleValue(0.01)     
        if self.algorithm == "SBO":
            dialog.setDoubleRange(0.0, 1.0)  
            dialog.setDoubleDecimals(2)    
            dialog.setDoubleValue(0.05)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.MR.setText(str(number))

    def get_MR(self):
        try:
            if (
                self.algorithm == "GA"
                or self.algorithm == "GAT"
                or self.algorithm == "SBO"
            ):
                number = float(self.MR.text())
                return number
        except Exception as e:
            if self.algorithm == "GA" or self.algorithm == "GAT":
                return 0.01
            elif self.algorithm == "SBO":
                return 0.05

    def set_Tour_size(self):
        number, ok = QInputDialog.getInt(
            self, "Tour_size", "Set Tour_size number", 3, 1, 10, 1
        )
        if ok:
            self.Tour_size.setText(str(number))

    def get_Tour_size(self):
        try:
            if self.algorithm == "GAT":
                number = int(self.Tour_size.text())
                return number
        except Exception as e:
            return 3

    def set_G0(self):
        number, ok = QInputDialog.getInt(self, "G0", "Set G0 number", 100, 1, 200, 10)
        if ok:
            self.G0.setText(str(number))

    def get_G0(self):
        try:
            if self.algorithm == "GSA":
                number = int(self.G0.text())
                return number
        except Exception as e:
            return 100

    def set_PAR(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set PAR number")
        dialog.setLabelText("Enter a value for PAR:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        dialog.setDoubleRange(0.0, 1.0)  
        dialog.setDoubleDecimals(2)    
        dialog.setDoubleValue(0.05)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.PAR.setText(str(number))

    def get_PAR(self):
        try:
            if self.algorithm == "HS":
                number = float(self.PAR.text())
                return number
        except Exception as e:
            return 0.05

    def set_HMCR(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set HMCR number")
        dialog.setLabelText("Enter a value for HMCR:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        dialog.setDoubleRange(0.0, 1.0)  
        dialog.setDoubleDecimals(1)    
        dialog.setDoubleValue(0.7)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.HMCR.setText(str(number))

    def get_HMCR(self):
        try:
            if self.algorithm == "HS":
                number = float(self.HMCR.text())
                return number
        except Exception as e:
            return 0.7

    def set_bw(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set bw number")
        dialog.setLabelText("Enter a value for bw:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        dialog.setDoubleRange(0.0, 1.0)  
        dialog.setDoubleDecimals(1)    
        dialog.setDoubleValue(0.2)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.bw.setText(str(number))

    def get_bw(self):
        try:
            if self.algorithm == "HS":
                number = float(self.bw.text())
                return number
        except Exception as e:
            return 0.2

    def set_num_gas(self):
        number, ok = QInputDialog.getInt(
            self, "num_gas", "Set num_gas number", 2, 1, 10, 1
        )
        if ok:
            self.num_gas.setText(str(number))

    def get_num_gas(self):
        try:
            if self.algorithm == "HGSO":
                number = int(self.num_gas.text())
                return number
        except Exception as e:
            return 2

    def set_K(self):
        number, ok = QInputDialog.getInt(self, "K", "Set K number", 1, 1, 10, 1)
        if ok:
            self.K.setText(str(number))

    def get_K(self):
        try:
            if self.algorithm == "HGSO":
                number = int(self.K.text())
                return number
        except Exception as e:
            return 1

    def set_L1(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set L1 number")
        dialog.setLabelText("Enter a value for L1:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        dialog.setDoubleRange(0.0, 1.0)  
        dialog.setDoubleDecimals(3)    
        dialog.setDoubleValue(0.005)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.L1.setText(str(number))

    def get_L1(self):
        try:
            if self.algorithm == "HGSO":
                number = float(self.L1.text())
                return number
        except Exception as e:
            return 0.005

    def set_L2(self):
        number, ok = QInputDialog.getInt(self, "L2", "Set L2 number", 1, 1, 200, 10)
        if ok:
            self.L2.setText(str(number))

    def get_L2(self):
        try:
            if self.algorithm == "HGSO":
                number = int(self.L2.text())
                return number
        except Exception as e:
            return 100

    def set_L3(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set L3 number")
        dialog.setLabelText("Enter a value for L3:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        dialog.setDoubleRange(0.0, 1.0)  
        dialog.setDoubleDecimals(2)    
        dialog.setDoubleValue(0.01)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.L3.setText(str(number))

    def get_L3(self):
        try:
            if self.algorithm == "HGSO":
                number = float(self.L3.text())
                return number
        except Exception as e:
            return 0.01

    def set_c1(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set c1 number")
        dialog.setLabelText("Enter a value for c1:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        if self.algorithm == "HGSO":
            dialog.setDoubleRange(0.0, 1.0)  
            dialog.setDoubleDecimals(1)    
            dialog.setDoubleValue(0.1)     
        elif self.algorithm == "PSO":
            dialog.setDoubleRange(1.0, 10.0)  
            dialog.setDoubleDecimals(1)    
            dialog.setDoubleValue(2.0)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.c1.setText(str(number))

    def get_c1(self):
        try:
            if self.algorithm == "HGSO" or self.algorithm == "PSO":
                number = float(self.c1.text())
                return number
        except Exception as e:
            if self.algorithm == "HGSO":
                return 0.1
            elif self.algorithm == "PSO":
                return 2.0

    def set_c2(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set c2 number")
        dialog.setLabelText("Enter a value for c2:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        if self.algorithm == "HGSO":
            dialog.setDoubleRange(0.0, 1.0)  
            dialog.setDoubleDecimals(1)    
            dialog.setDoubleValue(0.2)     
        elif self.algorithm == "PSO":
            dialog.setDoubleRange(1.0, 10.0)  
            dialog.setDoubleDecimals(1)    
            dialog.setDoubleValue(2.0)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.c2.setText(str(number))

    def get_c2(self):
        try:
            if self.algorithm == "HGSO" or self.algorithm == "PSO":
                number = float(self.c2.text())
                return number
        except Exception as e:
            if self.algorithm == "HGSO":
                return 0.2
            elif self.algorithm == "PSO":
                return 2.0

    def set_pi(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set pi number")
        dialog.setLabelText("Enter a value for pi:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        dialog.setDoubleRange(0.0, 1.0)  
        dialog.setDoubleDecimals(2)    
        dialog.setDoubleValue(0.85)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.pi.setText(str(number))

    def get_pi(self):
        try:
            if self.algorithm == "HLO":
                number = float(self.pi.text())
                return number
        except Exception as e:
            return 0.85

    def set_pr(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set pr number")
        dialog.setLabelText("Enter a value for pr:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        dialog.setDoubleRange(0.0, 1.0)  
        dialog.setDoubleDecimals(1)    
        dialog.setDoubleValue(0.1)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.pr.setText(str(number))

    def get_pr(self):
        try:
            if self.algorithm == "HLO":
                number = float(self.pr.text())
                return number
        except Exception as e:
            return 0.1

    def set_B(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set B number")
        dialog.setLabelText("Enter a value for B:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        dialog.setDoubleRange(0.0, 1.0)  
        dialog.setDoubleDecimals(1)    
        dialog.setDoubleValue(0.7)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.B.setText(str(number))

    def get_B(self):
        try:
            if self.algorithm == "HPO":
                number = float(self.B.text())
                return number
        except Exception as e:
            return 0.7

    def set_S(self):
        number, ok = QInputDialog.getInt(self, "S", "Set S number", 2, 1, 10, 1)
        if ok:
            self.S.setText(str(number))

    def get_S(self):
        try:
            if self.algorithm == "MRFO":
                number = int(self.S.text())
                return number
        except Exception as e:
            return 2

    def set_FADs(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set FADs number")
        dialog.setLabelText("Enter a value for FADs:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        dialog.setDoubleRange(0.0, 1.0)  
        dialog.setDoubleDecimals(1)    
        dialog.setDoubleValue(0.2)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.FADs.setText(str(number))

    def get_FADs(self):
        try:
            if self.algorithm == "MPA":
                number = float(self.FADs.text())
                return number
        except Exception as e:
            return 0.2

    def set_peri(self):
        number, ok = QInputDialog.getInt(self, "peri", "Set peri number", 1, 1, 10, 1)
        if ok:
            self.peri.setText(str(number))

    def get_peri(self):
        try:
            if self.algorithm == "MBO":
                number = int(self.peri.text())
                return number
        except Exception as e:
            return 1

    def set_Smax(self):
        number, ok = QInputDialog.getInt(self, "Smax", "Set Smax number", 1, 1, 10, 1)
        if ok:
            self.Smax.setText(str(number))

    def get_Smax(self):
        try:
            if self.algorithm == "MBO":
                number = int(self.Smax.text())
                return number
        except Exception as e:
            return 1

    def set_BAR(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set BAR number")
        dialog.setLabelText("Enter a value for BAR:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        dialog.setDoubleRange(0.0, 1.0)  
        dialog.setDoubleDecimals(1)    
        dialog.setDoubleValue(0.5)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.BAR.setText(str(number))

    def get_BAR(self):
        try:
            if self.algorithm == "MBO":
                number = float(self.BAR.text())
                return number
        except Exception as e:
            return 0.5

    def set_num_land1(self):
        number, ok = QInputDialog.getInt(
            self, "num_land1", "Set num_land1 number", 4, 1, 10, 1
        )
        if ok:
            self.num_land1.setText(str(number))

    def get_num_land1(self):
        try:
            if self.algorithm == "MBO":
                number = int(self.num_land1.text())
                return number
        except Exception as e:
            return 4

    def set_b(self):
        if self.algorithm == "MFO":
            number, ok = QInputDialog.getInt(self, "b", "Set b number", 1, 1, 10, 1)
            if ok:
                self.b.setText(str(number))
        if self.algorithm == "WOA":
            dialog = QInputDialog(self)
            dialog.setWindowTitle("Set b number")
            dialog.setLabelText("Enter a value for b:")
            dialog.setInputMode(QInputDialog.DoubleInput)
            dialog.setDoubleRange(0.0, 2.0)  
            dialog.setDoubleDecimals(1)    
            dialog.setDoubleValue(1.0)     
            if dialog.exec_() == QInputDialog.Accepted:
                number = dialog.doubleValue()
                self.b.setText(str(number))

    def get_b(self):
        try:
            if self.algorithm == "MFO":
                number = int(self.b.text())
                return number
            if self.algorithm == "WOA":
                number = float(self.b.text())
                return number
        except Exception as e:
            return 1

    def set_w(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set w number")
        dialog.setLabelText("Enter a value for w:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        dialog.setDoubleRange(0.0, 1.0)  
        dialog.setDoubleDecimals(1)    
        dialog.setDoubleValue(0.9)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.w.setText(str(number))

    def get_w(self):
        try:
            if self.algorithm == "PSO":
                number = float(self.w.text())
                return number
        except Exception as e:
            return 0.9

    def set_Pmut(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set Pmut number")
        dialog.setLabelText("Enter a value for Pmut:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        dialog.setDoubleRange(0.0, 1.0)  
        dialog.setDoubleDecimals(2)    
        dialog.setDoubleValue(0.06)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.Pmut.setText(str(number))

    def get_Pmut(self):
        try:
            if self.algorithm == "PRO":
                number = float(self.Pmut.text())
                return number
        except Exception as e:
            return 0.06

    def set_z(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set z number")
        dialog.setLabelText("Enter a value for z:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        if self.algorithm == "SBO":
            dialog.setDoubleRange(0.0, 1.0)  
            dialog.setDoubleDecimals(2)    
            dialog.setDoubleValue(0.02) 
        if self.algorithm == "SMA":
            dialog.setDoubleRange(0.0, 1.0)  
            dialog.setDoubleDecimals(2)    
            dialog.setDoubleValue(0.03)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.z.setText(str(number))

    def get_z(self):
        try:
            if self.algorithm == "SBO" or self.algorithm == "SMA":
                number = float(self.z.text())
                return number
        except Exception as e:
            if self.algorithm == "SBO":
                return 0.02
            if self.algorithm == "SMA":
                return 0.03

    def set_T0(self):
        number, ok = QInputDialog.getInt(self, "T0", "Set T0 number", 1, 1, 200, 10)
        if ok:
            self.T0.setText(str(number))

    def get_T0(self):
        try:
            if self.algorithm == "SA":
                number = int(self.T0.text())
                return number
        except Exception as e:
            return 100

    def set_num_tree1(self):
        number, ok = QInputDialog.getInt(
            self, "num_tree1", "Set num_tree1 number", 3, 1, 20, 1
        )
        if ok:
            self.num_tree1.setText(str(number))

    def get_num_tree1(self):
        try:
            if self.algorithm == "TGA":
                number = int(self.num_tree1.text())
                return number
        except Exception as e:
            return 3

    def set_num_tree2(self):
        number, ok = QInputDialog.getInt(
            self, "num_tree2", "Set num_tree2 number", 3, 1, 20, 1
        )
        if ok:
            self.num_tree2.setText(str(number))

    def get_num_tree2(self):
        try:
            if self.algorithm == "TGA":
                number = int(self.num_tree2.text())
                return number
        except Exception as e:
            return 3

    def set_num_tree4(self):
        number, ok = QInputDialog.getInt(
            self, "num_tree4", "Set num_tree4 number", 3, 1, 20, 1
        )
        if ok:
            self.num_tree4.setText(str(number))

    def get_num_tree4(self):
        try:
            if self.algorithm == "TGA":
                number = int(self.num_tree4.text())
                return number
        except Exception as e:
            return 3

    def set_lambda_(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set lambda_ number")
        dialog.setLabelText("Enter a value for lambda_:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        dialog.setDoubleRange(0.0, 1.0)  
        dialog.setDoubleDecimals(1)    
        dialog.setDoubleValue(0.5)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.lambda_.setText(str(number))

    def get_lambda_(self):
        try:
            if self.algorithm == "TGA":
                number = float(self.lambda_.text())
                return number
        except Exception as e:
            return 0.5

    def set_ST(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set ST number")
        dialog.setLabelText("Enter a value for ST:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        dialog.setDoubleRange(0.0, 1.0)  
        dialog.setDoubleDecimals(1)    
        dialog.setDoubleValue(0.1)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.ST.setText(str(number))

    def get_ST(self):
        try:
            if self.algorithm == "TSA":
                number = float(self.ST.text())
                return number
        except Exception as e:
            return 0.1

    def set_sl(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set sl number")
        dialog.setLabelText("Enter a value for sl:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        dialog.setDoubleRange(0.0, 2.0)  
        dialog.setDoubleDecimals(3)    
        dialog.setDoubleValue(0.035)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.sl.setText(str(number))

    def get_sl(self):
        try:
            if self.algorithm == "WSA":
                number = float(self.sl.text())
                return number
        except Exception as e:
            return 0.035

    def set_lambda_val(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set lambda_val number")
        dialog.setLabelText("Enter a value for lambda_val:")
        dialog.setInputMode(QInputDialog.DoubleInput)
        dialog.setDoubleRange(0.0, 1.0)  
        dialog.setDoubleDecimals(2)    
        dialog.setDoubleValue(0.75)     
        if dialog.exec_() == QInputDialog.Accepted:
            number = dialog.doubleValue()
            self.lambda_val.setText(str(number))

    def get_lambda_val(self):
        try:
            if self.algorithm == "WSA":
                number = float(self.lambda_val.text())
                return number
        except Exception as e:
            return 0.75

    @staticmethod
    def getValues(algorithm):
        dialog = QSwarmInput(algorithm)
        result = dialog.exec_()
        N = dialog.get_N()
        max_Iter = dialog.get_max_Iter()
        basic_model = dialog.get_basic_model()
        alpha = dialog.get_alpha()
        beta = dialog.get_beta()
        thres = dialog.get_thres()
        tau = dialog.get_tau()
        rho = dialog.get_rho()
        eta = dialog.get_eta()

        phi = dialog.get_phi()
        Alpha = dialog.get_Alpha()
        Mu = dialog.get_Mu()
        max_limit = dialog.get_max_limit()
        step_e = dialog.get_step_e()
        ratio = dialog.get_ratio()
        type = dialog.get_type()
        gamma = dialog.get_gamma()
        A_max = dialog.get_A_max()
        r0_max = dialog.get_r0_max()
        nBabysitter = dialog.get_nBabysitter()
        peep = dialog.get_peep()
        c = dialog.get_c()
        p = dialog.get_p()
        AP = dialog.get_AP()
        f1 = dialog.get_f1()
        Pa = dialog.get_Pa()
        CR = dialog.get_CR()
        F = dialog.get_F()
        P = dialog.get_P()
        Q = dialog.get_Q()
        M = dialog.get_M()
        f = dialog.get_f()
        l = dialog.get_l()
        beta0 = dialog.get_beta0()
        theta = dialog.get_theta()
        MR = dialog.get_MR()
        Tour_size = dialog.get_Tour_size()
        G0 = dialog.get_G0()
        PAR = dialog.get_PAR()
        HMCR = dialog.get_HMCR()
        bw = dialog.get_bw()
        num_gas = dialog.get_num_gas()
        K = dialog.get_K()
        L1 = dialog.get_L1()
        L2 = dialog.get_L2()
        L3 = dialog.get_L3()
        c1 = dialog.get_c1()
        c2 = dialog.get_c2()
        pi = dialog.get_pi()
        pr = dialog.get_pr()
        B = dialog.get_B()
        S = dialog.get_S()
        FADs = dialog.get_FADs()
        peri = dialog.get_peri()
        Smax = dialog.get_Smax()
        BAR = dialog.get_BAR()
        num_land1 = dialog.get_num_land1()
        b = dialog.get_b()
        w = dialog.get_w()
        Pmut = dialog.get_Pmut()
        z = dialog.get_z()
        T0 = dialog.get_T0()
        num_tree1 = dialog.get_num_tree1()
        num_tree2 = dialog.get_num_tree2()
        num_tree4 = dialog.get_num_tree4()
        lambda_ = dialog.get_lambda_()
        ST = dialog.get_ST()
        sl = dialog.get_sl()
        lambda_val = dialog.get_lambda_val()
        return (
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
            result == QDialog.Accepted,
        )


class QRandomForestInput(QDialog):
    def __init__(self):
        super(QRandomForestInput, self).__init__()
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QIcon("images/logo.ico"))
        self.setWindowTitle("Random Forest")
        self.setFont(QFont("Arial", 8))
        self.resize(430, 260)
        layout = QFormLayout(self)
        self.tree_number = MyLineEdit("100")
        self.tree_number.clicked.connect(self.setTreeNumber)
        self.cpu_number = MyLineEdit("1")
        self.cpu_number.clicked.connect(self.setCpuNumber)
        self.auto = QCheckBox("Auto optimization") 
        self.auto.stateChanged.connect(self.checkBoxStatus)
        self.start_tree_num = MyLineEdit("50")
        self.start_tree_num.setDisabled(True)
        self.end_tree_num = MyLineEdit("500")
        self.end_tree_num.setDisabled(True)
        self.step = MyLineEdit("50")
        self.step.setDisabled(True)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow("Tree number", self.tree_number)
        layout.addRow("Number of threads", self.cpu_number)
        layout.addRow(self.auto)
        layout.addRow("Trees range from", self.start_tree_num)
        layout.addRow("Trees range to", self.end_tree_num)
        layout.addRow("Tree step", self.step)
        layout.addRow(self.buttons)

    def setTreeNumber(self):
        number, ok = QInputDialog.getInt(
            self, "Tree number", "Set tree number", 100, 50, 3000, 50
        ) 
        if ok:
            self.tree_number.setText(str(number))

    def getTreeNumber(self):
        try:
            number = int(self.tree_number.text())
            return number
        except Exception as e:
            return 100

    def setCpuNumber(self):
        number, ok = QInputDialog.getInt(
            self, "Cpu number", "Set Cpu number", 1, 1, cpu_count(), 1
        )
        if ok:
            self.cpu_number.setText(str(number))

    def getCpuNumber(self):
        try:
            number = int(self.cpu_number.text())
            return number
        except Exception as e:
            return 1

    def checkBoxStatus(self):
        if self.auto.isChecked():
            self.tree_number.setDisabled(True)
            self.start_tree_num.setDisabled(False)
            self.end_tree_num.setDisabled(False)
            self.step.setDisabled(False)
        else:
            self.tree_number.setDisabled(False)
            self.start_tree_num.setDisabled(True)
            self.end_tree_num.setDisabled(True)
            self.step.setDisabled(True)

    def getTreeRange(self):
        try:
            start_number = int(self.start_tree_num.text())
            end_number = int(self.end_tree_num.text())
            step = int(self.step.text())
            return (start_number, end_number, step)
        except Exception as e:
            return (100, 1000, 100)

    def getState(self):
        return self.auto.isChecked()

    @staticmethod
    def getValues():
        dialog = QRandomForestInput()
        result = dialog.exec_()
        tree_number = dialog.getTreeNumber()
        tree_range = dialog.getTreeRange()
        cpu_number = dialog.getCpuNumber()
        state = dialog.getState()
        return tree_number, tree_range, cpu_number, state, result == QDialog.Accepted


class QSupportVectorMachineInput(QDialog):

    def __init__(self):
        super(QSupportVectorMachineInput, self).__init__()
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QIcon("images/logo.ico"))
        self.setWindowTitle("Support Vector Machine")
        self.setFont(QFont("Arial", 8))
        self.resize(430, 260)
        layout = QFormLayout(self)
        self.kernelComoBox = QComboBox()
        self.kernelComoBox.addItems(["linear", "rbf", "poly", "sigmoid"])
        self.kernelComoBox.setCurrentIndex(1)
        self.kernelComoBox.currentIndexChanged.connect(self.selectionChange)
        self.penaltyLineEdit = MyLineEdit("1.0")
        self.GLineEdit = MyLineEdit("auto")
        self.auto = QCheckBox("Auto optimization")
        self.auto.stateChanged.connect(self.checkBoxStatus)
        self.penaltyFromLineEdit = MyLineEdit("1.0")
        self.penaltyFromLineEdit.setDisabled(True)
        self.penaltyToLineEdit = MyLineEdit("15.0")
        self.penaltyToLineEdit.setDisabled(True)
        self.GFromLineEdit = MyLineEdit("-10.0")
        self.GFromLineEdit.setDisabled(True)
        self.GToLineEdit = MyLineEdit("5.0")
        self.GToLineEdit.setDisabled(True)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow("Kernel function:", self.kernelComoBox)
        layout.addRow("Penalty:", self.penaltyLineEdit)
        layout.addRow("Gamma:", self.GLineEdit)
        layout.addRow(self.auto)
        layout.addRow("Penalty from:", self.penaltyFromLineEdit)
        layout.addRow("Penalty to:", self.penaltyToLineEdit)
        layout.addRow("Gamma from (2^x):", self.GFromLineEdit)
        layout.addRow("Gamma to (2^x):", self.GToLineEdit)
        layout.addWidget(self.buttons)

    def selectionChange(self, i):
        svm_kernel = self.kernelComoBox.itemText(i)
        if svm_kernel == "linear" and not self.auto.isChecked():
            self.GLineEdit.setDisabled(True)
        else:
            self.GLineEdit.setDisabled(False)

    def checkBoxStatus(self):
        if self.auto.isChecked():
            self.penaltyLineEdit.setDisabled(True)
            self.GLineEdit.setDisabled(True)
            self.GFromLineEdit.setDisabled(False)
            self.GToLineEdit.setDisabled(False)
            self.penaltyFromLineEdit.setDisabled(False)
            self.penaltyToLineEdit.setDisabled(False)
        else:
            self.penaltyLineEdit.setDisabled(False)
            self.GLineEdit.setDisabled(False)
            self.GFromLineEdit.setDisabled(True)
            self.GToLineEdit.setDisabled(True)
            self.penaltyFromLineEdit.setDisabled(True)
            self.penaltyToLineEdit.setDisabled(True)

    def getKernel(self):
        return self.kernelComoBox.currentText()

    def getPenality(self):
        return (
            float(self.penaltyLineEdit.text())
            if self.penaltyLineEdit.text() != ""
            else 1.0
        )

    def getGamma(self):
        if self.GLineEdit.text() != "" and self.GLineEdit.text() != "auto":
            return float(self.GLineEdit.text())
        else:
            return "auto"

    def getAutoStatus(self):
        return self.auto.isChecked()

    def getPenalityRange(self):
        fromValue = (
            float(self.penaltyFromLineEdit.text())
            if self.penaltyFromLineEdit.text() != ""
            else 1.0
        )
        toValue = (
            float(self.penaltyToLineEdit.text())
            if self.penaltyToLineEdit.text() != ""
            else 15.0
        )
        return (fromValue, toValue)

    def getGammaRange(self):
        fromValue = (
            float(self.GFromLineEdit.text()) if self.GFromLineEdit.text() != "" else 1.0
        )
        toValue = (
            float(self.GToLineEdit.text()) if self.GToLineEdit.text() != "" else 15.0
        )
        return (fromValue, toValue)

    @staticmethod
    def getValues():
        try:
            dialog = QSupportVectorMachineInput()
            result = dialog.exec_()
            kernel = dialog.getKernel()
            penality = dialog.getPenality()
            gamma = dialog.getGamma()
            auto = dialog.getAutoStatus()
            penalityRange = dialog.getPenalityRange()
            gammaRange = dialog.getGammaRange()
            return (
                kernel,
                penality,
                gamma,
                auto,
                penalityRange,
                gammaRange,
                result == QDialog.Accepted,
            )
        except Exception as e:
            QMessageBox.critical(
                dialog,
                "Error",
                "Invalided parameter(s), use the default parameter!",
                QMessageBox.Ok | QMessageBox.No,
                QMessageBox.Ok,
            )
            return (
                "rbf",
                1.0,
                "auto",
                False,
                (1.0, 15.0),
                (-10.0, 5.0),
                result == QDialog.Accepted,
            )


class QMultiLayerPerceptronInput(QDialog):
    def __init__(self):
        super(QMultiLayerPerceptronInput, self).__init__()
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QIcon("images/logo.ico"))
        self.setWindowTitle("Multi-layer Perceptron")
        self.setFont(QFont("Arial", 8))
        self.resize(430, 200)
        layout = QFormLayout(self)
        self.layoutLineEdit = MyLineEdit("32;32")
        self.epochsLineEdit = MyLineEdit("200")
        self.activationComboBox = QComboBox()
        self.activationComboBox.addItems(["identity", "logistic", "tanh", "relu"])
        self.activationComboBox.setCurrentIndex(3)
        self.optimizerComboBox = QComboBox()
        self.optimizerComboBox.addItems(["lbfgs", "sgd", "adam"])
        self.optimizerComboBox.setCurrentIndex(2)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow("Hidden layer size:", self.layoutLineEdit)
        layout.addRow("Epochs:", self.epochsLineEdit)
        layout.addRow("Activation:", self.activationComboBox)
        layout.addRow("Optimizer:", self.optimizerComboBox)
        layout.addRow(self.buttons)

    def getLayer(self):
        return (
            self.layoutLineEdit.text() if self.layoutLineEdit.text() != "" else "32;32"
        )

    def getActivation(self):
        return self.activationComboBox.currentText()

    def getOptimizer(self):
        return self.optimizerComboBox.currentText()

    def getEpochs(self):
        return (
            int(self.epochsLineEdit.text()) if self.epochsLineEdit.text() != "" else 200
        )

    @staticmethod
    def getValues():
        dialog = QMultiLayerPerceptronInput()
        result = dialog.exec_()
        layer = dialog.getLayer()

        epochs = dialog.getEpochs()
        optimizer = dialog.getOptimizer()
        activation = dialog.getActivation()
        return layer, epochs, activation, optimizer, result == QDialog.Accepted


class QKNeighborsInput(QDialog):
    def __init__(self):
        super(QKNeighborsInput, self).__init__()
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QIcon("images/logo.ico"))
        self.setWindowTitle("KNN")
        self.setFont(QFont("Arial", 8))
        self.resize(430, 50)
        layout = QFormLayout(self)
        self.kValueLineEdit = MyLineEdit("3")
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow("Top K values:", self.kValueLineEdit)
        layout.addRow(self.buttons)

    def getTopKNNValue(self):
        return (
            int(self.kValueLineEdit.text()) if self.kValueLineEdit.text() != "" else 3
        )

    @staticmethod
    def getValues():
        dialog = QKNeighborsInput()
        result = dialog.exec_()
        topKValue = dialog.getTopKNNValue()
        return topKValue, result == QDialog.Accepted


class QLightGBMInput(QDialog):
    def __init__(self):
        super(QLightGBMInput, self).__init__()
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QIcon("images/logo.ico"))
        self.setWindowTitle("LightGBM")
        self.setFont(QFont("Arial", 8))
        self.resize(500, 260)
        layout = QFormLayout(self)
        self.boostingTypeComboBox = QComboBox()
        self.boostingTypeComboBox.addItems(["gbdt", "dart", "goss", "rf"])
        self.boostingTypeComboBox.setCurrentIndex(0)
        self.numLeavesLineEdit = MyLineEdit("31")
        self.maxDepthLineEdit = MyLineEdit("-1")
        self.learningRateLineEdit = MyLineEdit("0.1")
        self.cpu_number = MyLineEdit("1")
        self.cpu_number.clicked.connect(self.setCpuNumber)
        self.auto = QCheckBox("Auto optimization")
        self.auto.stateChanged.connect(self.checkBoxStatus)
        self.leavesRangeLineEdit = MyLineEdit("20:100:10")
        self.leavesRangeLineEdit.setDisabled(True)
        self.depthRangeLineEdit = MyLineEdit("15:55:10")
        self.depthRangeLineEdit.setDisabled(True)
        self.rateRangeLineEdit = MyLineEdit("0.01:0.15:0.02")
        self.rateRangeLineEdit.setDisabled(True)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow("Boosting type:", self.boostingTypeComboBox)
        layout.addRow("Number of leaves:", self.numLeavesLineEdit)
        layout.addRow("Max depth:", self.maxDepthLineEdit)
        layout.addRow("Learning rate:", self.learningRateLineEdit)
        layout.addRow("Number of threads:", self.cpu_number)
        layout.addRow(self.auto)
        layout.addRow("Leaves range (from:to:step)", self.leavesRangeLineEdit)
        layout.addRow("Depth range", self.depthRangeLineEdit)
        layout.addRow("Learning rate range:", self.rateRangeLineEdit)
        layout.addRow(self.buttons)

    def setCpuNumber(self):
        number, ok = QInputDialog.getInt(
            self, "Cpu number", "Set Cpu number", 1, 1, cpu_count(), 1
        )
        if ok:
            self.cpu_number.setText(str(number))

    def getCpuNumber(self):
        try:
            number = int(self.cpu_number.text())
            return number
        except Exception as e:
            return 1

    def checkBoxStatus(self):
        if self.auto.isChecked():
            self.numLeavesLineEdit.setDisabled(True)
            self.leavesRangeLineEdit.setDisabled(False)
            self.maxDepthLineEdit.setDisabled(True)
            self.depthRangeLineEdit.setDisabled(False)
            self.learningRateLineEdit.setDisabled(True)
            self.rateRangeLineEdit.setDisabled(False)
        else:
            self.numLeavesLineEdit.setDisabled(False)
            self.leavesRangeLineEdit.setDisabled(True)
            self.maxDepthLineEdit.setDisabled(False)
            self.depthRangeLineEdit.setDisabled(True)
            self.learningRateLineEdit.setDisabled(False)
            self.rateRangeLineEdit.setDisabled(True)

    def getState(self):
        return self.auto.isChecked()

    def getBoostingType(self):
        return self.boostingTypeComboBox.currentText()

    def getLeaves(self):
        return (
            int(self.numLeavesLineEdit.text())
            if self.numLeavesLineEdit.text() != ""
            else 31
        )

    def getMaxDepth(self):
        return (
            int(self.maxDepthLineEdit.text())
            if self.maxDepthLineEdit.text() != ""
            else -1
        )

    def getLearningRate(self):
        return (
            float(self.learningRateLineEdit.text())
            if self.learningRateLineEdit.text() != ""
            else 0.1
        )

    def getLeavesRange(self):
        return (
            tuple([int(i) for i in self.leavesRangeLineEdit.text().split(":")])
            if self.leavesRangeLineEdit.text() != ""
            else (20, 100, 10)
        )

    def getDepthRange(self):
        return (
            tuple([int(i) for i in self.depthRangeLineEdit.text().split(":")])
            if self.depthRangeLineEdit.text() != ""
            else (15, 55, 10)
        )

    def getRateRange(self):
        return (
            tuple([float(i) for i in self.rateRangeLineEdit.text().split(":")])
            if self.rateRangeLineEdit.text() != ""
            else (0.01, 0.15, 0.02)
        )

    @staticmethod
    def getValues():
        dialog = QLightGBMInput()
        result = dialog.exec_()
        threads = dialog.getCpuNumber()
        state = dialog.getState()
        type = dialog.getBoostingType()
        leaves = dialog.getLeaves()
        depth = dialog.getMaxDepth()
        rate = dialog.getLearningRate()
        leavesRange = dialog.getLeavesRange()
        depthRange = dialog.getDepthRange()
        rateRange = dialog.getRateRange()
        return (
            type,
            leaves,
            depth,
            rate,
            leavesRange,
            depthRange,
            rateRange,
            threads,
            state,
            result == QDialog.Accepted,
        )


class QXGBoostInput(QDialog):
    def __init__(self):
        super(QXGBoostInput, self).__init__()
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QIcon("images/logo.ico"))
        self.setWindowTitle("XGBoost")
        self.setFont(QFont("Arial", 8))
        self.resize(500, 260)
        layout = QFormLayout(self)
        self.boosterComboBox = QComboBox()
        self.boosterComboBox.addItems(["gbtree", "gblinear"])
        self.boosterComboBox.setCurrentIndex(0)
        self.cpu_number = MyLineEdit("1")
        self.cpu_number.clicked.connect(self.setCpuNumber)
        self.maxDepthLineEdit = MyLineEdit("6")
        self.n_estimatorLineEdit = QLineEdit("100")
        self.learningRateLineEdit = MyLineEdit("0.3")
        self.colsample_bytreeLineEdit = MyLineEdit("0.8")
        self.auto = QCheckBox("Auto optimization")
        self.auto.stateChanged.connect(self.checkBoxStatus)
        self.depthRangeLineEdit = MyLineEdit("3:10:1")
        self.depthRangeLineEdit.setDisabled(True)
        self.rateRangeLineEdit = MyLineEdit("0.01:0.3:0.05")
        self.rateRangeLineEdit.setDisabled(True)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow("Booster:", self.boosterComboBox)
        layout.addRow("Number of threads:", self.cpu_number)
        layout.addRow("Max depth (3~10):", self.maxDepthLineEdit)
        layout.addRow("Learnning rate:", self.learningRateLineEdit)
        # layout.addRow('n_estimator:', self.n_estimatorLineEdit)
        layout.addRow("colsample_bytree", self.colsample_bytreeLineEdit)
        layout.addRow(self.auto)
        layout.addRow("Depth range", self.depthRangeLineEdit)
        layout.addRow("Learning rate range:", self.rateRangeLineEdit)
        layout.addRow(self.buttons)

    def getBooster(self):
        return self.boosterComboBox.currentText()

    def setCpuNumber(self):
        number, ok = QInputDialog.getInt(
            self, "Cpu number", "Set Cpu number", 1, 1, cpu_count(), 1
        )
        if ok:
            self.cpu_number.setText(str(number))

    def getCpuNumber(self):
        try:
            number = int(self.cpu_number.text())
            return number
        except Exception as e:
            return 1

    def checkBoxStatus(self):
        if self.auto.isChecked():
            self.rateRangeLineEdit.setDisabled(False)
            self.depthRangeLineEdit.setDisabled(False)
            self.maxDepthLineEdit.setDisabled(True)
            self.learningRateLineEdit.setDisabled(True)
        else:
            self.rateRangeLineEdit.setDisabled(True)
            self.depthRangeLineEdit.setDisabled(True)
            self.maxDepthLineEdit.setDisabled(False)
            self.learningRateLineEdit.setDisabled(False)

    def getMaxDepth(self):
        return (
            int(self.maxDepthLineEdit.text())
            if self.maxDepthLineEdit.text() != ""
            else -1
        )

    def getLearningRate(self):
        return (
            float(self.learningRateLineEdit.text())
            if self.learningRateLineEdit.text() != ""
            else 0.1
        )

    def getNEstimator(self):
        return (
            int(self.n_estimatorLineEdit.text())
            if self.n_estimatorLineEdit.text() != ""
            else 100
        )

    def getColsample(self):
        return (
            float(self.colsample_bytreeLineEdit.text())
            if self.colsample_bytreeLineEdit.text() != ""
            else 0.8
        )

    def getState(self):
        return self.auto.isChecked()

    def getLearningRate(self):
        return (
            float(self.learningRateLineEdit.text())
            if self.learningRateLineEdit.text() != ""
            else 0.1
        )

    def getRateRange(self):
        return (
            tuple([float(i) for i in self.rateRangeLineEdit.text().split(":")])
            if self.rateRangeLineEdit.text() != ""
            else (0.01, 0.15, 0.02)
        )

    def getDepthRange(self):
        return (
            tuple([int(i) for i in self.depthRangeLineEdit.text().split(":")])
            if self.depthRangeLineEdit.text() != ""
            else (15, 55, 10)
        )

    @staticmethod
    def getValues():
        dialog = QXGBoostInput()
        result = dialog.exec_()
        state = dialog.getState()
        threads = dialog.getCpuNumber()
        booster = dialog.getBooster()
        maxdepth = dialog.getMaxDepth()
        rate = dialog.getLearningRate()
        estimator = dialog.getNEstimator()
        colsample = dialog.getColsample()
        depthRange = dialog.getDepthRange()
        rateRange = dialog.getRateRange()
        return (
            booster,
            maxdepth,
            rate,
            estimator,
            colsample,
            depthRange,
            rateRange,
            threads,
            state,
            result == QDialog.Accepted,
        )


class QBaggingInput(QDialog):
    def __init__(self):
        super(QBaggingInput, self).__init__()
        self.initUI()

    def initUI(self):
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowIcon(QIcon("images/logo.ico"))
        self.setWindowTitle("Bagging")
        self.setFont(QFont("Arial", 8))
        self.resize(500, 100)
        layout = QFormLayout(self)
        self.n_estimators = MyLineEdit("10")
        self.cpu_number = MyLineEdit("1")
        self.cpu_number.clicked.connect(self.setCpuNumber)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow("n_estimators", self.n_estimators)
        layout.addRow("CPU number", self.cpu_number)
        layout.addRow(self.buttons)

    def setCpuNumber(self):
        number, ok = QInputDialog.getInt(
            self, "Cpu number", "Set Cpu number", 1, 1, cpu_count(), 1
        )
        if ok:
            self.cpu_number.setText(str(number))

    def getCpuNumber(self):
        try:
            number = int(self.cpu_number.text())
            return number
        except Exception as e:
            return 1

    def getEstimator(self):
        try:
            if self.n_estimators.text() != "":
                n_estimator = int(self.n_estimators.text())
                if 0 < n_estimator <= 1000:
                    return int(self.n_estimators.text())
                else:
                    return 10
        except Exception as e:
            return 10

    @staticmethod
    def getValues():
        dialog = QBaggingInput()
        result = dialog.exec_()
        n_estimators = dialog.getEstimator()
        threads = dialog.getCpuNumber()
        return n_estimators, threads, result == QDialog.Accepted


class QStaticsInput(QDialog):
    def __init__(self):
        super(QStaticsInput, self).__init__()
        self.initUI()

    def initUI(self):
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowIcon(QIcon("images/logo.ico"))
        self.setWindowTitle("SIFS Statics")
        self.setFont(QFont("Arial", 8))
        self.resize(300, 100)
        layout = QFormLayout(self)
        self.method = QComboBox()
        self.method.addItems(["bootstrap"])
        self.bootstrap_num = MyLineEdit("500")
        self.bootstrap_num.clicked.connect(self.setBootstrapNum)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow("Method:", self.method)
        layout.addRow("Bootstrap number:", self.bootstrap_num)
        layout.addWidget(self.buttons)

    def setBootstrapNum(self):
        number, ok = QInputDialog.getInt(
            self, "Bootstrap number", "SBootstrap number", 500, 100, 2000, 100
        )
        if ok:
            self.bootstrap_num.setText(str(number))

    def getMethod(self):
        return self.method.currentText()

    def getBootstrapNum(self):
        return int(self.bootstrap_num.text())

    @staticmethod
    def getValues():
        dialog = QStaticsInput()
        result = dialog.exec_()
        method = dialog.getMethod()
        bootstrap_n = dialog.getBootstrapNum()
        return method, bootstrap_n, result == QDialog.Accepted


class QNetInput_1(QDialog):
    def __init__(self, dim):
        super(QNetInput_1, self).__init__()
        self.dim = dim
        self.initUI()

    def initUI(self):
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowIcon(QIcon("images/logo.ico"))
        self.setWindowTitle("SIFS Deeplearning")
        self.setFont(QFont("Arial", 8))
        self.resize(600, 200)
        layout = QFormLayout(self)
        self.in_channels = MyLineEdit("1")
        self.in_channels.clicked.connect(self.set_in_channel)
        self.in_channels.textChanged.connect(self.check_input_channel)
        self.in_length = MyLineEdit(str(self.dim))
        self.in_length.clicked.connect(self.set_in_length)
        self.in_length.textChanged.connect(self.check_input_length)
        label = QLabel(
            "Note: Input channels X Input length = Feature number (i.e. %s)" % self.dim
        )
        label.setAlignment(Qt.AlignLeft)
        label.setFont(QFont("Arial", 7))
        self.out_channels = MyLineEdit("64")
        self.padding = MyLineEdit("3")
        self.kernel_size = MyLineEdit("5")
        self.drop_out = MyLineEdit("0.5")
        self.learning_rate = MyLineEdit("0.001")
        self.epochs = MyLineEdit("1000")
        self.early_stopping = MyLineEdit("100")
        self.batch_size = MyLineEdit("64")
        self.fc_size = MyLineEdit("64")
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow("Input channels (integer):", self.in_channels)
        layout.addRow("Input length (integer):", self.in_length)
        layout.addWidget(label)
        layout.addRow("Output channels (integer):", self.out_channels)
        layout.addRow("Padding (integer):", self.padding)
        layout.addRow("Kernel size (integer):", self.kernel_size)
        layout.addRow("FC layer size (integer):", self.fc_size)
        layout.addRow("Dropout rate (0~1):", self.drop_out)
        layout.addRow("Learning rate (0~1):", self.learning_rate)
        layout.addRow("Epochs (integer):", self.epochs)
        layout.addRow("Early stopping (integer):", self.early_stopping)
        layout.addRow("Batch size (integer):", self.batch_size)
        layout.addWidget(self.buttons)

    def set_in_channel(self):
        number, ok = QInputDialog.getInt(
            self, "Input channels", "Input channels", 1, 1, 100, 1
        )
        if ok:
            self.in_channels.setText(str(number))

    def set_in_length(self):
        number, ok = QInputDialog.getInt(
            self, "Input length", "Input length", self.dim, 1, self.dim, 1
        )
        if ok:
            self.in_length.setText(str(number))

    def check_input_channel(self, text):
        try:
            if self.dim % int(text) == 0:
                self.in_length.setText(str(int(self.dim / int(text))))
            else:
                self.in_channels.setText("1")
                self.in_length.setText(str(int(self.dim)))
        except Exception as e:
            self.in_channels.setText("1")
            QMessageBox.critical(
                self,
                "Error",
                "Invalid parameter value.",
                QMessageBox.Ok | QMessageBox.No,
                QMessageBox.Ok,
            )

    def check_input_length(self, text):
        try:
            if self.dim % int(text) == 0:
                self.in_channels.setText(str(int(self.dim / int(text))))
            else:
                self.in_channels.setText("1")
                self.in_length.setText(str(int(self.dim)))
        except Exception as e:
            self.in_length.setText(str(int(self.dim)))
            QMessageBox.critical(
                self,
                "Error",
                "Invalid parameter value.",
                QMessageBox.Ok | QMessageBox.No,
                QMessageBox.Ok,
            )

    def getChannels(self):
        try:
            if self.in_channels != "":
                return int(self.in_channels.text())
            else:
                return 1
        except Exception as e:
            return 1

    def getLength(self):
        try:
            if self.in_length != "":
                return int(self.in_length.text())
            else:
                return self.dim
        except Exception as e:
            return self.dim

    def getOutputChannel(self):
        try:
            if self.out_channels.text() != "":
                channel = int(self.out_channels.text())
                if 0 < channel <= 1024:
                    return channel
                else:
                    return 64
            else:
                return 64
        except Exception as e:
            return 64

    def getPadding(self):
        try:
            if self.padding.text() != "":
                if 0 < int(self.padding.text()) <= 64:
                    return int(self.padding.text())
                else:
                    return 3
            else:
                return 3
        except Exception as e:
            return 3

    def getKernelSize(self):
        try:
            if self.kernel_size.text() != "":
                if 0 < int(self.kernel_size.text()):
                    return int(self.kernel_size.text())
                else:
                    return 5
            else:
                return 5
        except Exception as e:
            return 5

    def getLearningRate(self):
        try:
            if self.learning_rate.text() != "":
                lr = float(self.learning_rate.text())
                if 0 < lr < 1:
                    return lr
                else:
                    return 0.001
            else:
                return 0.001
        except Exception as e:
            return 0.001

    def getDroprate(self):
        try:
            if self.drop_out.text() != "":
                dropout = float(self.drop_out.text())
                if 0 < dropout < 1:
                    return dropout
                else:
                    return 0.5
            else:
                return 0.5
        except Exception as e:
            return 0.5

    def getEpochs(self):
        try:
            if self.epochs.text() != "":
                if int(self.epochs.text()) > 0:
                    return int(self.epochs.text())
                else:
                    return 1000
            else:
                return 1000
        except Exception as e:
            return 1000

    def getEarlyStopping(self):
        try:
            if self.early_stopping.text() != "":
                if int(self.early_stopping.text()) > 0:
                    return int(self.early_stopping.text())
                else:
                    return 100
            else:
                return 100
        except Exception as e:
            return 100

    def getBatchSize(self):
        try:
            if self.batch_size.text() != "":
                if int(self.batch_size.text()) > 0:
                    return int(self.batch_size.text())
                else:
                    return 64
            else:
                return 64
        except Exception as e:
            return 64

    def getFCSize(self):
        try:
            if self.fc_size.text() != "":
                if int(self.fc_size.text()) > 0:
                    return int(self.fc_size.text())
                else:
                    return 64
            else:
                return 64
        except Exception as e:
            return 64

    @staticmethod
    def getValues(dim):
        try:
            dialog = QNetInput_1(dim)
            result = dialog.exec_()
            input_channel = dialog.getChannels()
            input_length = dialog.getLength()
            output_channel = dialog.getOutputChannel()
            padding = dialog.getPadding()
            kernel_size = dialog.getKernelSize()
            dropout = dialog.getDroprate()
            learning_rate = dialog.getLearningRate()
            epochs = dialog.getEpochs()
            early_stopping = dialog.getEarlyStopping()
            batch_size = dialog.getBatchSize()
            fc_size = dialog.getFCSize()
            return (
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
                result == QDialog.Accepted,
            )
        except Exception as e:
            return 1, dim, 64, 3, 5, 0.5, 0.001, 1000, 100, 64, 64, False


class QNetInput_2(QDialog):
    def __init__(self, dim):
        super(QNetInput_2, self).__init__()
        self.dim = dim
        self.initUI()

    def initUI(self):
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowIcon(QIcon("images/logo.ico"))
        self.setWindowTitle("SIFS Deeplearning")
        self.setFont(QFont("Arial", 8))
        self.resize(600, 200)
        layout = QFormLayout(self)
        self.in_channels = MyLineEdit("1")
        self.in_channels.clicked.connect(self.set_in_channel)
        self.in_channels.textChanged.connect(self.check_input_channel)
        self.in_length = MyLineEdit(str(self.dim))
        self.in_length.clicked.connect(self.set_in_length)
        self.in_length.textChanged.connect(self.check_input_length)
        label = QLabel(
            "Note: Input channels X Input length = Feature number (i.e. %s)" % self.dim
        )
        label.setAlignment(Qt.AlignLeft)
        label.setFont(QFont("Arial", 7))
        self.hidden_size = MyLineEdit("32")
        self.num_layers = MyLineEdit("1")
        self.fc_size = MyLineEdit("64")
        self.drop_out = MyLineEdit("0.5")
        self.learning_rate = MyLineEdit("0.001")
        self.epochs = MyLineEdit("1000")
        self.early_stopping = MyLineEdit("100")
        self.batch_size = MyLineEdit("64")
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow("Input channels (integer):", self.in_channels)
        layout.addRow("Input length (integer):", self.in_length)
        layout.addWidget(label)
        layout.addRow("Hidden size (integer):", self.hidden_size)
        layout.addRow("Number of recurrent layers (integer):", self.num_layers)
        layout.addRow("FC layer size (integer):", self.fc_size)
        layout.addRow("Dropout rate (0~1):", self.drop_out)
        layout.addRow("Learning rate (0~1):", self.learning_rate)
        layout.addRow("Epochs (integer):", self.epochs)
        layout.addRow("Early stopping (integer):", self.early_stopping)
        layout.addRow("Batch size (integer):", self.batch_size)
        layout.addWidget(self.buttons)

    def set_in_channel(self):
        number, ok = QInputDialog.getInt(
            self, "Input channels", "Input channels", 1, 1, 100, 1
        )
        if ok:
            self.in_channels.setText(str(number))

    def set_in_length(self):
        number, ok = QInputDialog.getInt(
            self, "Input length", "Input length", self.dim, 1, self.dim, 1
        )
        if ok:
            self.in_length.setText(str(number))

    def check_input_channel(self, text):
        try:
            if self.dim % int(text) == 0:
                self.in_length.setText(str(int(self.dim / int(text))))
            else:
                self.in_channels.setText("1")
                self.in_length.setText(str(int(self.dim)))
        except Exception as e:
            self.in_channels.setText("1")
            QMessageBox.critical(
                self,
                "Error",
                "Invalid parameter value.",
                QMessageBox.Ok | QMessageBox.No,
                QMessageBox.Ok,
            )

    def check_input_length(self, text):
        try:
            if self.dim % int(text) == 0:
                self.in_channels.setText(str(int(self.dim / int(text))))
            else:
                self.in_channels.setText("1")
                self.in_length.setText(str(int(self.dim)))
        except Exception as e:
            self.in_length.setText(str(int(self.dim)))
            QMessageBox.critical(
                self,
                "Error",
                "Invalid parameter value.",
                QMessageBox.Ok | QMessageBox.No,
                QMessageBox.Ok,
            )

    def getChannels(self):
        try:
            if self.in_channels != "":
                if int(self.in_channels.text()) > 0:
                    return int(self.in_channels.text())
                else:
                    return 1
            else:
                return 1
        except Exception as e:
            return 1

    def getLength(self):
        try:
            if self.in_length != "":
                return int(self.in_length.text())
            else:
                return self.dim
        except Exception as e:
            return self.dim

    def getHiddenSize(self):
        try:
            if self.hidden_size != "":
                if 0 < int(self.hidden_size.text()) <= 512:
                    return int(self.hidden_size.text())
                else:
                    return 32
            else:
                return 32
        except Exception as e:
            return 32

    def getRnnLayers(self):
        try:
            if self.num_layers.text() != "":
                if 0 < int(self.num_layers.text()) <= 32:
                    return int(self.num_layers.text())
                else:
                    return 1
            else:
                return 1
        except Exception as e:
            return 1

    def getLearningRate(self):
        try:
            if self.learning_rate.text() != "":
                lr = float(self.learning_rate.text())
                if 0 < lr < 1:
                    return lr
                else:
                    return 0.001
            else:
                return 0.001
        except Exception as e:
            return 0.001

    def getDroprate(self):
        try:
            if self.drop_out.text() != "":
                dropout = float(self.drop_out.text())
                if 0 < dropout < 1:
                    return dropout
                else:
                    return 0.5
            else:
                return 0.5
        except Exception as e:
            return 0.5

    def getEpochs(self):
        try:
            if self.epochs.text() != "":
                if int(self.epochs.text()) > 0:
                    return int(self.epochs.text())
                else:
                    return 1000
            else:
                return 1000
        except Exception as e:
            return 1000

    def getEarlyStopping(self):
        try:
            if self.early_stopping.text() != "":
                if int(self.early_stopping.text()) > 0:
                    return int(self.early_stopping.text())
                else:
                    return 100
            else:
                return 100
        except Exception as e:
            return 100

    def getBatchSize(self):
        try:
            if self.batch_size.text() != "":
                if int(self.batch_size.text()) > 0:
                    return int(self.batch_size.text())
                else:
                    return 64
            else:
                return 64
        except Exception as e:
            return 64

    def getFCSize(self):
        try:
            if self.fc_size.text() != "":
                if int(self.fc_size.text()) > 0:
                    return int(self.fc_size.text())
                else:
                    return 64
            else:
                return 64
        except Exception as e:
            return 64

    @staticmethod
    def getValues(dim):
        dialog = QNetInput_2(dim)
        result = dialog.exec_()
        input_channel = dialog.getChannels()
        input_length = dialog.getLength()
        hidden_size = dialog.getHiddenSize()
        num_layers = dialog.getRnnLayers()
        dropout = dialog.getDroprate()
        learning_rate = dialog.getLearningRate()
        epochs = dialog.getEpochs()
        early_stopping = dialog.getEarlyStopping()
        batch_size = dialog.getBatchSize()
        fc_size = dialog.getFCSize()
        return (
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
            result == QDialog.Accepted,
        )


class QNetInput_4(QDialog):
    def __init__(self, dim):
        super(QNetInput_4, self).__init__()
        self.dim = dim
        self.initUI()

    def initUI(self):
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowIcon(QIcon("images/logo.ico"))
        self.setWindowTitle("SIFS Deeplearning")
        self.setFont(QFont("Arial", 8))
        self.resize(600, 200)
        layout = QFormLayout(self)
        self.in_channels = MyLineEdit("1")
        self.in_channels.clicked.connect(self.set_in_channel)
        self.in_channels.textChanged.connect(self.check_input_channel)
        self.in_length = MyLineEdit(str(self.dim))
        self.in_length.clicked.connect(self.set_in_length)
        self.in_length.textChanged.connect(self.check_input_length)
        label = QLabel(
            "Note: Input channels X Input length = Feature number (i.e. %s)" % self.dim
        )
        label.setAlignment(Qt.AlignLeft)
        label.setFont(QFont("Arial", 7))
        self.drop_out = MyLineEdit("0.5")
        self.learning_rate = MyLineEdit("0.001")
        self.epochs = MyLineEdit("1000")
        self.early_stopping = MyLineEdit("100")
        self.batch_size = MyLineEdit("64")
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow("Input channels (integer):", self.in_channels)
        layout.addRow("Input length (integer):", self.in_length)
        layout.addWidget(label)
        layout.addRow("Dropout rate (0~1):", self.drop_out)
        layout.addRow("Learning rate (0~1):", self.learning_rate)
        layout.addRow("Epochs (integer):", self.epochs)
        layout.addRow("Early stopping (integer):", self.early_stopping)
        layout.addRow("Batch size (integer):", self.batch_size)
        layout.addWidget(self.buttons)

    def set_in_channel(self):
        number, ok = QInputDialog.getInt(
            self, "Input channels", "Input channels", 1, 1, 100, 1
        )
        if ok:
            self.in_channels.setText(str(number))

    def set_in_length(self):
        number, ok = QInputDialog.getInt(
            self, "Input length", "Input length", self.dim, 1, self.dim, 1
        )
        if ok:
            self.in_length.setText(str(number))

    def check_input_channel(self, text):
        try:
            if self.dim % int(text) == 0:
                self.in_length.setText(str(int(self.dim / int(text))))
            else:
                self.in_channels.setText("1")
                self.in_length.setText(str(int(self.dim)))
        except Exception as e:
            self.in_channels.setText("1")
            QMessageBox.critical(
                self,
                "Error",
                "Invalid parameter value.",
                QMessageBox.Ok | QMessageBox.No,
                QMessageBox.Ok,
            )

    def check_input_length(self, text):
        try:
            if self.dim % int(text) == 0:
                self.in_channels.setText(str(int(self.dim / int(text))))
            else:
                self.in_channels.setText("1")
                self.in_length.setText(str(int(self.dim)))
        except Exception as e:
            self.in_length.setText(str(int(self.dim)))
            QMessageBox.critical(
                self,
                "Error",
                "Invalid parameter value.",
                QMessageBox.Ok | QMessageBox.No,
                QMessageBox.Ok,
            )

    def getChannels(self):
        try:
            if self.in_channels != "":
                return int(self.in_channels.text())
            else:
                return 1
        except Exception as e:
            return 1

    def getLength(self):
        try:
            if self.in_length != "":
                return int(self.in_length.text())
            else:
                return self.dim
        except Exception as e:
            return self.dim

    def getLearningRate(self):
        try:
            if self.learning_rate.text() != "":
                lr = float(self.learning_rate.text())
                if 0 < lr < 1:
                    return lr
                else:
                    return 0.001
            else:
                return 0.001
        except Exception as e:
            return 0.001

    def getDroprate(self):
        try:
            if self.drop_out.text() != "":
                dropout = float(self.drop_out.text())
                if 0 < dropout < 1:
                    return dropout
                else:
                    return 0.5
            else:
                return 0.5
        except Exception as e:
            return 0.5

    def getEpochs(self):
        try:
            if self.epochs.text() != "":
                if int(self.epochs.text()) > 0:
                    return int(self.epochs.text())
                else:
                    return 1000
            else:
                return 1000
        except Exception as e:
            return 1000

    def getEarlyStopping(self):
        try:
            if self.early_stopping.text() != "":
                if int(self.early_stopping.text()) > 0:
                    return int(self.early_stopping.text())
                else:
                    return 100
            else:
                return 100
        except Exception as e:
            return 100

    def getBatchSize(self):
        try:
            if self.batch_size.text() != "":
                if int(self.batch_size.text()) > 0:
                    return int(self.batch_size.text())
                else:
                    return 64
            else:
                return 64
        except Exception as e:
            return 64

    @staticmethod
    def getValues(dim):
        dialog = QNetInput_4(dim)
        result = dialog.exec_()
        input_channel = dialog.getChannels()
        input_length = dialog.getLength()
        dropout = dialog.getDroprate()
        learning_rate = dialog.getLearningRate()
        epochs = dialog.getEpochs()
        early_stopping = dialog.getEarlyStopping()
        batch_size = dialog.getBatchSize()
        return (
            input_channel,
            input_length,
            dropout,
            learning_rate,
            epochs,
            early_stopping,
            batch_size,
            result == QDialog.Accepted,
        )


class QNetInput_5(QDialog):
    def __init__(self, dim):
        super(QNetInput_5, self).__init__()
        self.dim = dim
        self.initUI()

    def initUI(self):
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowIcon(QIcon("images/logo.ico"))
        self.setWindowTitle("SIFS Deeplearning")
        self.setFont(QFont("Arial", 8))
        self.resize(600, 200)
        layout = QFormLayout(self)
        self.in_channels = MyLineEdit("1")
        self.in_channels.clicked.connect(self.set_in_channel)
        self.in_channels.textChanged.connect(self.check_input_channel)
        self.in_length = MyLineEdit(str(self.dim))
        self.in_length.clicked.connect(self.set_in_length)
        self.in_length.textChanged.connect(self.check_input_length)
        label = QLabel(
            "Note: Input channels X Input length = Feature number (i.e. %s)" % self.dim
        )
        label.setAlignment(Qt.AlignLeft)
        label.setFont(QFont("Arial", 7))
        self.learning_rate = MyLineEdit("0.001")
        self.epochs = MyLineEdit("1000")
        self.early_stopping = MyLineEdit("100")
        self.batch_size = MyLineEdit("64")
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow("Input channels (integer):", self.in_channels)
        layout.addRow("Input length (integer):", self.in_length)
        layout.addWidget(label)
        layout.addRow("Learning rate (0~1):", self.learning_rate)
        layout.addRow("Epochs (integer):", self.epochs)
        layout.addRow("Early stopping (integer):", self.early_stopping)
        layout.addRow("Batch size (integer):", self.batch_size)
        layout.addWidget(self.buttons)

    def set_in_channel(self):
        number, ok = QInputDialog.getInt(
            self, "Input channels", "Input channels", 1, 1, 100, 1
        )
        if ok:
            self.in_channels.setText(str(number))

    def set_in_length(self):
        number, ok = QInputDialog.getInt(
            self, "Input length", "Input length", self.dim, 1, self.dim, 1
        )
        if ok:
            self.in_length.setText(str(number))

    def check_input_channel(self, text):
        try:
            if self.dim % int(text) == 0:
                self.in_length.setText(str(int(self.dim / int(text))))
            else:
                self.in_channels.setText("1")
                self.in_length.setText(str(int(self.dim)))
        except Exception as e:
            self.in_channels.setText("1")
            QMessageBox.critical(
                self,
                "Error",
                "Invalid parameter value.",
                QMessageBox.Ok | QMessageBox.No,
                QMessageBox.Ok,
            )

    def check_input_length(self, text):
        try:
            if self.dim % int(text) == 0:
                self.in_channels.setText(str(int(self.dim / int(text))))
            else:
                self.in_channels.setText("1")
                self.in_length.setText(str(int(self.dim)))
        except Exception as e:
            self.in_length.setText(str(int(self.dim)))
            QMessageBox.critical(
                self,
                "Error",
                "Invalid parameter value.",
                QMessageBox.Ok | QMessageBox.No,
                QMessageBox.Ok,
            )

    def getChannels(self):
        if self.in_channels != "":
            return int(self.in_channels.text())
        else:
            return 1

    def getLength(self):
        if self.in_length != "":
            return int(self.in_length.text())
        else:
            return self.dim

    def getLearningRate(self):
        try:
            if self.learning_rate.text() != "":
                lr = float(self.learning_rate.text())
                if 0 < lr < 1:
                    return lr
                else:
                    return 0.001
            else:
                return 0.001
        except Exception as e:
            return 0.001

    def getEpochs(self):
        try:
            if self.epochs.text() != "":
                if int(self.epochs.text()) > 0:
                    return int(self.epochs.text())
                else:
                    return 1000
            else:
                return 1000
        except Exception as e:
            return 1000

    def getEarlyStopping(self):
        try:
            if self.early_stopping.text() != "":
                if int(self.early_stopping.text()) > 0:
                    return int(self.early_stopping.text())
                else:
                    return 100
            else:
                return 100
        except Exception as e:
            return 100

    def getBatchSize(self):
        try:
            if self.batch_size.text() != "":
                if int(self.batch_size.text()) > 0:
                    return int(self.batch_size.text())
                else:
                    return 64
            else:
                return 64
        except Exception as e:
            return 64

    @staticmethod
    def getValues(dim):
        dialog = QNetInput_5(dim)
        result = dialog.exec_()
        input_channel = dialog.getChannels()
        input_length = dialog.getLength()
        learning_rate = dialog.getLearningRate()
        epochs = dialog.getEpochs()
        early_stopping = dialog.getEarlyStopping()
        batch_size = dialog.getBatchSize()
        return (
            input_channel,
            input_length,
            learning_rate,
            epochs,
            early_stopping,
            batch_size,
            result == QDialog.Accepted,
        )


class QNetInput_6(QDialog):
    def __init__(self, dim):
        super(QNetInput_6, self).__init__()
        self.dim = dim
        self.initUI()

    def initUI(self):
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowIcon(QIcon("images/logo.ico"))
        self.setWindowTitle("SIFS Deeplearning")
        self.setFont(QFont("Arial", 8))
        self.resize(600, 200)
        layout = QFormLayout(self)
        self.in_channels = MyLineEdit(str(self.dim))
        self.in_channels.setEnabled(False)
        self.drop_out = MyLineEdit("0.5")
        self.learning_rate = MyLineEdit("0.001")
        self.epochs = MyLineEdit("1000")
        self.early_stopping = MyLineEdit("100")
        self.batch_size = MyLineEdit("64")
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow("Input size (integer):", self.in_channels)
        layout.addRow("Dropout rate (0~1):", self.drop_out)
        layout.addRow("Learning rate (0~1):", self.learning_rate)
        layout.addRow("Epochs (integer):", self.epochs)
        layout.addRow("Early stopping (integer):", self.early_stopping)
        layout.addRow("Batch size (integer):", self.batch_size)
        layout.addWidget(self.buttons)

    def getChannels(self):
        if self.in_channels != "":
            return int(self.in_channels.text())
        else:
            return 1

    def getLearningRate(self):
        try:
            if self.learning_rate.text() != "":
                lr = float(self.learning_rate.text())
                if 0 < lr < 1:
                    return lr
                else:
                    return 0.001
            else:
                return 0.001
        except Exception as e:
            return 0.001

    def getDroprate(self):
        try:
            if self.drop_out.text() != "":
                dropout = float(self.drop_out.text())
                if 0 < dropout < 1:
                    return dropout
                else:
                    return 0.5
            else:
                return 0.5
        except Exception as e:
            return 0.5

    def getEpochs(self):
        try:
            if self.epochs.text() != "":
                if int(self.epochs.text()) > 0:
                    return int(self.epochs.text())
                else:
                    return 1000
            else:
                return 1000
        except Exception as e:
            return 1000

    def getEarlyStopping(self):
        try:
            if self.early_stopping.text() != "":
                if int(self.early_stopping.text()) > 0:
                    return int(self.early_stopping.text())
                else:
                    return 100
            else:
                return 100
        except Exception as e:
            return 100

    def getBatchSize(self):
        try:
            if self.batch_size.text() != "":
                if int(self.batch_size.text()) > 0:
                    return int(self.batch_size.text())
                else:
                    return 64
            else:
                return 64
        except Exception as e:
            return 64

    @staticmethod
    def getValues(dim):
        dialog = QNetInput_6(dim)
        result = dialog.exec_()
        input_channel = dialog.getChannels()
        dropout = dialog.getDroprate()
        learning_rate = dialog.getLearningRate()
        epochs = dialog.getEpochs()
        early_stopping = dialog.getEarlyStopping()
        batch_size = dialog.getBatchSize()
        return (
            input_channel,
            dropout,
            learning_rate,
            epochs,
            early_stopping,
            batch_size,
            result == QDialog.Accepted,
        )

class QPlotInput(QDialog):
    def __init__(self, curve="ROC"):
        super(QPlotInput, self).__init__()
        self.initUI()
        self.curve = curve
        self.auc = None
        self.dot = None
        self.color = "#000000"
        self.lineWidth = 1
        self.lineStyle = "solid"
        self.raw_data = None

    def initUI(self):
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setFont(QFont("Arial", 8))
        self.setWindowIcon(QIcon("images/logo.ico"))
        self.setWindowTitle("SIFS PlotCurve")
        self.resize(500, 200)
        layout = QFormLayout(self)
        self.file = MyLineEdit()
        self.file.clicked.connect(self.getFileName)
        self.colorLineEdit = MyLineEdit("#000000")
        self.colorLineEdit.clicked.connect(self.setColor)
        self.widthLineEdit = MyLineEdit("1")
        self.widthLineEdit.clicked.connect(self.setWidth)
        self.styleLineEdit = QComboBox()
        self.styleLineEdit.addItems(["solid", "dashed", "dashdot", "dotted"])
        self.styleLineEdit.currentIndexChanged.connect(self.getLineStyle)
        self.legendLineEdit = MyLineEdit()
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow("Data file:", self.file)
        layout.addRow("Set color:", self.colorLineEdit)
        layout.addRow("Set line Width:", self.widthLineEdit)
        layout.addRow("Set line style:", self.styleLineEdit)
        layout.addRow("Set legend prefix:", self.legendLineEdit)
        layout.addWidget(self.buttons)

    def getFileName(self):
        file, ok = QFileDialog.getOpenFileName(
            self, "Open", "./data", "CSV Files (*.csv);;TSV Files (*.tsv)"
        )
        if ok:
            if file.endswith(".csv"):
                df = pd.read_csv(file, delimiter=",", header=0)
            elif file.endswith(".tsv"):
                df = pd.read_csv(file, delimiter="\t", header=0)
            else:
                QMessageBox.critical(
                    self,
                    "Error",
                    "Incorrect file format!",
                    QMessageBox.Ok | QMessageBox.No,
                    QMessageBox.Ok,
                )

            if "Label" in df.columns and "Score" in df.columns:
                self.raw_data = pd.DataFrame(
                    {
                        "col0": df.Label,
                        "col1": df.Label,
                        "col2": df.Score,
                        "col3": df.Score,
                    }
                )
                if self.curve == "ROC":
                    fpr, tpr, _ = roc_curve(
                        df.Label.astype(int), df.Score.astype(float)
                    )
                    self.auc = round(auc(fpr, tpr), 4)
                    self.dot = pd.DataFrame(
                        np.hstack((fpr.reshape((-1, 1)), tpr.reshape((-1, 1)))),
                        columns=["fpr", "tpr"],
                    )
                    self.file.setText(file)
                if self.curve == "PRC":
                    precision, recall, _ = precision_recall_curve(
                        df.Label.astype(int), df.Score.astype(float)
                    )
                    self.auc = round(auc(recall, precision), 4)
                    self.dot = pd.DataFrame(
                        np.hstack(
                            (recall.reshape((-1, 1)), precision.reshape((-1, 1)))
                        ),
                        columns=["recall", "precision"],
                    )
                    self.file.setText(file)
            else:
                QMessageBox.critical(
                    self,
                    "Error",
                    "Incorrect file format!",
                    QMessageBox.Ok | QMessageBox.No,
                    QMessageBox.Ok,
                )

    def setColor(self):
        self.color = QColorDialog.getColor().name()
        self.colorLineEdit.setText(self.color)

    def setWidth(self):
        lw, ok = QInputDialog.getInt(self, "Line width", "Get line width", 1, 1, 6, 1)
        if ok:
            self.lineWidth = lw
            self.widthLineEdit.setText(str(lw))

    def getLineStyle(self):
        self.lineStyle = self.styleLineEdit.currentText()

    def getPrefix(self):
        return self.legendLineEdit.text()

    @staticmethod
    def getValues(curve):
        dialog = QPlotInput(curve)
        result = dialog.exec_()
        prefix = dialog.getPrefix()
        if prefix != "":
            return (
                dialog.auc,
                dialog.dot,
                dialog.color,
                dialog.lineWidth,
                dialog.lineStyle,
                prefix,
                dialog.raw_data,
                result == QDialog.Accepted,
            )
        else:
            QMessageBox.critical(
                dialog,
                "Error",
                "Empty field!",
                QMessageBox.Ok | QMessageBox.No,
                QMessageBox.Ok,
            )
            return (
                dialog.auc,
                dialog.dot,
                dialog.color,
                dialog.lineWidth,
                dialog.lineStyle,
                prefix,
                dialog.raw_data,
                False,
            )


class QBoxPlotInput(QDialog):
    def __init__(self):
        super(QBoxPlotInput, self).__init__()
        self.initUI()
        self.dataframe = None

    def initUI(self):
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setFont(QFont("Arial", 8))
        self.setWindowIcon(QIcon("images/logo.ico"))
        self.setWindowTitle("SIFS Boxplot")
        self.resize(500, 50)
        layout = QFormLayout(self)
        self.file = MyLineEdit()
        self.file.clicked.connect(self.getFileName)
        self.x_label = MyLineEdit("X label name")
        self.y_label = MyLineEdit("Y label name")
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow("Data file:", self.file)
        layout.addRow("X label:", self.x_label)
        layout.addRow("Y label:", self.y_label)
        layout.addWidget(self.buttons)

    def getFileName(self):
        file, ok = QFileDialog.getOpenFileName(
            self, "Open", "./data", "CSV Files (*.csv);;TSV Files (*.tsv)"
        )
        if ok:
            if file.endswith(".csv"):
                self.dataframe = pd.read_csv(file, delimiter=",", header=0)
            elif file.endswith(".tsv"):
                self.dataframe = pd.read_csv(file, delimiter="\t", header=0)
            else:
                QMessageBox.critical(
                    self,
                    "Error",
                    "Incorrect file format!",
                    QMessageBox.Ok | QMessageBox.No,
                    QMessageBox.Ok,
                )
            self.file.setText(file)

    def getLabelNames(self):
        return self.x_label.text(), self.y_label.text()

    @staticmethod
    def getValues():
        dialog = QBoxPlotInput()
        result = dialog.exec_()
        labels = dialog.getLabelNames()
        return labels[0], labels[1], dialog.dataframe, result == QDialog.Accepted


class QFileTransformation(QDialog):
    def __init__(self):
        super(QFileTransformation, self).__init__()
        self.initUI()

    def initUI(self):
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setFont(QFont("Arial", 8))
        self.setWindowIcon(QIcon("images/logo.ico"))
        self.setWindowTitle("SIFS File Transformer")
        self.resize(500, 50)
        layout = QFormLayout(self)
        self.file = MyLineEdit()
        self.file.clicked.connect(self.getFileName)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow("Data file:", self.file)
        layout.addWidget(self.buttons)

    def getFileName(self):
        file, ok = QFileDialog.getOpenFileName(
            self,
            "Open",
            "./data",
            "CSV Files (*.csv);;TSV Files (*.tsv);;SVM Files(*.svm);;Weka Files (*.arff)",
        )
        self.file.setText(file)

    def getName(self):
        return self.file.text()

    @staticmethod
    def getValues():
        dialog = QFileTransformation()
        result = dialog.exec_()
        fileName = dialog.getName()
        return fileName, result == QDialog.Accepted


class QHeatmapInput(QDialog):
    def __init__(self):
        super(QHeatmapInput, self).__init__()
        self.initUI()
        self.dataframe = None

    def initUI(self):
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setFont(QFont("Arial", 8))
        self.setWindowIcon(QIcon("images/logo.ico"))
        self.setWindowTitle("SIFS Boxplot")
        self.resize(500, 50)
        layout = QFormLayout(self)
        self.file = MyLineEdit()
        self.file.clicked.connect(self.getFileName)
        self.x_label = MyLineEdit("X label name")
        self.y_label = MyLineEdit("Y label name")
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow("Data file:", self.file)
        layout.addRow("X label:", self.x_label)
        layout.addRow("Y label:", self.y_label)
        layout.addWidget(self.buttons)

    def getFileName(self):
        file, ok = QFileDialog.getOpenFileName(
            self, "Open", "./data", "CSV Files (*.csv);;TSV Files (*.tsv)"
        )
        if ok:
            if file.endswith(".csv"):
                self.dataframe = pd.read_csv(file, delimiter=",", header=0, index_col=0)
            elif file.endswith(".tsv"):
                self.dataframe = pd.read_csv(
                    file, delimiter="\t", header=0, index_col=0
                )
            else:
                QMessageBox.critical(
                    self,
                    "Error",
                    "Incorrect file format!",
                    QMessageBox.Ok | QMessageBox.No,
                    QMessageBox.Ok,
                )
            self.file.setText(file)

    def getLabelNames(self):
        return self.x_label.text(), self.y_label.text()

    @staticmethod
    def getValues():
        dialog = QHeatmapInput()
        result = dialog.exec_()
        labels = dialog.getLabelNames()

        return labels[0], labels[1], dialog.dataframe, result == QDialog.Accepted


class QSelectModel(QDialog):
    def __init__(self, model_list):
        super(QSelectModel, self).__init__()
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QIcon("images/logo.ico"))
        self.model_list = model_list
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.resize(400, 100)
        self.setWindowTitle("Select")
        self.setFont(QFont("Arial", 10))
        layout = QFormLayout(self)
        self.modelComboBox = QComboBox()
        self.modelComboBox.addItems(self.model_list)
        self.modelComboBox.setCurrentIndex(0)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow("Select model to save:", self.modelComboBox)
        layout.addWidget(self.buttons)

    @staticmethod
    def getValues(model_list):
        dialog = QSelectModel(model_list)
        result = dialog.exec_()
        model = dialog.modelComboBox.currentText()
        return model, result == QDialog.Accepted


class QSCombineModelDialog(QDialog):
    def __init__(self, model_list):
        super(QSCombineModelDialog, self).__init__()
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QIcon("images/logo.ico"))
        self.model_list = model_list
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowTitle("Combine models")
        self.resize(500, 90)
        self.setFont(QFont("Arial", 10))
        layout = QFormLayout(self)
        self.modelComboBox = QComboBox()
        self.modelComboBox.addItems(
            [
                "LR",
                "RF",
                "SVM",
                "DecisionTree",
                "LightGBM",
                "XGBoost",
                "KNN",
                "LDA",
                "QDA",
                "SGD",
                "NaiveBayes",
                "Bagging",
                "AdaBoost",
                "GBDT",
            ]
        )
        self.modelComboBox.setCurrentIndex(0)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow("Use what algorithm to combine models:", self.modelComboBox)
        layout.addWidget(self.buttons)

    @staticmethod
    def getValues(model_list):
        dialog = QSCombineModelDialog(model_list)
        result = dialog.exec_()
        model = dialog.modelComboBox.currentText()
        return model, result == QDialog.Accepted


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = QFileTransformation()
    win.show()
    sys.exit(app.exec_())
