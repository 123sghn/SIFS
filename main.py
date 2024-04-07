# GUI for program startup
import os
import re
import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QDesktopWidget,
    QLabel,
    QHBoxLayout,
    QMessageBox,
    QAction,
    QFileDialog,
    QFrame,
)
from PyQt5.QtGui import QIcon, QFont, QPixmap, QCloseEvent, QDesktopServices
from PyQt5.QtCore import Qt, QUrl
import qdarkstyle
import threading
import pandas as pd
import numpy as np

import SIFS
from qdarkstyle.light.palette import LightPalette
from qt_material import apply_stylesheet
import qdarktheme


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.initUI()

    def initUI(self):

        self.setWindowTitle("SIFS")
        self.setMaximumSize(540, 484)
        self.setMinimumSize(540, 484)
        self.setWindowIcon(QIcon("images/logo.ico"))
        self.setFont(QFont("Arial"))
        bar = self.menuBar()
        app = bar.addMenu("Applications")
        basic = QAction("SIFS", self)
        basic.triggered.connect(self.openBasicWindow)

        quit = QAction("Exit", self)
        quit.triggered.connect(self.closeEvent)
        app.addAction(basic)
        app.addSeparator()
        app.addAction(quit)

        help = bar.addMenu("Help")
        document = QAction("Document", self)
        document.triggered.connect(self.openDocumentUrl)
        help.addActions([document])

        # move window to center
        self.moveCenter()

        self.widget = QWidget()
        hLayout = QHBoxLayout(self.widget)
        hLayout.setAlignment(Qt.AlignCenter)
        label = QLabel()
        pixmap = QPixmap("images/logo.png").scaled(label.size(), Qt.KeepAspectRatio)
        label.setPixmap(pixmap)
        label.setFrameStyle(QFrame.NoFrame)
        hLayout.addWidget(label)
        self.setCentralWidget(self.widget)

    def moveCenter(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        newLeft = (screen.width() - size.width()) / 2
        newTop = (screen.height() - size.height()) / 2
        self.move(int(newLeft), int(newTop))

    def openDocumentUrl(self):
        local_pdf_path = "document\SIFS_manual.pdf"
        if os.path.exists(local_pdf_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(local_pdf_path))
        else:
            print("File does not exist or path is incorrect.")

    def openBasicWindow(self):
        self.basicWin = SIFS.SIFS()
        self.basicWin.setFont(QFont("Arial", 10))
        self.basicWin.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        # self.basicWin.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5', palette=LightPalette()))
        # apply_stylesheet(self.basicWin, theme='dark_pink.xml')
        self.basicWin.close_signal.connect(self.recover)
        self.basicWin.show()
        self.setDisabled(True)
        self.setVisible(False)

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self,
            "Confirm Exit",
            "Are you sure want to quit SIFS?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            sys.exit(0)
        else:
            if event:
                event.ignore()

    def recover(self, module):
        try:
            if module == "Basic":
                del self.basicWin
            else:
                pass
        except Exception as e:
            pass
        self.setDisabled(False)
        self.setVisible(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()

    window.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    window.show()
    sys.exit(app.exec_())
