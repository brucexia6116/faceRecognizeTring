import sys
import os

from PyQt5.QtWidgets import QMainWindow, QAction, qApp, QApplication
from PyQt5.QtWidgets import QPushButton,QWidget,QLabel,QSlider
from PyQt5.QtWidgets import QHBoxLayout,QVBoxLayout,QFileDialog,QDialog\
    ,QListWidget,QListWidgetItem,QInputDialog,QMessageBox,QLineEdit
from PyQt5.QtGui import QImage,QPixmap,QFont,QPainter
from PyQt5.QtCore import Qt,pyqtSignal,QTimer

import cv2

import sampler
import recognizer
import classfier

class ShowLabel(QLabel):
    def __init__(self,parent):
        super().__init__(parent)

        self.x0=None
        self.x1=None
        self.y0=None
        self.y1=None

    def setRect(self,rect):
        if rect==None:
            self.x0 = None
            self.x1 = None
            self.y0 = None
            self.y1 = None
            return
        self.x0=rect[0]
        self.x1=rect[1]
        self.y0=rect[2]
        self.y1=rect[3]

    def paintEvent(self,e):
        super().paintEvent(e)
        if self.x0==None or self.x1==None or self.y0==None or self.y1==None:
            return
        geometry=self.geometry()

        x0=int(geometry.width()*self.x0)
        x1=int(geometry.width()*self.x1)
        y0=int(geometry.height()*self.y0)
        y1=int(geometry.height()*self.y1)

        painter = QPainter(self)
        painter.setPen(Qt.red)

        painter.drawLine(x0,y0,x0,y1)
        painter.drawLine(x0, y0, x1, y0)
        painter.drawLine(x1, y1, x1, y0)
        painter.drawLine(x1, y1, x0, y1)

class ControlWindow(QMainWindow):
    def __init__(self,parent):
        super().__init__(parent)

        self.title="tring"
        self.openCamera=False
        self.caper=None

        self.samplerAction=QAction("sampler",self)
        self.trainAction=QAction("train",self)
        self.saveAction=QAction("save",self)
        self.loadAction=QAction("load",self)
        self.selectImageAction=QAction("selectIamge",self)
        self.cameraAction=QAction("openCamera",self)

        self.imageLabel=ShowLabel(None)
        self.timer=QTimer(self)

        self.initUI()
        self.initAction()

    def initUI(self):
        widget=QWidget(None)
        layout=QHBoxLayout(None)
        layout.addWidget(self.imageLabel,1,Qt.AlignCenter)
        widget.setLayout(layout)

        self.setCentralWidget(widget)

        menubar = self.menuBar()

        moduleMenu=menubar.addMenu("module")
        moduleMenu.addAction(self.samplerAction)
        moduleMenu.addAction(self.trainAction)
        moduleMenu.addAction(self.saveAction)
        moduleMenu.addAction(self.loadAction)

        imageMenu=menubar.addMenu("image")
        imageMenu.addAction(self.selectImageAction)
        imageMenu.addAction(self.cameraAction)

        self.resize(300,300)

    def initAction(self):
        self.samplerAction.triggered.connect(self.onSampler)
        self.trainAction.triggered.connect(self.onTrain)
        self.saveAction.triggered.connect(self.onSave)
        self.loadAction.triggered.connect(self.onLoad)

        self.selectImageAction.triggered.connect(self.onSelectImage)
        self.cameraAction.triggered.connect(self.onCamera)

        self.timer.timeout.connect(self.onTimer)
        self.timer.start(100)

    def onSampler(self):
        path = QFileDialog.getExistingDirectory(self, "select sample dir")
        if len(path)<=0:return

        sampler.initSampler(path)

    def onTrain(self):
        classfier.startTrain()

    def onSave(self):
        path = QFileDialog.getSaveFileName(self, "save path")[0]
        if len(path) == 0: return
        print(path)
        classfier.setSavePath(path)

    def onLoad(self):
        path = QFileDialog.getOpenFileName(self, "select")[0]
        if len(path) == 0: return
        path = os.path.splitext(path)[0]
        print(path)
        classfier.loadModule(path)

    def passImage(self,image):
        retRect = recognizer.getFaceRegion(image)
        self.imageLabel.setRect(retRect)

        cv2.imwrite("temp.png", image)
        pixmap = QPixmap.fromImage(QImage("temp.png"))
        #pixmap=QPixmap.fromImage(image)
        self.imageLabel.setPixmap(pixmap)

        self.imageLabel.update()

    def onSelectImage(self):
        if self.openCamera==True:return

        path = QFileDialog.getOpenFileName(self, "select")[0]
        if len(path) == 0: return

        image=cv2.imread(path)
        self.passImage(image)

    def onCamera(self):
        if self.openCamera==False:
            try:
                self.caper=cv2.VideoCapture(0)
            except Exception as e:
                print(str(e))
                return
            self.openCamera=True
        else:
            try:
                self.caper.release()
                self.caper=None
            except Exception as e:
                print(str(e))
                return
            self.openCamera=False


    def onTimer(self):
        if len(classfier.message)==0:
            self.setWindowTitle(self.title)
        else:
            self.setWindowTitle(classfier.message)

        if self.openCamera==True:
            ret, image = self.caper.read()
            self.passImage(image)


if __name__ =="__main__":
    app = QApplication(sys.argv)
    window = ControlWindow(None)
    window.show()
    sys.exit(app.exec_())