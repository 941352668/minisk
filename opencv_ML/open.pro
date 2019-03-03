#-------------------------------------------------
#
# Project created by QtCreator 2019-02-03T10:21:51
#
#-------------------------------------------------

QT += core gui opengl multimedia
QT += multimediawidgets
QT += network  multimedia


TARGET = opencv_test
CONFIG += console
CONFIG -= app_bundle

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = open
TEMPLATE = app

INCLUDEPATH +=  usr/local/include \
               /usr/local/include/opencv \
               /usr/local/include/opencv2 \
               /usr/local/include/tfjs-node/deps \
               /usr/local/include/gflags \
               /usr/local/include/glog \
               /usr/local/include/caffe\
               /usr/local/include/caffe/modules/dnn/src/caffe
            # /home/berlink/opencv/modules/dnn/src/caffe
#INCLUDEPATH += /home/berlink/.local/share/Trash/files/caffe-master/include/caffe
#INCLUDEPATH += /home/berlink/.local/share/Trash/files/caffe-master/include/caffe/util
#INCLUDEPATH += /home/berlink/.local/share/Trash/files/caffe-master/include/caffe/test
#INCLUDEPATH += /home/berlink/.local/share/Trash/files/caffe-master/include/caffe/layers


#INCLUDEPATH += /home/berlink/下载/tf_test/include\
               #/home/berlink/下载/tf_test/include\tensorflow



LIBS += /usr/local/lib/libopencv_highgui.so \
        /usr/local/lib/libopencv_core.so \
        /usr/local/lib/libopencv_imgproc.so \
        /usr/local/lib/libopencv_imgcodecs.so
        #/usr/local/include/caffe
LIBS += -L/home/berlink/.local/share/Trash/files/caffe-master/src/caffe
LIBS += -L/home/berlink/.local/share/Trash/files/caffe-master/src/caffe
LIBS += -L/home/berlink/.local/share/Trash/files/caffe-master/src/caffe/util
LIBS += -L/home/berlink/.local/share/Trash/files/caffe-master/src/caffe/test
LIBS += -L/home/berlink/.local/share/Trash/files/caffe-master/src/caffe/layers
LIBS += -L/home/berlink/.local/share/Trash/files/caffe-master/tools
LIBS += -L/home/berlink/.local/share/Trash/files/caffe-master/solvers
LIBS += -L/home/berlink/opencv/modules/dnn/src/caffe






# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0


SOURCES += \
        main.cpp \
        mainwindow.cpp


HEADERS += \
        mainwindow.h

FORMS += \
        mainwindow.ui \
    udpclient.ui

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../../usr/local/lib/release/ -lCGAL
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../../usr/local/lib/debug/ -lCGAL
else:unix: LIBS += -L$$PWD/../../../../usr/local/lib/ -lCGAL

INCLUDEPATH += $$PWD/../../../../usr/local/include
DEPENDPATH += $$PWD/../../../../usr/local/include

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../../usr/local/lib/release/ -lopencv_video
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../../usr/local/lib/debug/ -lopencv_video
else:unix: LIBS += -L$$PWD/../../../../usr/local/lib/ -lopencv_video

INCLUDEPATH += $$PWD/../../../../usr/local/include
DEPENDPATH += $$PWD/../../../../usr/local/include

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../../usr/local/lib/release/ -lopencv_videoio
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../../usr/local/lib/debug/ -lopencv_videoio
else:unix: LIBS += -L$$PWD/../../../../usr/local/lib/ -lopencv_videoio

INCLUDEPATH += $$PWD/../../../../usr/local/include
DEPENDPATH += $$PWD/../../../../usr/local/include

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../../usr/local/lib/release/ -lCGAL_ImageIO
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../../usr/local/lib/debug/ -lCGAL_ImageIO
else:unix: LIBS += -L$$PWD/../../../../usr/local/lib/ -lCGAL_ImageIO

INCLUDEPATH += $$PWD/../../../../usr/local/include
DEPENDPATH += $$PWD/../../../../usr/local/include

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../../usr/local/lib/release/ -lopencv_video
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../../usr/local/lib/debug/ -lopencv_video
else:unix: LIBS += -L$$PWD/../../../../usr/local/lib/ -lopencv_video

INCLUDEPATH += $$PWD/../../../../usr/local/include
DEPENDPATH += $$PWD/../../../../usr/local/include

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../../usr/local/lib/release/ -lopencv_core
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../../usr/local/lib/debug/ -lopencv_core
else:unix: LIBS += -L$$PWD/../../../../usr/local/lib/ -lopencv_core

INCLUDEPATH += $$PWD/../../../../usr/local/include
DEPENDPATH += $$PWD/../../../../usr/local/include

unix: CONFIG += link_pkgconfig
unix: PKGCONFIG += opencv










win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../下载/release/ -ltensorflow-core
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../下载/debug/ -ltensorflow-core
else:unix: LIBS += -L$$PWD/../../下载/ -ltensorflow-core

INCLUDEPATH += $$PWD/../../下载
DEPENDPATH += $$PWD/../../下载

win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../../下载/release/libtensorflow-core.a
else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../../下载/debug/libtensorflow-core.a
else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../../下载/release/tensorflow-core.lib
else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../../下载/debug/tensorflow-core.lib
else:unix: PRE_TARGETDEPS += $$PWD/../../下载/libtensorflow-core.a

unix|win32: LIBS += -ltensorflow-core

LIBS += -lboost_system
LIBS += -Lcaffe
# other dependencies

