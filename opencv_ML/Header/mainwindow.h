#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include <QMainWindow>
#include <opencv/cv.hpp>
#include <QImage>
#include <opencv2/videoio/videoio.hpp>
#include <QPixmap>
#include <QFileDialog>
#include <QUdpSocket>
#include <tfjs-node/binding/tf_auto_tensor.h>
#include <tfjs-node/deps/include/tensorflow/c/c_api.h>
#include <opencv/cv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
//#include <QMediaPlayer>
//#include <QVideoWidget>
//#include <QMediaPlaylist>

using namespace std;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
     void opencv();
     bool flag;


private slots:
     void on_pushButton_clicked();
     void on(QImage);
     void on_photo_clicked();
     void cam(QImage);
     void on_stopbutton_clicked(QImage);


signals:
     void mysignal(QImage);

private:
    Ui::MainWindow *ui;
    cv::Mat minisk;
    double rate;
    QTimer *timer;
    QPainter *painter;
    QPixmap pixmap;
    QUdpSocket *m_udpSocket;
};

#endif // MAINWINDOW_H
