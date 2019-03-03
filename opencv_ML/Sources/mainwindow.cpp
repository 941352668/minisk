#define CPU_ONLY
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include<cvaux.hpp>
#include <opencv2/video/video.hpp>
#include <QImage>
#include <QPixmap>
#include <QPainter>
#include <QMovie>
#include <QLabel>
#include <QPoint>
#include <QTimer>
#include <QFileDialog>
#include <vector>
#include <QDebug>
#include <QScreen>
#include<QByteArray>
#include <QTime>
#include <QCoreApplication>
#include <tfjs-node/binding/tf_auto_tensor.h>
#include <tfjs-node/deps/include/tensorflow/c/c_api.h>
#include <QVector>
#include <stddef.h>
#include <stdint.h>
#include <QBuffer>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/face.hpp>
#include <opencv2/face/facerec.hpp>
#include <opencv2/face/predict_collector.hpp>
#include <iostream>
#include <fstream>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/core/utils/trace.hpp>
#include <cstdlib>
#include <stdio.h>
#include <sys/types.h>
#include <fcntl.h>
#include <dirent.h>





using namespace cv::dnn;
using namespace std;
MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    connect(this,SIGNAL(mysignal(QImage)),this,SLOT(on(QImage)));
    minisk = cv::imread("/home/berlink/图片/5d6034a85edf8db118ad83830823dd54574e7452.jpg");
    QImage openimg((const uchar*)minisk.data, minisk.cols, minisk.rows, minisk.step, QImage::Format_RGB888 );
    ui->ImageCapture->setPixmap(QPixmap::fromImage(openimg));
    ui->ImageCapture->resize(openimg.size());
    ui->ImageCapture->setScaledContents(true);
    m_udpSocket = new QUdpSocket(this);
    //QFile file("/home/berlink/demo/open/deploy.prototxt");
        //file.open(QIODevice::ReadOnly | QIODevice::Text);
        //QByteArray arr = file.readAll();
        //ui->ImageCapture->setText(QString(arr));
       // file.close();

}

MainWindow::~MainWindow()
{
    delete ui;

}

void MainWindow::opencv(){

    cv::VideoCapture cap;
     //cap.open(0);
    //cap.open("http://192.168.1.4:8080/video");
    //vivo
    //cap.open("http://192.168.1.5:8081");
    //bilibili
    cap.open("/home/berlink/音乐/妹子.mp4");
    //::play("http://upos-hz-mirrorks3u.acgvideo.com/upgcxcode/64/49/79144964/79144964-1-6.mp4?e=ig8euxZM2rNcNbug7WdVtWug7WdVNEVEuCIv29hEn0l5QK==&deadline=1551597440&gen=playurl&nbs=1&oi=22407493&os=ks3u&platform=html5&trid=80e2577b403942dca0390fe23730a6bc&uipk=5&upsig=5cc2948e626884ee413e0208517d39da&uparams=e,deadline,gen,nbs,oi,os,platform,trid,uipk");
    cv::Mat minisk;
    minisk = cv::imread("/home/berlink/图片/5d6034a85edf8db118ad83830823dd54574e7452.jpg");
    cv::String prototxt = "/home/berlink/demo/open/deploy.prototxt";
    cv::String weights = "/home/berlink/demo/open/res10_300x300_ssd_iter_140000.caffemodel";
    cv::dnn::Net net = cv::dnn::readNetFromCaffe(prototxt, weights);
    const cv::Scalar meanVal(104.0, 177.0, 123.0);
    cv::Mat frame; //定义Mat变量，用来存储每一帧

    while(flag)
    {   

    cap>>frame; //读取当前帧方法一
    //cv::cvtColor(frame,frame,CV_HSV2RGB_FULL);
    //cv::cvtColor(frame,frame,CV_BGR2GRAY);
       frame.channels() == 3;
    cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
    const size_t inWidth = 300;
    const size_t inHeight = 300;
    const float WHRatio = inWidth / (float)inHeight;
    const double inScaleFactor = 1.0;
     //vector<int> labels;

    //QString soundpath = "/home/berlink/音乐/9981.wav";
   // effect->setSource(QUrl::fromLocalFile("/home/berlink/音乐/9981.wav"));
   // effect->setLoopCount(QSoundEffect::Infinite);
   // effect->setVolume(1.0f);
   // effect->play();
    //QSound *sound = new QSound("/home/berlink/音乐/9981.wav");
    //sound->play();
    //const char* classNames[] = { "background","face" };

   // cv::dnn::Net net = cv::dnn::readNetFromCaffe(weights, prototxt);
   //cv::dnn::Net net = cv::dnn::readNetFromTensorflow(weights);
    cv::Size frame_size = frame.size();
    cv::Size cropSize;
    if (frame_size.width / (float)frame_size.height > WHRatio)
        {
            cropSize = cv::Size(static_cast<int>(frame_size.height * WHRatio), frame_size.height);
        }
    else
       {
           cropSize = cv::Size(frame_size.width,
               static_cast<int>(frame_size.width / WHRatio));
       }
   // cv::Rect crop(cv::Point((frame_size.width - cropSize.width) / 2,
           // (frame_size.height - cropSize.height) / 2),
            //cropSize);

    cv::Mat inputBlob = blobFromImage(frame,inScaleFactor,cv::Size(inWidth, inHeight), meanVal, false, false);
    //cv::Mat blob = cv::dnn::blobFromImage(frame,1./255,cv::Size(300,300));
     //cv::Net.setInput(cv2.dnn.blobFromImage(img, 1.0/127.5, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    //cv::dnn::Net.setInput(cv::dnn::blobFromImage(frame, 1.0/127.5, (300, 300),cv::Size(300,300));
    net.setInput(inputBlob, "data");
    //cv::Mat output = net.forward();

    cv::Mat detection= net.forward("detection_out");

    vector<double> layersTimings;
           double freq = cv::getTickFrequency() / 1000;
           double time = net.getPerfProfile(layersTimings) / freq;

    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

   // frame = frame(crop);

    float confidenceThreshold = 0.20;
    //float confidenceThreshold =  min_confidence;
    ostringstream ss;
     ss << "FPS: " << 1000 / time << " ; time: " << time << " ms";
     cv::putText(frame, ss.str(), cv::Point(20, 20), 0, 0.5, cv::Scalar(0, 0, 255));
    for (int i = 0; i < detectionMat.rows; i++)
        {
            float confidence = detectionMat.at<float>(i, 2);
                if (confidence > confidenceThreshold)
                {
                int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
                int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
                int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
                int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

               cv::Rect object((int)xLeftBottom, (int)yLeftBottom, (int)(xRightTop - xLeftBottom), (int)(yRightTop - yLeftBottom));
               cv::rectangle(frame, object, cv::Scalar(0, 255, 0));

                ss.str("");
                ss << confidence;
                cv::String conf(ss.str());

                cv::String label = "face" + conf;
                //float labels = conf;
               //cv::String conf(ss.str());
               int baseLine = 0;
               cv::Size labelSize = cv::getTextSize(label, CV_FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
               cv::rectangle (frame, cv::Rect(cv::Point(xLeftBottom, yLeftBottom - labelSize.height),
                                             cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(0, 255, 0), CV_FILLED);
               cv::putText(frame, label, cv::Point(xLeftBottom, yLeftBottom), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0)); }
            cv::namedWindow("image", CV_WINDOW_NORMAL);
            //cv::cvtColor(frame,frame,CV_BGR2);
            //cv::equalizeHist( frame, frame );
        float labels[2] = { 0, 1 };
            cv::Mat train_data_mat(6, 2, CV_32FC1, frame.data);
            cv::Mat labels_mat(6, 1, CV_32FC1, labels);
            cv::Mat layers_size = (cv::Mat_<int>(1, 3) << 2, 6, 1);
            cv::Ptr<cv::ml::ANN_MLP> ann = cv::ml::ANN_MLP::create();
                ann->setLayerSizes(layers_size);
                ann->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 1, 1);
                ann->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER+cv::TermCriteria::EPS, 300, FLT_EPSILON));
                ann->setTrainMethod(cv::ml::ANN_MLP::BACKPROP, 0.001);
                cv::Ptr<cv::ml::TrainData> tData = cv::ml::TrainData::create(train_data_mat, cv::ml::ROW_SAMPLE, labels_mat);
                ann->train(tData);
                ann->save("/home/berlink/demo/open/FacetoFace.xml");
               // ann->write("/home/berlink/demo/open/FacetoFace.xml"));
              // cv::imwrite("/home/berlink/demo/open/FacetoFace.xml",ann->train(tData));
               // cv::imshow("hhh",tData);
                  //QFile file("/home/berlink/demo/open/FacetoFace.xml");
                  // file.write(newtData);
            cv::imshow("image", frame);
            if (cv::waitKey(10) >= 0)
                //cv::cvtColor(frame,frame,CV_GRAY2BGR);
                cv::imshow("image", frame);
                break;
               }
     //cv::cvtColor(frame,frame,CV_BGR2GRAY);
     //cv::threshold(minisk, frame,125, 600,CV_THRESH_TOZERO);
     //cv::Canny(frame, frame, 200,400, 5);
    //cv::imshow("image", frame);
     //cv::cvtColor(frame,frame,CV_GRAY2BGR);
     //cv::Ptr<cv::face::FaceRecognizer> model = cv::face::createEigenFaceRecognizer();

      QImage image((const uchar*)frame.data, frame.cols, frame.rows, frame.step, QImage::Format_RGB888 );
     emit mysignal(image);
            }









   // cv::blur(frame, frame, cv::Size(3, 3));
    //cv::equalizeHist( frame, frame );
   //cv::imshow("Image",frame);
   // cv::Mat baby= cv::imread("/home/berlink/baby.jpg");
   // std::string str = "/home/berlink/baby.jpg";


   // cv::Mat baby;


    //cv::cvtColor(baby,baby,CV_BGR2GRAY);
   // cv::namedWindow("Original Image");

   // cv::imshow("ddd",caffe::DecodeDatumToCVMatNative(caffimg));
   // cv::bilateralFilter(frame, baby1,10, 10 * 2, 10 / 2);
    // cv::Mat  kernel = (cv::Mat_<float>(3, 3) << 0, -1, 0, 0, 5, 0, 0, -1, 0);
    // cv::filter2D(baby1, frame, CV_8UC3, kernel);
  // cv::threshold(frame, baby1,140, 250,  CV_THRESH_BINARY     );
   //cv::imshow("Image",frame);




      // cv::Canny(baby1, baby, 100,250, 5);
        //cv::Mat newphoto = 255 - baby;
        //cv::imshow("Original Image",newphoto);

    //cv::imshow("Original Image",baby);
    //vector<cv::Mat> rgbChannels(3);
    //cv::split(frame,rgbChannels);
   // cv::Mat blank_ch, fin_img;
    //blank_ch = cv::Mat::zeros(cv::Size(frame.cols, frame.rows), CV_8UC1);

    //vector<cv::Mat> channels_r;
   // channels_r.push_back(blank_ch);
    //channels_r.push_back(blank_ch);
   // channels_r.push_back(rgbChannels[2]);

   // cv::merge(channels_r, fin_img);
    //cv::imshow("R", fin_img);

   // vector<cv::Mat> channels_g;
      //  channels_g.push_back(blank_ch);
      //  channels_g.push_back(rgbChannels[1]);
       // channels_g.push_back(blank_ch);
       // cv::merge(channels_g, fin_img);
        //cv::imshow("G", fin_img);


    // vector<cv::Mat> channels_b;
       // channels_b.push_back(rgbChannels[0]);
       // channels_b.push_back(blank_ch);
       // channels_b.push_back(blank_ch);
       // cv::merge(channels_b, fin_img);
       // cv::imshow("B", fin_img);



   //cv::medianBlur(frame, frame, 3);
     //cv::blur(frame, edge, Size(3, 3));
    //cv::threshold(frame, frame, 0, 255, CV_THRESH_OTSU);
   //cv::Canny(frame, frame, 150,550, 5);
   //色度反向
   //cv::Mat inversedMat = 255 - frame;
   //cv::imshow("canny",inversedMat);
  //cv::imwrite("MyQrcode02.jpg",frame);



    //cap.read(frame); //读取当前帧方法二
    //cv::namedWindow("222", CV_WINDOW_NORMAL);

//显示一帧画面
    cv::waitKey(10); //延时30ms

    //void QPainter::drawPixmap(int x, int y, int width, int height, const QPixmap &pixmap)
    //QImage image((const uchar*)frame.data, frame.cols, frame.rows, frame.step, QImage::Format_RGB888 );
   //
    }



void MainWindow::on_pushButton_clicked()
{
    opencv();
}

void MainWindow::on(QImage image){

    ui->ImageCapture->setPixmap(QPixmap::fromImage(image));
    ui->ImageCapture->resize(image.size());
    ui->ImageCapture->setScaledContents(true);
    if(Qt::WindowMaximized){
        ui->pushButton->hide();
    }
}

void MainWindow::on_stopbutton_clicked(QImage image)
{
  QByteArray byte;
  QBuffer buff(&byte);
  image.save(&buff,"JPEG");
  QByteArray compress = qCompress(byte,1);

  QByteArray base64Byte = compress.toBase64();

  m_udpSocket->writeDatagram(base64Byte.data(),base64Byte.size(),QHostAddress::Any,45454);

}

void MainWindow::on_photo_clicked()
{
   flag = false;
   connect(this,SIGNAL(mysignal(QImage)),this,SLOT(cam(QImage)),Qt::UniqueConnection);
   connect(this,SIGNAL(mysignal(QImage)),this,SLOT(on_stopbutton_clicked(QImage)));


}

void MainWindow::cam(QImage image)
{
     while (!flag) {
        image.save("bbb.png");
        break;
     }
    flag = true;
    return;
}






