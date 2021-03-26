//
// Created by qiulinmin on 8/1/17.
//
#include <jni.h>
#include <string>
#include <android_utils.h>
#include <Scanner.h>
#include <Filter.h>
#include "android/log.h"

using namespace std;

static const char* const kClassDocScanner = "me/pqpo/smartcropperlib/SmartCropper";

static struct {
    jclass jClassPoint;
    jmethodID jMethodInit;
    jfieldID jFieldIDX;
    jfieldID jFieldIDY;
} gPointInfo;

static void initClassInfo(JNIEnv *env) {
    gPointInfo.jClassPoint = reinterpret_cast<jclass>(env -> NewGlobalRef(env -> FindClass("android/graphics/Point")));
    gPointInfo.jMethodInit = env -> GetMethodID(gPointInfo.jClassPoint, "<init>", "(II)V");
    gPointInfo.jFieldIDX = env -> GetFieldID(gPointInfo.jClassPoint, "x", "I");
    gPointInfo.jFieldIDY = env -> GetFieldID(gPointInfo.jClassPoint, "y", "I");
}

static jobject createJavaPoint(JNIEnv *env, Point point_) {
    return env -> NewObject(gPointInfo.jClassPoint, gPointInfo.jMethodInit, point_.x, point_.y);
}

static void native_scan(JNIEnv *env, jclass type, jobject srcBitmap, jobjectArray outPoint_, jboolean canny) {
    if (env -> GetArrayLength(outPoint_) != 4) {
        return;
    }
    Mat srcBitmapMat;
    bitmap_to_mat(env, srcBitmap, srcBitmapMat);
    Mat bgrData(srcBitmapMat.rows, srcBitmapMat.cols, CV_8UC3);
    cvtColor(srcBitmapMat, bgrData, COLOR_RGBA2BGR);
    scanner::Scanner docScanner(bgrData, canny);
    std::vector<Point> scanPoints = docScanner.scanPoint();
    if (scanPoints.size() == 4) {
        for (int i = 0; i < 4; ++i) {
            env -> SetObjectArrayElement(outPoint_, i, createJavaPoint(env, scanPoints[i]));
        }
    }
}

static vector<Point> pointsToNative(JNIEnv *env, jobjectArray points_) {
    int arrayLength = env->GetArrayLength(points_);
    vector<Point> result;
    for(int i = 0; i < arrayLength; i++) {
        jobject point_ = env -> GetObjectArrayElement(points_, i);
        int pX = env -> GetIntField(point_, gPointInfo.jFieldIDX);
        int pY = env -> GetIntField(point_, gPointInfo.jFieldIDY);
        result.push_back(Point(pX, pY));
    }
    return result;
}

static void native_crop(JNIEnv *env, jclass type, jobject srcBitmap, jobjectArray points_, jobject outBitmap) {
    std::vector<Point> points = pointsToNative(env, points_);
    if (points.size() != 4) {
        return;
    }
    Point leftTop = points[0];
    Point rightTop = points[1];
    Point rightBottom = points[2];
    Point leftBottom = points[3];

    Mat srcBitmapMat;
    bitmap_to_mat(env, srcBitmap, srcBitmapMat);

    AndroidBitmapInfo outBitmapInfo;
    AndroidBitmap_getInfo(env, outBitmap, &outBitmapInfo);
    Mat dstBitmapMat;
    int newHeight = outBitmapInfo.height;
    int newWidth = outBitmapInfo.width;
    dstBitmapMat = Mat::zeros(newHeight, newWidth, srcBitmapMat.type());

    std::vector<Point2f> srcTriangle;
    std::vector<Point2f> dstTriangle;

    srcTriangle.push_back(Point2f(leftTop.x, leftTop.y));
    srcTriangle.push_back(Point2f(rightTop.x, rightTop.y));
    srcTriangle.push_back(Point2f(leftBottom.x, leftBottom.y));
    srcTriangle.push_back(Point2f(rightBottom.x, rightBottom.y));

    dstTriangle.push_back(Point2f(0, 0));
    dstTriangle.push_back(Point2f(newWidth, 0));
    dstTriangle.push_back(Point2f(0, newHeight));
    dstTriangle.push_back(Point2f(newWidth, newHeight));

    Mat transform = getPerspectiveTransform(srcTriangle, dstTriangle);
    warpPerspective(srcBitmapMat, dstBitmapMat, transform, dstBitmapMat.size());

    mat_to_bitmap(env, dstBitmapMat, outBitmap);
}

//操作类型 optType : 1 黑白 2 上色 3 锐化 4. 增亮 5 灰度
static void native_commonProcess(JNIEnv *env, jobject srcBitmap, jobject outBitmap, int optType){
    Mat srcBitmapMat,outBitmapMat;
    bitmap_to_mat(env, srcBitmap, srcBitmapMat);
    cv::Mat img;
    if (srcBitmapMat.channels() == 1) {
        cv::cvtColor(srcBitmapMat, img, cv::COLOR_GRAY2BGR);
    } else if (srcBitmapMat.channels() == 4) {
        cv::cvtColor(srcBitmapMat, img, cv::COLOR_BGRA2BGR);
    } else {
        srcBitmapMat.copyTo(img);
    }

    Filter filter = Filter();
    int height = img.rows;
    int width = img.cols;
    float fScale = filter.getScale(height, width, 2000);  //1024值可以设定，表示长边尺寸不得超过1024，按照长边归一化

    int window_size = (int)(width*fScale * 20 / 256);
    window_size = window_size % 2 ? window_size : window_size + 1;

    cv::Mat gray, blackWhite,colorImg, enhanceImg, brightImg;
    if (fScale == 1.0) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        filter.sauvolaWithSigmoid(gray, blackWhite, window_size, 0.1); //黑白功能
        filter.coloring(img, blackWhite, colorImg);//上色功能
        filter.colorEnhance(colorImg, enhanceImg);//上色锐化功能
        filter.brighten(colorImg,brightImg);
    } else {
        //将图像缩小
        cv::Mat imgReisize, blackWhiteResize, colorImgResize, enhanceImgResize, brightImgResize;
        cv::resize(img, imgReisize, cv::Size(), fScale, fScale, cv::INTER_CUBIC);
        //对缩小后的图像进行处理
        cv::cvtColor(imgReisize, gray, cv::COLOR_BGR2GRAY);
        filter.sauvolaWithSigmoid(gray, blackWhiteResize, window_size, 0.1);  //黑白功能
        filter.coloring(imgReisize, blackWhiteResize, colorImgResize); //上色功能
        filter.colorEnhance(colorImgResize, enhanceImgResize); //上色锐化功能
        filter.brighten(imgReisize, brightImgResize); //增亮功能

        //对处理后的图片放大到原有尺寸
        cv::resize(blackWhiteResize, blackWhite, cv::Size(width, height), cv::INTER_CUBIC);
        cv::resize(colorImgResize, colorImg, cv::Size(width, height), cv::INTER_CUBIC);
        cv::resize(enhanceImgResize, enhanceImg, cv::Size(width, height), cv::INTER_CUBIC);
        cv::resize(brightImgResize, brightImg, cv::Size(width, height), cv::INTER_CUBIC);
    }

    if(optType == 1) {
        outBitmapMat = blackWhite.clone();
    } else if (optType == 2){
        outBitmapMat = colorImg.clone();
    } else if (optType == 3) {
        outBitmapMat = enhanceImg.clone();
    } else if(optType == 4){
        outBitmapMat = brightImg.clone();
    } else if(optType == 5) {
        outBitmapMat = gray.clone();
    }
    mat_to_bitmap(env, outBitmapMat, outBitmap);
    //    if((srcBitmapMat.data != outBitmapMat.data)){
//        __android_log_print(ANDROID_LOG_ERROR, "JNI", "====>%i ===> %i",outBitmapMat.cols,outBitmapMat.rows);
//    }
}

static void native_enhance(JNIEnv *env, jclass type, jobject srcBitmap, jobject outBitmap){
//    Mat srcBitmapMat,outBitmapMat;
//    bitmap_to_mat(env, srcBitmap, srcBitmapMat);
//    cv::cvtColor(srcBitmapMat,srcBitmapMat,COLOR_BGRA2BGR);
//    outBitmapMat = Mat::zeros(srcBitmapMat.rows,srcBitmapMat.cols, srcBitmapMat.type());
//    Filter().colorEnhance(srcBitmapMat,outBitmapMat);
//    mat_to_bitmap(env, outBitmapMat, outBitmap);
    native_commonProcess(env, srcBitmap, outBitmap, 3);
}

static void native_blackWhite(JNIEnv *env, jclass type, jobject srcBitmap, jobject outBitmap){
//    Mat srcBitmapMat,outBitmapMat;
//    bitmap_to_mat(env, srcBitmap, srcBitmapMat);
//    int height = srcBitmapMat.rows;
//    int width = srcBitmapMat.cols;
//    float fScale = Filter().getScale(height, width, 2000);  //1024值可以设定，表示长边尺寸不得超过1024，按照长边归一化
//    int window_size = (int)(width*fScale * 20 / 256);
//    window_size = window_size % 2 ? window_size : window_size + 1;
//    if((srcBitmapMat.data != outBitmapMat.data)){
//        __android_log_print(ANDROID_LOG_ERROR, "JNI", "阿达收到====>%i  ====>%f",window_size,fScale);
//    }
//    cv::cvtColor(srcBitmapMat, srcBitmapMat, cv::COLOR_BGR2GRAY);
//    outBitmapMat = Mat::zeros(srcBitmapMat.rows,srcBitmapMat.cols, srcBitmapMat.type());
//    Filter().sauvolaWithSigmoid(srcBitmapMat,outBitmapMat,window_size , 0.1);
//    mat_to_bitmap(env, outBitmapMat, outBitmap);
    native_commonProcess(env, srcBitmap, outBitmap, 1);
}

static void native_brighten(JNIEnv *env, jclass type, jobject srcBitmap, jobject outBitmap){
    native_commonProcess(env, srcBitmap, outBitmap, 4);
}

static void native_grey(JNIEnv *env, jclass type, jobject srcBitmap, jobject outBitmap){
    native_commonProcess(env, srcBitmap, outBitmap, 5);
}

static void native_solfColor(JNIEnv *env, jclass type, jobject srcBitmap, jobject outBitmap){
    native_commonProcess(env, srcBitmap, outBitmap, 2);
}


static JNINativeMethod gMethods[] = {

        {
                "nativeScan",
                "(Landroid/graphics/Bitmap;[Landroid/graphics/Point;Z)V",
                (void*)native_scan
        },

        {
                "nativeCrop",
                "(Landroid/graphics/Bitmap;[Landroid/graphics/Point;Landroid/graphics/Bitmap;)V",
                (void*)native_crop
        },

        {
                "enhance",
                "(Landroid/graphics/Bitmap;Landroid/graphics/Bitmap;)V",
                (void*)native_enhance
        },

        {
                "blackWhite",
                "(Landroid/graphics/Bitmap;Landroid/graphics/Bitmap;)V",
                (void*)native_blackWhite
        },

        {
            "brighten",
            "(Landroid/graphics/Bitmap;Landroid/graphics/Bitmap;)V",
            (void*)native_brighten
        },

        {
            "grey",
            "(Landroid/graphics/Bitmap;Landroid/graphics/Bitmap;)V",
            (void*)native_grey
        },

        {
            "solfColor",
            "(Landroid/graphics/Bitmap;Landroid/graphics/Bitmap;)V",
            (void*)native_solfColor
        }

};

extern "C"
JNIEXPORT jint JNICALL
JNI_OnLoad(JavaVM* vm, void* reserved) {
    JNIEnv *env = NULL;
    if (vm->GetEnv((void **) &env, JNI_VERSION_1_4) != JNI_OK) {
        return JNI_FALSE;
    }
    jclass classDocScanner = env->FindClass(kClassDocScanner);
    if(env -> RegisterNatives(classDocScanner, gMethods, sizeof(gMethods)/ sizeof(gMethods[0])) < 0) {
        return JNI_FALSE;
    }
    initClassInfo(env);
    return JNI_VERSION_1_4;
}

