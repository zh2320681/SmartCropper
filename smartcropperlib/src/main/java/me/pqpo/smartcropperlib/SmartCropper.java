package me.pqpo.smartcropperlib;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Point;

import java.io.IOException;

import me.pqpo.smartcropperlib.utils.CropUtils;

/**
 * Created by qiulinmin on 8/1/17.
 */

public class SmartCropper {

    private static ImageDetector sImageDetector = null;

    public static void buildImageDetector(Context context) {
        SmartCropper.buildImageDetector(context, null);
    }

    public static void buildImageDetector(Context context, String modelFile) {
        try {
            sImageDetector = new ImageDetector(context, modelFile);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     *  输入图片扫描边框顶点
     * @param srcBmp 扫描图片
     * @return 返回顶点数组，以 左上，右上，右下，左下排序
     */
    public static Point[] scan(Bitmap srcBmp, Context context) {
        if (srcBmp == null) {
            throw new IllegalArgumentException("srcBmp cannot be null");
        }
        if (sImageDetector == null) {
            try {
                sImageDetector = new ImageDetector(context);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        if (sImageDetector != null) {
            Bitmap bitmap = sImageDetector.detectImage(srcBmp);
            if (bitmap != null) {
                srcBmp = Bitmap.createScaledBitmap(bitmap, srcBmp.getWidth(), srcBmp.getHeight(), false);
            }
        }
        Point[] outPoints = new Point[4];
        nativeScan(srcBmp, outPoints, sImageDetector == null);
        return outPoints;
    }

    /**
     * 裁剪图片
     * @param srcBmp 待裁剪图片
     * @param cropPoints 裁剪区域顶点，顶点坐标以图片大小为准
     * @return 返回裁剪后的图片
     */
    public static Bitmap crop(Bitmap srcBmp, Point[] cropPoints) {
        if (srcBmp == null || cropPoints == null) {
            throw new IllegalArgumentException("srcBmp and cropPoints cannot be null");
        }
        if (cropPoints.length != 4) {
            throw new IllegalArgumentException("The length of cropPoints must be 4 , and sort by leftTop, rightTop, rightBottom, leftBottom");
        }
        Point leftTop = cropPoints[0];
        Point rightTop = cropPoints[1];
        Point rightBottom = cropPoints[2];
        Point leftBottom = cropPoints[3];

        int cropWidth = (int) ((CropUtils.getPointsDistance(leftTop, rightTop)
                + CropUtils.getPointsDistance(leftBottom, rightBottom))/2);
        int cropHeight = (int) ((CropUtils.getPointsDistance(leftTop, leftBottom)
                + CropUtils.getPointsDistance(rightTop, rightBottom))/2);

        Bitmap cropBitmap = Bitmap.createBitmap(cropWidth, cropHeight, Bitmap.Config.ARGB_8888);
        SmartCropper.nativeCrop(srcBmp, cropPoints, cropBitmap);
        return cropBitmap;
    }

    public static Bitmap filteImage(Bitmap srcBitmap,FilterType type){
        if(srcBitmap == null){
            return srcBitmap;
        }
        Bitmap cropBitmap = Bitmap.createBitmap(srcBitmap.getWidth(), srcBitmap.getHeight(), Bitmap.Config.ARGB_8888);
        if(type == FilterType.enhance) {
            enhance(srcBitmap,cropBitmap);
        } else if(type == FilterType.blackWhite) {
            blackWhite(srcBitmap,cropBitmap);
        } else if(type == FilterType.brighten) {
            brighten(srcBitmap,cropBitmap);
        } else if(type == FilterType.grey) {
            grey(srcBitmap,cropBitmap);
        } else if(type == FilterType.soft) {
            solfColor(srcBitmap,cropBitmap);
        }
        return cropBitmap;
    }


    private static native void nativeScan(Bitmap srcBitmap, Point[] outPoints, boolean canny);

    private static native void nativeCrop(Bitmap srcBitmap, Point[] points, Bitmap outBitmap);

    //增强
    private static native void enhance(Bitmap srcBitmap,Bitmap outBitmap);

    //柔和
    private static native void solfColor(Bitmap srcBitmap,Bitmap outBitmap);

    //黑白
    private static native void blackWhite(Bitmap srcBitmap,Bitmap outBitmap);

    //增亮
    private static native void brighten(Bitmap srcBitmap,Bitmap outBitmap);

    //增亮
    private static native void grey(Bitmap srcBitmap,Bitmap outBitmap);

//    public static native int brighten(String inputPath, String outputPath);

    static {
        System.loadLibrary("smart_cropper");
    }

}
