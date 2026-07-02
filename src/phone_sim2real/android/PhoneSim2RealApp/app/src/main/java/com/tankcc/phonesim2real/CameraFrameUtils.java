package com.tankcc.phonesim2real;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;

import androidx.camera.core.ImageProxy;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;

public final class CameraFrameUtils {
    private CameraFrameUtils() {}

    public static byte[] imageProxyToJpeg416(ImageProxy imageProxy, int jpegQuality) throws Exception {
        byte[] nv21 = yuv420ToNv21(imageProxy);
        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, imageProxy.getWidth(), imageProxy.getHeight(), null);
        ByteArrayOutputStream yuvJpeg = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, imageProxy.getWidth(), imageProxy.getHeight()), 90, yuvJpeg);

        Bitmap bitmap = BitmapFactory.decodeByteArray(yuvJpeg.toByteArray(), 0, yuvJpeg.size());
        if (bitmap == null) {
            throw new IllegalStateException("Failed to decode camera frame");
        }

        int rotation = imageProxy.getImageInfo().getRotationDegrees();
        if (rotation != 0) {
            Matrix matrix = new Matrix();
            matrix.postRotate(rotation);
            Bitmap rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
            bitmap.recycle();
            bitmap = rotated;
        }

        Bitmap square = centerCropSquare(bitmap);
        if (square != bitmap) {
            bitmap.recycle();
        }
        Bitmap resized = Bitmap.createScaledBitmap(square, 416, 416, true);
        if (resized != square) {
            square.recycle();
        }

        ByteArrayOutputStream out = new ByteArrayOutputStream();
        resized.compress(Bitmap.CompressFormat.JPEG, jpegQuality, out);
        resized.recycle();
        return out.toByteArray();
    }

    private static Bitmap centerCropSquare(Bitmap src) {
        int size = Math.min(src.getWidth(), src.getHeight());
        int left = (src.getWidth() - size) / 2;
        int top = (src.getHeight() - size) / 2;
        return Bitmap.createBitmap(src, left, top, size, size);
    }

    private static byte[] yuv420ToNv21(ImageProxy image) {
        int width = image.getWidth();
        int height = image.getHeight();
        int ySize = width * height;
        int uvSize = width * height / 4;
        byte[] nv21 = new byte[ySize + uvSize * 2];

        ImageProxy.PlaneProxy yPlane = image.getPlanes()[0];
        ImageProxy.PlaneProxy uPlane = image.getPlanes()[1];
        ImageProxy.PlaneProxy vPlane = image.getPlanes()[2];

        copyYPlane(yPlane.getBuffer(), yPlane.getRowStride(), width, height, nv21);
        copyVUPlanes(uPlane.getBuffer(), vPlane.getBuffer(), uPlane.getRowStride(), uPlane.getPixelStride(), width, height, nv21, ySize);
        return nv21;
    }

    private static void copyYPlane(ByteBuffer yBuffer, int rowStride, int width, int height, byte[] out) {
        ByteBuffer y = yBuffer.duplicate();
        int outIndex = 0;
        for (int row = 0; row < height; row++) {
            int rowStart = row * rowStride;
            for (int col = 0; col < width; col++) {
                out[outIndex++] = y.get(rowStart + col);
            }
        }
    }

    private static void copyVUPlanes(ByteBuffer uBuffer, ByteBuffer vBuffer, int rowStride, int pixelStride,
                                     int width, int height, byte[] out, int offset) {
        ByteBuffer u = uBuffer.duplicate();
        ByteBuffer v = vBuffer.duplicate();
        int outIndex = offset;
        int uvWidth = width / 2;
        int uvHeight = height / 2;
        for (int row = 0; row < uvHeight; row++) {
            int rowStart = row * rowStride;
            for (int col = 0; col < uvWidth; col++) {
                int index = rowStart + col * pixelStride;
                out[outIndex++] = v.get(index);
                out[outIndex++] = u.get(index);
            }
        }
    }
}
