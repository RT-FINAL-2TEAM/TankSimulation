package com.tankcc.phonesim2real;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class DetectionOverlayView extends View {
    private final Paint boxPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint textBgPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint crossPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final List<Detection> detections = new ArrayList<>();

    public DetectionOverlayView(Context context) {
        super(context);
        init();
    }

    public DetectionOverlayView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(4.0f);
        boxPaint.setColor(Color.rgb(128, 195, 66));

        textPaint.setColor(Color.rgb(234, 243, 225));
        textPaint.setTextSize(30.0f);
        textPaint.setFakeBoldText(true);

        textBgPaint.setStyle(Paint.Style.FILL);
        textBgPaint.setColor(Color.argb(210, 7, 10, 8));

        crossPaint.setColor(Color.argb(190, 128, 195, 66));
        crossPaint.setStrokeWidth(2.0f);
    }

    public void setDetections(List<Detection> newDetections) {
        synchronized (detections) {
            detections.clear();
            detections.addAll(newDetections);
        }
        postInvalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        int w = getWidth();
        int h = getHeight();

        // Tactical center reticle.
        float cx = w / 2.0f;
        float cy = h / 2.0f;
        canvas.drawLine(cx - 30, cy, cx - 8, cy, crossPaint);
        canvas.drawLine(cx + 8, cy, cx + 30, cy, crossPaint);
        canvas.drawLine(cx, cy - 30, cx, cy - 8, crossPaint);
        canvas.drawLine(cx, cy + 8, cx, cy + 30, crossPaint);

        List<Detection> snapshot;
        synchronized (detections) {
            snapshot = new ArrayList<>(detections);
        }

        float sx = w / 416.0f;
        float sy = h / 416.0f;
        for (Detection d : snapshot) {
            RectF r = new RectF(d.x1 * sx, d.y1 * sy, d.x2 * sx, d.y2 * sy);
            canvas.drawRect(r, boxPaint);

            String text = String.format(Locale.US, "%s %.2f", d.label, d.confidence);
            if (d.distanceM != null) {
                text += String.format(Locale.US, "  %.1fm", d.distanceM);
            }
            float textWidth = textPaint.measureText(text);
            float top = Math.max(0, r.top - 38);
            canvas.drawRoundRect(new RectF(r.left, top, r.left + textWidth + 18, top + 36), 8, 8, textBgPaint);
            canvas.drawText(text, r.left + 9, top + 27, textPaint);
        }
    }
}
