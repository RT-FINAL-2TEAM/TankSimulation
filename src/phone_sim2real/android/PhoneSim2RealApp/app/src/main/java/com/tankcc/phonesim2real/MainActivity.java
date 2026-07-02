package com.tankcc.phonesim2real;

import android.Manifest;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.os.SystemClock;
import android.text.InputType;
import android.util.Size;
import android.view.Gravity;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.ComponentActivity;
import androidx.annotation.NonNull;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class MainActivity extends ComponentActivity implements SensorEventListener {
    private static final int REQ_CAMERA = 4126;
    private static final int INPUT_SIZE = 416;
    private static final MediaType JPEG = MediaType.parse("image/jpeg");
    private static final MediaType JSON = MediaType.parse("application/json; charset=utf-8");

    private PreviewView previewView;
    private DetectionOverlayView overlayView;
    private EditText ipEdit;
    private EditText portEdit;
    private EditText endpointEdit;
    private EditText intervalEdit;
    private Switch imuSwitch;
    private Switch injectSwitch;
    private Button startButton;
    private Button lockButton;
    private Button clearButton;
    private TextView statusText;
    private TextView telemetryText;

    private final ExecutorService cameraExecutor = Executors.newSingleThreadExecutor();
    private final OkHttpClient httpClient = new OkHttpClient.Builder()
            .connectTimeout(700, TimeUnit.MILLISECONDS)
            .readTimeout(1400, TimeUnit.MILLISECONDS)
            .writeTimeout(1400, TimeUnit.MILLISECONDS)
            .build();

    private final AtomicBoolean running = new AtomicBoolean(false);
    private final AtomicBoolean inFlight = new AtomicBoolean(false);
    private final AtomicBoolean pendingManualLock = new AtomicBoolean(false);
    private final AtomicBoolean pendingClear = new AtomicBoolean(false);
    private long lastSentAtMs = 0L;
    private long sentCount = 0L;
    private long okCount = 0L;
    private long failCount = 0L;
    private long lastLatencyMs = 0L;

    private SensorManager sensorManager;
    private Sensor accelerometer;
    private Sensor gyroscope;
    private Sensor rotationVector;
    private final Object imuLock = new Object();
    private final float[] acc = new float[]{0f, 0f, 0f};
    private final float[] gyro = new float[]{0f, 0f, 0f};
    private final float[] quat = new float[]{0f, 0f, 0f, 1f};
    private float yawRad = 0f;
    private boolean imuEnabled = true;

    private SharedPreferences prefs;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        prefs = getSharedPreferences("phone_sim2real", MODE_PRIVATE);
        setupSensors();
        buildUi();
        loadPrefs();

        if (hasCameraPermission()) {
            startCamera();
        } else {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, REQ_CAMERA);
        }
    }

    private void setupSensors() {
        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        if (sensorManager != null) {
            accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
            gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
            rotationVector = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR);
        }
    }

    private void buildUi() {
        FrameLayout root = new FrameLayout(this);
        root.setBackgroundColor(Color.rgb(7, 10, 8));

        previewView = new PreviewView(this);
        previewView.setScaleType(PreviewView.ScaleType.FILL_CENTER);
        root.addView(previewView, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
        ));

        overlayView = new DetectionOverlayView(this);
        root.addView(overlayView, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
        ));

        LinearLayout panel = new LinearLayout(this);
        panel.setOrientation(LinearLayout.VERTICAL);
        panel.setPadding(dp(12), dp(10), dp(12), dp(10));
        panel.setBackgroundResource(getResources().getIdentifier("panel_background", "drawable", getPackageName()));

        TextView title = new TextView(this);
        title.setText("TANK SIM2REAL LINK");
        title.setTextColor(Color.rgb(128, 195, 66));
        title.setTextSize(21f);
        title.setGravity(Gravity.CENTER_VERTICAL);
        title.setTypeface(null, android.graphics.Typeface.BOLD);
        panel.addView(title, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        TextView subtitle = new TextView(this);
        subtitle.setText("PHONE CAMERA / IMU → ROS2 VIRTUAL OBSTACLE");
        subtitle.setTextColor(Color.rgb(234, 243, 225));
        subtitle.setTextSize(11f);
        subtitle.setLetterSpacing(0.08f);
        panel.addView(subtitle);

        LinearLayout row1 = horizontalRow();
        ipEdit = input("Ubuntu IP", InputType.TYPE_CLASS_TEXT);
        portEdit = input("Port", InputType.TYPE_CLASS_NUMBER);
        row1.addView(wrapLabeled("IP", ipEdit), weightLp(2.0f));
        row1.addView(wrapLabeled("PORT", portEdit), weightLp(1.0f));
        panel.addView(row1);

        LinearLayout row2 = horizontalRow();
        endpointEdit = input("/phone/detect", InputType.TYPE_CLASS_TEXT);
        intervalEdit = input("350", InputType.TYPE_CLASS_NUMBER);
        row2.addView(wrapLabeled("ENDPOINT", endpointEdit), weightLp(2.0f));
        row2.addView(wrapLabeled("INTERVAL ms", intervalEdit), weightLp(1.0f));
        panel.addView(row2);

        LinearLayout row3 = horizontalRow();
        imuSwitch = new Switch(this);
        imuSwitch.setText("IMU TX");
        imuSwitch.setTextColor(Color.rgb(234, 243, 225));
        imuSwitch.setChecked(true);
        imuSwitch.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                imuEnabled = isChecked;
                updateSensorRegistration();
            }
        });
        row3.addView(imuSwitch, weightLp(1.0f));

        injectSwitch = new Switch(this);
        injectSwitch.setText("INJECT ON");
        injectSwitch.setTextColor(Color.rgb(234, 243, 225));
        injectSwitch.setChecked(true);
        injectSwitch.setOnCheckedChangeListener((buttonView, isChecked) -> {
            savePrefs();
            sendControlCommand(isChecked ? "inject_on" : "inject_off");
        });
        row3.addView(injectSwitch, weightLp(1.0f));

        startButton = new Button(this);
        startButton.setText("LINK START");
        startButton.setTextColor(Color.rgb(7, 10, 8));
        startButton.setTypeface(null, android.graphics.Typeface.BOLD);
        startButton.setBackgroundResource(getResources().getIdentifier("button_primary", "drawable", getPackageName()));
        startButton.setOnClickListener(v -> toggleRun());
        row3.addView(startButton, weightLp(1.0f));
        panel.addView(row3);

        LinearLayout row4 = horizontalRow();
        lockButton = new Button(this);
        lockButton.setText("LOCK OBSTACLE");
        lockButton.setTextColor(Color.rgb(7, 10, 8));
        lockButton.setTypeface(null, android.graphics.Typeface.BOLD);
        lockButton.setBackgroundResource(getResources().getIdentifier("button_primary", "drawable", getPackageName()));
        lockButton.setOnClickListener(v -> {
            pendingManualLock.set(true);
            sendControlCommand("lock");
            setStatus("LOCK REQUESTED", false);
        });
        row4.addView(lockButton, weightLp(1.0f));

        clearButton = new Button(this);
        clearButton.setText("CLEAR OBSTACLE");
        clearButton.setTextColor(Color.rgb(7, 10, 8));
        clearButton.setTypeface(null, android.graphics.Typeface.BOLD);
        clearButton.setBackgroundResource(getResources().getIdentifier("button_primary", "drawable", getPackageName()));
        clearButton.setOnClickListener(v -> {
            pendingClear.set(true);
            runOnUiThread(() -> overlayView.setDetections(new ArrayList<>()));
            sendControlCommand("clear");
            setStatus("CLEAR REQUESTED", false);
        });
        row4.addView(clearButton, weightLp(1.0f));
        panel.addView(row4);

        statusText = new TextView(this);
        statusText.setText("STANDBY");
        statusText.setTextColor(Color.rgb(255, 183, 77));
        statusText.setTextSize(13f);
        statusText.setPadding(0, dp(6), 0, 0);
        panel.addView(statusText);

        telemetryText = new TextView(this);
        telemetryText.setText("416x416 JPEG | YOLO gateway idle");
        telemetryText.setTextColor(Color.rgb(234, 243, 225));
        telemetryText.setTextSize(12f);
        panel.addView(telemetryText);

        ScrollView scroll = new ScrollView(this);
        scroll.setFillViewport(false);
        scroll.addView(panel);

        FrameLayout.LayoutParams panelLp = new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        );
        panelLp.gravity = Gravity.BOTTOM;
        panelLp.leftMargin = dp(10);
        panelLp.rightMargin = dp(10);
        panelLp.bottomMargin = dp(12);
        root.addView(scroll, panelLp);

        setContentView(root);
    }

    private LinearLayout horizontalRow() {
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);
        row.setPadding(0, dp(8), 0, 0);
        return row;
    }

    private LinearLayout wrapLabeled(String label, EditText editText) {
        LinearLayout wrap = new LinearLayout(this);
        wrap.setOrientation(LinearLayout.VERTICAL);
        wrap.setPadding(0, 0, dp(8), 0);
        TextView tv = new TextView(this);
        tv.setText(label);
        tv.setTextColor(Color.rgb(128, 195, 66));
        tv.setTextSize(10f);
        tv.setTypeface(null, android.graphics.Typeface.BOLD);
        wrap.addView(tv);
        wrap.addView(editText);
        return wrap;
    }

    private EditText input(String hint, int inputType) {
        EditText e = new EditText(this);
        e.setHint(hint);
        e.setSingleLine(true);
        e.setInputType(inputType);
        e.setTextColor(Color.rgb(234, 243, 225));
        e.setHintTextColor(Color.rgb(90, 110, 88));
        e.setTextSize(14f);
        e.setBackgroundResource(getResources().getIdentifier("input_background", "drawable", getPackageName()));
        return e;
    }

    private LinearLayout.LayoutParams weightLp(float weight) {
        return new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, weight);
    }

    private void loadPrefs() {
        ipEdit.setText(prefs.getString("ip", "192.168.0.32"));
        portEdit.setText(prefs.getString("port", "5002"));
        endpointEdit.setText(prefs.getString("endpoint", "/phone/detect"));
        intervalEdit.setText(prefs.getString("interval", "350"));
        imuSwitch.setChecked(prefs.getBoolean("imu", true));
        injectSwitch.setChecked(prefs.getBoolean("inject", true));
        imuEnabled = imuSwitch.isChecked();
        updateSensorRegistration();
    }

    private void savePrefs() {
        prefs.edit()
                .putString("ip", ipEdit.getText().toString().trim())
                .putString("port", portEdit.getText().toString().trim())
                .putString("endpoint", endpointEdit.getText().toString().trim())
                .putString("interval", intervalEdit.getText().toString().trim())
                .putBoolean("imu", imuSwitch.isChecked())
                .putBoolean("inject", injectSwitch != null && injectSwitch.isChecked())
                .apply();
    }

    private void toggleRun() {
        if (running.get()) {
            running.set(false);
            startButton.setText("LINK START");
            setStatus("STANDBY", false);
            return;
        }
        if (ipEdit.getText().toString().trim().isEmpty()) {
            Toast.makeText(this, "Ubuntu IP를 입력하세요", Toast.LENGTH_SHORT).show();
            return;
        }
        savePrefs();
        sentCount = 0;
        okCount = 0;
        failCount = 0;
        lastLatencyMs = 0;
        running.set(true);
        startButton.setText("LINK STOP");
        setStatus("LINK ACTIVE → " + getServerUrl(), false);
    }

    private boolean hasCameraPermission() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQ_CAMERA && grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            Toast.makeText(this, "Camera permission is required", Toast.LENGTH_LONG).show();
        }
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                ImageAnalysis analysis = new ImageAnalysis.Builder()
                        .setTargetResolution(new Size(INPUT_SIZE, INPUT_SIZE))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();
                analysis.setAnalyzer(cameraExecutor, this::analyzeFrame);

                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis);
                setStatus("CAMERA READY", false);
            } catch (Exception e) {
                setStatus("CAMERA ERROR: " + e.getMessage(), true);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void analyzeFrame(ImageProxy image) {
        try {
            if (!running.get()) {
                return;
            }
            long now = SystemClock.elapsedRealtime();
            int intervalMs = parseInt(intervalEdit.getText().toString(), 350);
            if (now - lastSentAtMs < intervalMs || inFlight.get()) {
                return;
            }
            lastSentAtMs = now;
            inFlight.set(true);
            byte[] jpeg = CameraFrameUtils.imageProxyToJpeg416(image, 82);
            sendFrame(jpeg, now);
        } catch (Exception e) {
            failCount++;
            inFlight.set(false);
            setStatus("FRAME ERROR: " + e.getMessage(), true);
        } finally {
            image.close();
        }
    }

    private int parseInt(String s, int fallback) {
        try {
            return Integer.parseInt(s.trim());
        } catch (Exception ignored) {
            return fallback;
        }
    }

    private String getServerUrl() {
        String ip = ipEdit.getText().toString().trim();
        String port = portEdit.getText().toString().trim();
        String endpoint = endpointEdit.getText().toString().trim();
        if (!endpoint.startsWith("/")) {
            endpoint = "/" + endpoint;
        }
        return "http://" + ip + ":" + port + endpoint;
    }

    private String getControlUrl() {
        String ip = ipEdit.getText().toString().trim();
        String port = portEdit.getText().toString().trim();
        return "http://" + ip + ":" + port + "/phone/control";
    }

    private void sendControlCommand(String command) {
        try {
            JSONObject payload = new JSONObject();
            payload.put("source", "android_phone_sim2real");
            payload.put("command", command);
            payload.put("inject_enabled", injectSwitch == null || injectSwitch.isChecked());
            payload.put("manual_lock_request", "lock".equals(command));
            payload.put("clear_request", "clear".equals(command));
            payload.put("timestamp_ms", System.currentTimeMillis());
            RequestBody body = RequestBody.create(payload.toString(), JSON);
            Request request = new Request.Builder().url(getControlUrl()).post(body).build();
            httpClient.newCall(request).enqueue(new Callback() {
                @Override public void onFailure(@NonNull Call call, @NonNull IOException e) {
                    setStatus("CTRL FAIL: " + e.getMessage(), true);
                }
                @Override public void onResponse(@NonNull Call call, @NonNull Response response) throws IOException {
                    String text = response.body() != null ? response.body().string() : "";
                    if (!response.isSuccessful()) {
                        setStatus("CTRL HTTP " + response.code() + ": " + trim(text, 80), true);
                    } else {
                        setStatus("CTRL OK: " + command, false);
                    }
                }
            });
        } catch (Exception e) {
            setStatus("CTRL ERROR: " + e.getMessage(), true);
        }
    }

    private void sendFrame(byte[] jpeg, long frameTimeMs) throws Exception {
        sentCount++;
        JSONObject meta = buildMetaJson();
        MultipartBody body = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("image", "phone_416.jpg", RequestBody.create(jpeg, JPEG))
                .addFormDataPart("meta", null, RequestBody.create(meta.toString(), JSON))
                .addFormDataPart("metadata", null, RequestBody.create(meta.toString(), JSON))
                .addFormDataPart("width", "416")
                .addFormDataPart("height", "416")
                .addFormDataPart("source", "android_phone_sim2real")
                .build();

        Request request = new Request.Builder()
                .url(getServerUrl())
                .post(body)
                .build();

        httpClient.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(@NonNull Call call, @NonNull IOException e) {
                failCount++;
                inFlight.set(false);
                setStatus("TX FAIL: " + e.getMessage(), true);
                updateTelemetry(0);
            }

            @Override
            public void onResponse(@NonNull Call call, @NonNull Response response) throws IOException {
                long latency = SystemClock.elapsedRealtime() - frameTimeMs;
                lastLatencyMs = latency;
                String text = response.body() != null ? response.body().string() : "";
                if (!response.isSuccessful()) {
                    failCount++;
                    inFlight.set(false);
                    setStatus("HTTP " + response.code() + ": " + trim(text, 80), true);
                    updateTelemetry(0);
                    return;
                }
                okCount++;
                List<Detection> detections = parseDetections(text);
                runOnUiThread(() -> overlayView.setDetections(detections));
                inFlight.set(false);
                setStatus("LINK OK  latency=" + latency + "ms", false);
                updateTelemetry(detections.size());
            }
        });
    }

    private JSONObject buildMetaJson() throws Exception {
        JSONObject root = new JSONObject();
        root.put("source", "android_phone_sim2real");
        root.put("image_width", INPUT_SIZE);
        root.put("image_height", INPUT_SIZE);
        root.put("timestamp_ms", System.currentTimeMillis());
        root.put("imu_enabled", imuEnabled);
        root.put("inject_enabled", injectSwitch == null || injectSwitch.isChecked());
        boolean lockReq = pendingManualLock.getAndSet(false);
        boolean clearReq = pendingClear.getAndSet(false);
        root.put("manual_lock_request", lockReq);
        root.put("clear_request", clearReq);
        if (lockReq) root.put("command", "lock");
        if (clearReq) root.put("command", "clear");
        if (imuEnabled) {
            JSONObject imu = new JSONObject();
            synchronized (imuLock) {
                imu.put("accel", new JSONArray(new double[]{acc[0], acc[1], acc[2]}));
                imu.put("gyro", new JSONArray(new double[]{gyro[0], gyro[1], gyro[2]}));
                imu.put("quat_xyzw", new JSONArray(new double[]{quat[0], quat[1], quat[2], quat[3]}));
                imu.put("yaw_rad", yawRad);
            }
            root.put("imu", imu);
        }
        return root;
    }

    private List<Detection> parseDetections(String responseText) {
        List<Detection> result = new ArrayList<>();
        try {
            JSONObject root = new JSONObject(responseText);
            JSONArray arr = root.optJSONArray("detections");
            if (arr == null) arr = root.optJSONArray("objects");
            if (arr == null && root.has("result")) {
                JSONObject nested = root.optJSONObject("result");
                if (nested != null) {
                    arr = nested.optJSONArray("detections");
                    if (arr == null) arr = nested.optJSONArray("objects");
                }
            }
            if (arr == null) return result;

            for (int i = 0; i < arr.length(); i++) {
                JSONObject o = arr.optJSONObject(i);
                if (o == null) continue;
                String label = firstNonEmpty(
                        o.optString("className", ""),
                    o.optString("class_name", ""),
                        o.optString("label", ""),
                        o.optString("name", ""),
                        o.optString("class", "object")
                );
                float conf = (float) (o.has("confidence") ? o.optDouble("confidence", 0.0) : o.optDouble("conf", 0.0));
                float[] box = readBox(o);
                if (box == null) continue;
                Float distance = null;
                if (o.has("distance_m")) distance = (float) o.optDouble("distance_m", 0.0);
                else if (o.has("depth_m")) distance = (float) o.optDouble("depth_m", 0.0);
                result.add(new Detection(label, conf, box[0], box[1], box[2], box[3], distance));
            }
        } catch (Exception ignored) {
            // Keep overlay unchanged on malformed response.
        }
        return result;
    }

    private String firstNonEmpty(String... values) {
        for (String v : values) {
            if (v != null && !v.trim().isEmpty()) return v.trim();
        }
        return "object";
    }

    private float[] readBox(JSONObject o) {
        try {
            JSONArray b = o.optJSONArray("bbox");
            if (b == null) b = o.optJSONArray("box");
            if (b == null) b = o.optJSONArray("xyxy");
            if (b != null && b.length() >= 4) {
                float x1 = (float) b.optDouble(0);
                float y1 = (float) b.optDouble(1);
                float x2 = (float) b.optDouble(2);
                float y2 = (float) b.optDouble(3);
                return normalizeBox(x1, y1, x2, y2);
            }
            if (o.has("x1") && o.has("y1") && o.has("x2") && o.has("y2")) {
                return normalizeBox((float) o.optDouble("x1"), (float) o.optDouble("y1"),
                        (float) o.optDouble("x2"), (float) o.optDouble("y2"));
            }
        } catch (Exception ignored) {}
        return null;
    }

    private float[] normalizeBox(float x1, float y1, float x2, float y2) {
        float max = Math.max(Math.max(Math.abs(x1), Math.abs(y1)), Math.max(Math.abs(x2), Math.abs(y2)));
        if (max <= 1.5f) {
            x1 *= INPUT_SIZE; y1 *= INPUT_SIZE; x2 *= INPUT_SIZE; y2 *= INPUT_SIZE;
        }
        x1 = clamp(x1, 0, INPUT_SIZE);
        y1 = clamp(y1, 0, INPUT_SIZE);
        x2 = clamp(x2, 0, INPUT_SIZE);
        y2 = clamp(y2, 0, INPUT_SIZE);
        return new float[]{x1, y1, x2, y2};
    }

    private float clamp(float v, float lo, float hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    private String trim(String s, int maxLen) {
        if (s == null) return "";
        return s.length() <= maxLen ? s : s.substring(0, maxLen) + "...";
    }

    private void setStatus(String text, boolean error) {
        runOnUiThread(() -> {
            statusText.setText(text);
            statusText.setTextColor(error ? Color.rgb(239, 83, 80) : Color.rgb(128, 195, 66));
        });
    }

    private void updateTelemetry(int detCount) {
        runOnUiThread(() -> telemetryText.setText(String.format(Locale.US,
                "TX %d | OK %d | FAIL %d | DET %d | %dms | 416x416",
                sentCount, okCount, failCount, detCount, lastLatencyMs)));
    }

    private void updateSensorRegistration() {
        if (sensorManager == null) return;
        sensorManager.unregisterListener(this);
        if (!imuEnabled) return;
        if (accelerometer != null) sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_GAME);
        if (gyroscope != null) sensorManager.registerListener(this, gyroscope, SensorManager.SENSOR_DELAY_GAME);
        if (rotationVector != null) sensorManager.registerListener(this, rotationVector, SensorManager.SENSOR_DELAY_GAME);
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        synchronized (imuLock) {
            if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
                System.arraycopy(event.values, 0, acc, 0, Math.min(3, event.values.length));
            } else if (event.sensor.getType() == Sensor.TYPE_GYROSCOPE) {
                System.arraycopy(event.values, 0, gyro, 0, Math.min(3, event.values.length));
            } else if (event.sensor.getType() == Sensor.TYPE_ROTATION_VECTOR) {
                float[] q = new float[4];
                SensorManager.getQuaternionFromVector(q, event.values);
                // Android returns w,x,y,z. Publish xyzw for ROS-friendly convention.
                quat[0] = q[1];
                quat[1] = q[2];
                quat[2] = q[3];
                quat[3] = q[0];
                float[] rot = new float[9];
                float[] ori = new float[3];
                SensorManager.getRotationMatrixFromVector(rot, event.values);
                SensorManager.getOrientation(rot, ori);
                yawRad = ori[0];
            }
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {}

    @Override
    protected void onResume() {
        super.onResume();
        updateSensorRegistration();
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (sensorManager != null) sensorManager.unregisterListener(this);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        running.set(false);
        cameraExecutor.shutdown();
    }

    private int dp(int value) {
        return (int) (value * getResources().getDisplayMetrics().density + 0.5f);
    }
}
