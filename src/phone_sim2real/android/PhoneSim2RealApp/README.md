# PhoneSim2RealApp

Android Studio에서 `~/tankcc/src/phone_sim2real/android/PhoneSim2RealApp` 폴더를 직접 열어 실행합니다.

## 기능

- CameraX 후면 카메라
- 416x416 JPEG 전송
- Ubuntu IP / Port / Endpoint 변경 가능
- IMU TX ON/OFF
- INJECT ON/OFF: ROS 가상 장애물 주입 제어
- LOCK OBSTACLE: 현재 탐지된 객체를 map 좌표에 고정
- CLEAR OBSTACLE: phone_sim2real active obstacle 제거
- YOLO bbox overlay 표시

## 서버 설정

```text
Ubuntu IP : 예) 192.168.0.32
Port      : 5002
Endpoint  : /phone/detect
Interval  : 300~500 ms
```

제어 명령은 자동으로 `/phone/control`로 전송됩니다.

## SDK 오류 해결

`SDK location not found`가 뜨면:

```bash
cd ~/tankcc/src/phone_sim2real/android/PhoneSim2RealApp
./fix_android_sdk.sh
```

또는 `local.properties`에 직접 지정합니다.

```properties
sdk.dir=/home/tankcc/Android/Sdk
```
