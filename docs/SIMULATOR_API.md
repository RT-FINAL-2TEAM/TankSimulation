# 3.2 API Docs

## 3.2.1  객체 탐지

### <mark style="color:green;">POST</mark>    /detect

시뮬레이터에서 이미지를 수신하고, 객체 감지를 수행하고, 필터링된 결과를 반환합니다.

**Headers**

| Name         | Value              |
| ------------ | ------------------ |
| Content-Type | `application/json` |

**Body**

| Name  | Type  | Description    |
| ----- | ----- | -------------- |
| image | image | 터랫 뷰 화면 이미지 파일 |

**Response**

{% tabs %}
{% tab title="200" %}

```json
[
  {
    "className": "person",
    "bbox": [10, 10, 50, 50],
    "confidence": 0.85,
    "color": "#00FF00",
    "filled": False,
    "updateBoxWhileMoving": False
  },
  ...
]
```

{% endtab %}

{% tab title="400" %}

```json
{
  "error": "Invalid request"
}
```

{% endtab %}
{% endtabs %}

key 별 정보

<table><thead><tr><th width="76.33331298828125">번호</th><th>입력값</th><th>전차 기동</th></tr></thead><tbody><tr><td>1</td><td>className</td><td>Bound Box에 표시할 class명</td></tr><tr><td>2</td><td>bbox</td><td>Bound Box의 좌표 [ x1, y1, x2, y2]</td></tr><tr><td>3</td><td>confidence</td><td>신뢰도</td></tr><tr><td>4</td><td>color</td><td>Bound Box 색상</td></tr><tr><td>5</td><td>filled</td><td>Bound Box의 색상 채우기 여부<br>( False의 경우 외곽선 박스 표시)</td></tr><tr><td>6</td><td>updateBoxWhileMoving</td><td>전차 이동시 Bound Box보정 기능 사용 유무</td></tr></tbody></table>

***

## 3.2.2 시뮬레이터 정보 수신 및 명령 실행

### <mark style="color:green;">POST</mark>    /info

시뮬레이터에서 전송된 로그데이터를 수신합니다.

**Headers**

| Name         | Value              |
| ------------ | ------------------ |
| Content-Type | `application/json` |

**Body**

{% tabs %}
{% tab title="JSON" %}

```json
{
   "time":3.9867117404937744,
   "distance":252.40330505371094,
   "playerPos":{
      "x":60.0,
      "y":8.002116203308105,
      "z":27.229999542236328
   },
   "playerSpeed":0.2987206280231476,
   "playerHealth":100.0,
   "playerTurretX":2.9181717042803257e-09,
   "playerTurretY":0.0,
   "playerBodyX":2.9181717042803257e-09,
   "playerBodyY":-3.256887814995224e-11,
   "playerBodyZ":-9.338149453697042e-08,
   "enemyPos":{
      "x":59.22119903564453,
      "y":8.869277954101562,
      "z":279.630615234375
   },
   "enemySpeed":19.135791778564453,
   "enemyHealth":100.0,
   "enemyTurretX":179.5301055908203,
   "enemyTurretY":-2.180685520172119,
   "enemyBodyX":179.5301055908203,
   "enemyBodyY":2.180685520172119,
   "enemyBodyZ":2.2678720951080322,
   "lidarOrigin":{
      "x":60.00704574584961,
      "y":8.0,
      "z":27.579877853393555
   },
   "lidarRotation": {
      "x": 0.0,
      "y": 135.2,
      "z": 0.0
   }
   "lidarPoints":[
      {
         "angle":0.0,
         "verticalAngle":22.5,
         "distance":30.0,
         "position":{
            "x":60.00704574584961,
            "y":-3.4805030822753906,
            "z":55.2962646484375
         },
         "isDetected":false,
         "channelIndex":1
      },
      {
         "angle":1.0,
         "verticalAngle":22.5,
         "distance":30.0,
         "position":{
            "x":60.49076461791992,
            "y":-3.480502128601074,
            "z":55.29204559326172
         },
         "isDetected":false,
         "channelIndex":1
      },
      {
         "angle":2.0,
         "verticalAngle":22.5,
         "distance":30.0,
         "position":{
            "x":60.974334716796875,
            "y":-3.4805030822753906,
            "z":55.279380798339844
         },
         "isDetected":false,
         "channelIndex":1
      },
      {
         "angle":3.0,
         "verticalAngle":22.5,
         "distance":30.0,
         "position":{
            "x":61.457611083984375,
            "y":-3.4805030822753906,
            "z":55.258277893066406
         },
         "isDetected":false,
         "channelIndex":1
      },
	  
	  ...
	  
      {
         "angle":359.0,
         "verticalAngle":-22.499998092651367,
         "distance":30.0,
         "position":{
            "x":59.52332305908203,
            "y":19.480501174926758,
            "z":55.29204559326172
         },
         "isDetected":false,
         "channelIndex":8
      }
   ]
}
```

{% endtab %}
{% endtabs %}

**Response**

{% tabs %}
{% tab title="200" %}

```json
{
  "status": "success",
  "message": "Data received", 
  "control": "pause"
}
```

{% endtab %}

{% tab title="400" %}

```json
{
  "error": "Invalid request"
}
```

{% endtab %}
{% endtabs %}

control의 범위

<table><thead><tr><th width="76.33331298828125">번호</th><th>입력값</th><th>전차 기동</th></tr></thead><tbody><tr><td>1</td><td>pause</td><td>일시정지</td></tr><tr><td>2</td><td>reset</td><td>초기화</td></tr></tbody></table>

***

{% hint style="info" %}
아래 EndPoint는 v2.2.3 부터 /get\_action 과 통합되어 사용되지 않습니다.
{% endhint %}

## ~~3.2.3 현재 위치 전송~~

### ~~<mark style="color:green;">POST</mark>    /update\_position~~

~~전차의 현재 위치를 End Point에 전송합니다.~~

**Headers**

| Name         | Value              |
| ------------ | ------------------ |
| Content-Type | `application/json` |

**Body**

{% tabs %}
{% tab title="JSON" %}

```json
{
  "position": "123.4,56.7,89.0"
}
```

{% endtab %}
{% endtabs %}

**Response**

{% tabs %}
{% tab title="200" %}

```json
{
  "status": "OK"
}
```

{% endtab %}

{% tab title="400" %}

```json
{
  "status": "ERROR",
  "message": "Missing position data"
}
```

{% endtab %}
{% endtabs %}

***

{% hint style="info" %}
아래 EndPoint는 v2.2.3 부터 /get\_action 과 통합되어 사용되지 않습니다.
{% endhint %}

## ~~3.2.4 이동 명령 요청~~

### ~~<mark style="color:orange;">GET</mark>    /get\_move~~

~~전차에 보낼 다음 이동 명령을 제공합니다.~~

**Headers**

| Name         | Value              |
| ------------ | ------------------ |
| Content-Type | `application/json` |

**Response**

{% tabs %}
{% tab title="200" %}

```json
{
  "move": "W", 
  "weight": 1.0
}
```

{% endtab %}

{% tab title="400" %}

```json
{
  "status": "ERROR"
}
```

{% endtab %}
{% endtabs %}

move값의 범위

<table><thead><tr><th width="76.33331298828125">번호</th><th>입력값</th><th>전차 기동</th></tr></thead><tbody><tr><td>1</td><td>W</td><td>전진</td></tr><tr><td>2</td><td>S</td><td>후진</td></tr><tr><td>3</td><td>A</td><td>좌로 회전</td></tr><tr><td>4</td><td>D</td><td>우로 회전</td></tr><tr><td>5</td><td>STOP</td><td>정지</td></tr></tbody></table>

key 정보

<table><thead><tr><th width="76.33331298828125">번호</th><th>key</th><th>value</th></tr></thead><tbody><tr><td>1</td><td>weight</td><td>이동에 대한 가중치(0.1 ~ 1.0)<br>- 기존 물리량 * weight</td></tr></tbody></table>

***

## 3.2.5 액션 명령 요청

### <mark style="color:green;">POST</mark>    /get\_action

전차의 현재 위치와 포탑의 정보를 취득하고 이동과 포탑 제어를 위한 다음 액션 명령을 제공합니다.

**Headers**

| Name         | Value              |
| ------------ | ------------------ |
| Content-Type | `application/json` |

**Body**

{% tabs %}
{% tab title="JSON" %}

```json
{
  "position": { "x": 57.35, "y": 0.00, "z": 210.85 },
  "turret": { "x": 45.00, "y": -5.50 }
}
```

{% endtab %}
{% endtabs %}

**Response**

{% tabs %}
{% tab title="200" %}

```json
{
  "moveWS": {"command": "W", "weight": 0.8},
  "moveAD": {"command": "A", "weight": 0.6},
  "turretQE": {"command": "E", "weight": 0.9},
  "turretRF": {"command": "R", "weight": 0.2},
  "fire": True
}
```

{% endtab %}

{% tab title="400" %}

```json
{
  "status": "ERROR",
}
```

{% endtab %}
{% endtabs %}

key별값의 범위

<table><thead><tr><th width="76.33331298828125">번호</th><th>key</th><th>입력값</th><th>전차 기동</th></tr></thead><tbody><tr><td>1</td><td>moveWS</td><td>W</td><td>전진</td></tr><tr><td>2</td><td>moveWS</td><td>S</td><td>후진</td></tr><tr><td>3</td><td>moveWS</td><td>STOP</td><td>정지</td></tr><tr><td>4</td><td>moveAD</td><td>A</td><td>전차좌로 회전</td></tr><tr><td>5</td><td>moveAD</td><td>D</td><td>전차 우로 회전</td></tr><tr><td>6</td><td>turretQE</td><td>Q</td><td>포탑좌로 회전</td></tr><tr><td>7</td><td>turretQE</td><td>E</td><td>포탑우로 회전</td></tr><tr><td>8</td><td>turretRF</td><td>R</td><td>포각 상승</td></tr><tr><td>9</td><td>turretRF</td><td>F</td><td>포각 하강</td></tr><tr><td>10</td><td>fire</td><td>True</td><td>포탄 발사</td></tr></tbody></table>

공통key 정보

<table><thead><tr><th width="76.33331298828125">번호</th><th>key</th><th>value</th></tr></thead><tbody><tr><td>1</td><td>weight</td><td>액션에 대한 가중치(0.1 ~ 1.0)<br>- 기존 물리량 * weight</td></tr></tbody></table>

{% hint style="info" %}
Enemy용 End Point에서도 동일하게 기능을 구현하여 사용 가능합니다.
{% endhint %}

***

## 3.2.6 포탄  충돌 정보 수신

### <mark style="color:green;">POST</mark>    /update\_bullet

포탄이 충돌한 위치 및 대상 정보를 End Point에 전달합니다.

**Headers**

| Name         | Value              |
| ------------ | ------------------ |
| Content-Type | `application/json` |

**Body**

{% tabs %}
{% tab title="JSON" %}

```json
{
  "x": 12.3,
  "y": 0.5,
  "z": 45.6,
  "hit": "terrain"
}
```

{% endtab %}
{% endtabs %}

**Response**

{% tabs %}
{% tab title="200" %}

```json
{
  "status": "OK",
  "message": "Bullet impact data received"
}
```

{% endtab %}

{% tab title="400" %}

```json
{
  "status": "ERROR",
}
```

{% endtab %}
{% endtabs %}

{% hint style="info" %}
Enemy용 End Point에서도 동일하게 기능을 구현하여 사용 가능합니다.
{% endhint %}

***

## 3.2.7 목적지 설정

### <mark style="color:green;">POST</mark>    /set\_destination

Tracking Edit Mode 에서 설정한 목적지를 전달합니다.

**Headers**

| Name         | Value              |
| ------------ | ------------------ |
| Content-Type | `application/json` |

**Body**

{% tabs %}
{% tab title="JSON" %}

```json
{
  "destination": "100.0,0.0,250.0"
}
```

{% endtab %}
{% endtabs %}

**Response**

{% tabs %}
{% tab title="200" %}

```json
{
  "status": "OK"
}
```

{% endtab %}

{% tab title="400" %}

```json
{
  "status": "ERROR",
  "message": "Missing destination data"
}
```

{% endtab %}
{% endtabs %}

***

## 3.2.8 장애물 정보 전송

### <mark style="color:green;">POST</mark>    /update\_obstacle

시뮬레이터 환경에 추가된 Obstacle 정보를 End Point에 전달합니다

**Headers**

| Name         | Value              |
| ------------ | ------------------ |
| Content-Type | `application/json` |

**Body**

{% tabs %}
{% tab title="JSON" %}

```json
{
  "obstacles": [
    {
      "x_min": 70.57522583007812,
      "x_max": 73.57522583007812,
      "z_min": 105.32206726074219,
      "z_max": 111.32206726074219
    },
    {
      "x_min": 92.5589828491211,
      "x_max": 95.5589828491211,
      "z_min": 73.52107238769531,
      "z_max": 79.52107238769531
    }
  ]
}
```

{% endtab %}
{% endtabs %}

**Response**

{% tabs %}
{% tab title="200" %}

```json
{
  "status": "OK"
}
```

{% endtab %}

{% tab title="400" %}

```json
{
  "status": "ERROR",
}
```

{% endtab %}
{% endtabs %}

***

## 3.2.9  초기설정

### <mark style="color:orange;">GET</mark>   /init

Unity 씬이 시작될 때, 시뮬레이션 초기화 정보(탱크 위치, 진행/중지 여부 등)를 설정합니다.

**Headers**

| Name         | Value              |
| ------------ | ------------------ |
| Content-Type | `application/json` |

**Response**

{% tabs %}
{% tab title="200" %}

```json
{
  "startMode": "start",
  "blStartX": 60,
  "blStartY": 10,
  "blStartZ": 27.23,
  "rdStartX": 59,
  "rdStartY": 10,
  "rdStartZ": 280,
  "trackingMode": True,
  "detectMode": False,
  "logMode": True,
  "stereoCameraMode": False,
  "enemyTracking": True,
  "saveSnapshot": False,
  "saveLog": True,
  "saveLidarData": False,
  "destoryObstaclesOnHit" : True
}
```

{% endtab %}
{% endtabs %}

startMode의 범위

<table><thead><tr><th width="76.33331298828125">번호</th><th>입력값</th><th>전차 기동</th></tr></thead><tbody><tr><td>1</td><td>start</td><td>에피소드 시작 시 진행상태</td></tr><tr><td>2</td><td>pause</td><td>에피소드 시작 시 중지상태</td></tr></tbody></table>

key 정보

<table><thead><tr><th width="76.33331298828125">번호</th><th width="149.66668701171875">type</th><th>key</th><th>value</th></tr></thead><tbody><tr><td>1</td><td>float</td><td>blStartX</td><td>아군 전차 시작 위치 X 좌표</td></tr><tr><td>2</td><td>float</td><td>blStartY</td><td>아군 전차 시작 위치 Y 좌표</td></tr><tr><td>3</td><td>float</td><td>blStartZ</td><td>아군 전차 시작 위치 Z좌표</td></tr><tr><td>4</td><td>float</td><td>rdStartX</td><td>적전차 시작 위치 X 좌표</td></tr><tr><td>5</td><td>float</td><td>rdStartY</td><td>적 전차 시작 위치 Y 좌표</td></tr><tr><td>6</td><td>float</td><td>rdStartZ</td><td>적 전차 시작 위치 Z좌표</td></tr><tr><td>7</td><td>bool</td><td>trackingMode</td><td>Tracking Mode 설정 여부</td></tr><tr><td>8</td><td>bool</td><td>detactMode</td><td>Detact Mode 설정 여부</td></tr><tr><td>9</td><td>bool</td><td>logMode</td><td>Log Mode 설정 여부</td></tr><tr><td>10</td><td>bool</td><td>stereoCameraMode</td><td>Stereo Camera Mode 설정 여부</td></tr><tr><td>11</td><td>bool</td><td>enemyTracking</td><td>Enemy Tracking 설정 여부</td></tr><tr><td>12</td><td>bool</td><td>saveSnapshot</td><td>Save Snapshot 설정 여부</td></tr><tr><td>13</td><td>bool</td><td>saveStereoCamera</td><td>Save Stereo Camera 설정여부</td></tr><tr><td>14</td><td>bool</td><td>saveLog</td><td>Save Log 설정 여부</td></tr><tr><td>15</td><td>bool</td><td>saveLidarData</td><td>Save LiDAR Data 설정 여부</td></tr><tr><td>16</td><td>int</td><td>lux</td><td>조명값 설정</td></tr><tr><td>17</td><td>bool</td><td>destoryObstaclesOnHit</td><td>destory Obstacles On Hit 설정 여부</td></tr></tbody></table>

***

## 3.2.10  시작 명령

### <mark style="color:orange;">GET</mark>   /start

에피소드가 중지상태일 때 1초에 한번씩 요청하면서 시작명령을 처리합니다.

**Headers**

| Name         | Value              |
| ------------ | ------------------ |
| Content-Type | `application/json` |

**Response**

{% tabs %}
{% tab title="200" %}

```json
{
  "control": "start"
}
```

{% endtab %}
{% endtabs %}

control의 범위

<table><thead><tr><th width="76.33331298828125">번호</th><th>입력값</th><th>전차 기동</th></tr></thead><tbody><tr><td>1</td><td>start</td><td>시작</td></tr><tr><td>2</td><td>pause</td><td>중지(현 상태 유지)</td></tr></tbody></table>

***

## 3.2.11  충돌정보

### <mark style="color:green;">POST</mark>    /collision

전차와 충돌된 객체의 정보를 전달합니다.

**Headers**

| Name         | Value              |
| ------------ | ------------------ |
| Content-Type | `application/json` |

**Response**

**Body**

{% tabs %}
{% tab title="JSON" %}

```json
{
  "objectName": "Wall001(Clone)",
  "position": {
    "x": 123.45,
    "y": 7.89,
    "z": 98.76
  }
}
```

{% endtab %}
{% endtabs %}

**Response**

{% tabs %}
{% tab title="200" %}

```json
{
  "status": "OK"
}
```

{% endtab %}

{% tab title="400" %}

```json
{
  "status": "ERROR",
  "message": "Missing collision data"
}
```

{% endtab %}
{% endtabs %}

## 3.2.12  스테레오 카메라 이미지

### <mark style="color:green;">POST</mark>    /stereo\_image

시뮬레이터에서 스테레오 카메라  이미지를 수신합니다.

**Headers**

| Name         | Value              |
| ------------ | ------------------ |
| Content-Type | `application/json` |

**Body**

<table><thead><tr><th width="223.3333740234375">Name</th><th width="178.3333740234375">Type</th><th>Description</th></tr></thead><tbody><tr><td>left_image</td><td>image</td><td>왼쪽스테레오 카메라 화면 이미지</td></tr><tr><td>right_image</td><td>image</td><td>오른쪽 스테레오 카메라 화면 이미지</td></tr></tbody></table>
