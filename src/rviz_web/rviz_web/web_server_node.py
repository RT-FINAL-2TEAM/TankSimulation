# -*- coding: utf-8 -*-
"""Standalone Flask web server for Tank Challenge RViz-like 3D visualization.

rviz_web is intentionally separated from ros_bridge:
- ros_bridge keeps simulator HTTP APIs on port 5000.
- rviz_web serves the 3D operations view on port 5055.
- The browser subscribes to ROS2 topics through rosbridge_server, usually ws://<host>:9090.

Coordinate policy used by the Tank Challenge stack:
- Unity raw:      x=left/right, y=height, z=forward/back
- ROS tank_map:   x=raw.x,     y=raw.z,  z=raw.y
- Three.js web:   x=map.x,     y=map.z,  z=-map.y

The negative sign on web.z keeps the displayed map orientation consistent with the
right-handed Three.js Y-up scene while matching RViz tank_map directions.
"""

from __future__ import annotations

import argparse
import os
from copy import deepcopy
from typing import Any, Dict

from flask import Flask, jsonify, request


DEFAULT_CONFIG: Dict[str, Any] = {
    "viewer": "rviz_web_manual_threejs_v4_terrain_mesh_20260625",
    "coordinatePolicy": "Unity raw(x,y,z) -> ROS tank_map(x=raw.x,y=raw.z,z=raw.y) -> Three.js(x=map.x,y=map.z,z=-map.y)",
    "fixedFrame": os.environ.get("TANK_RVIZ_WEB_FIXED_FRAME", "tank_map"),
    "rosbridgeHost": os.environ.get("TANK_RVIZ_WEB_ROSBRIDGE_HOST", ""),
    "rosbridgePort": int(os.environ.get("TANK_RVIZ_WEB_ROSBRIDGE_PORT", "9090")),
    "defaultCloud": os.environ.get("TANK_RVIZ_WEB_DEFAULT_CLOUD", "detected"),
    "defaultRays": os.environ.get("TANK_RVIZ_WEB_DEFAULT_RAYS", "true").lower() in ("1", "true", "yes", "y"),
    "defaultVectors": os.environ.get("TANK_RVIZ_WEB_DEFAULT_VECTORS", "true").lower() in ("1", "true", "yes", "y"),
    "limits": {
        "maxCloudPoints": int(os.environ.get("TANK_RVIZ_WEB_MAX_CLOUD_POINTS", "14000")),
        "maxRayLines": int(os.environ.get("TANK_RVIZ_WEB_MAX_RAYS", "90")),
        "potentialVectorScale": float(os.environ.get("TANK_RVIZ_WEB_POTENTIAL_VECTOR_SCALE", "12.0")),
        "potentialVectorZOffset": float(os.environ.get("TANK_RVIZ_WEB_POTENTIAL_VECTOR_Z_OFFSET", "4.0")),
    },
    "topics": {
        "paths": [],
        "markerArrays": [
            "/tank/rviz/object_markers",
            "/tank/rviz/obstacle_markers",
            "/tank/rviz/lidar_markers",
            "/tank/rviz/risk_markers",
            "/tank/rviz/potential_markers",
            "/tank/rviz/potential_field_markers",
            "/tank/rviz/lidar_cluster_markers",
            "/tank/rviz/fused_object_markers",
            "/tank/rviz/discovered_object_markers",
            "/tank/rviz/recon_map_markers",
            "/tank/rviz/terrain_markers",
            "/tank/terrain/final_elevation_markers",
            "/tank/terrain/final_wireframe_markers",
            "/tank/rviz/mission_map_markers",
            "/tank/rviz/map_diff_markers",
        ],
        "pointCloud2": {
            "detected": "/tank/sensor/lidar/detected_points_map",
            "all": "/tank/sensor/lidar/all_detected_points_map",
            "terrain": "/tank/sensor/lidar/terrain_points_map",
            "final": "/tank/terrain/final_accumulated_cloud",
            "ground": "/tank/terrain/final_ground_points",
            "nonground": "/tank/terrain/final_non_ground_points",
        },
        "rayCloud": "/tank/sensor/lidar/all_detected_points_map",
        "poses": [
            "/tank/player/pose",
            "/tank/enemy/pose",
            "/tank/local_target/pose",
            "/tank/goal/pose",
            "/tank/latest_pose",
        ],
        "vectors": {
            "attractive": "/tank/potential/attractive_vector",
            "repulsive": "/tank/potential/repulsive_vector",
            "result": "/tank/potential/result_vector",
        },
        "lidarOrigin": "/tank/sensor/lidar/origin",
    },
}


def make_app(config: Dict[str, Any]) -> Flask:
    app = Flask(__name__)

    @app.route("/api/config", methods=["GET"])
    def api_config():
        payload = deepcopy(config)
        frame = request.args.get("frame")
        if frame:
            payload["fixedFrame"] = frame
        return jsonify(payload)

    @app.route("/", methods=["GET"])
    @app.route("/rviz3d", methods=["GET"])
    def rviz3d():
        return HTML_PAGE

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"ok": True, "package": "rviz_web", "viewer": config.get("viewer")})

    return app


HTML_PAGE = r"""
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TANK RViz Web 3D</title>
  <style>
    :root { --bg:#020706; --panel:rgba(5,18,12,.94); --line:rgba(81,255,150,.38); --text:#dcffe9; --muted:#8fcaa1; --ok:#52ff98; --bad:#ff6171; --warn:#ffd75a; }
    html,body{margin:0;width:100%;height:100%;overflow:hidden;background:var(--bg);color:var(--text);font-family:Consolas,Monaco,"Noto Sans KR",monospace;}
    #topbar{position:absolute;top:0;left:0;right:0;height:46px;z-index:10;display:flex;align-items:center;justify-content:space-between;padding:0 16px;background:linear-gradient(90deg,rgba(5,18,10,.99),rgba(7,33,20,.9));border-bottom:1px solid var(--line);box-sizing:border-box;}
    #brand{font-size:16px;font-weight:700;letter-spacing:1px;} #brand span{color:var(--ok)}
    #status{font-size:12px;padding:6px 10px;border:1px solid rgba(255,255,255,.16);border-radius:999px;background:rgba(0,0,0,.24)}
    .ok{color:var(--ok)} .bad{color:var(--bad)} .warn{color:var(--warn)}
    #viewer{position:absolute;inset:46px 0 0 0;width:100vw;height:calc(100vh - 46px)}
    #side{position:absolute;top:62px;left:14px;z-index:11;width:360px;max-height:calc(100vh - 88px);overflow:auto;padding:12px;box-sizing:border-box;background:var(--panel);border:1px solid var(--line);border-radius:12px;box-shadow:0 12px 32px rgba(0,0,0,.42);}
    h2{margin:0 0 8px;font-size:13px;color:var(--ok);letter-spacing:1px}.hint{margin:8px 0 12px;color:var(--muted);line-height:1.45;font-size:11px}.section{margin-top:12px;padding-top:10px;border-top:1px solid rgba(81,255,150,.18)}
    .row{display:flex;justify-content:space-between;gap:10px;margin:6px 0;font-size:11px}.key{color:var(--muted)}.value{text-align:right;word-break:break-all}
    button{border:1px solid rgba(77,255,145,.44);background:rgba(24,72,43,.72);color:var(--text);border-radius:8px;padding:7px 9px;margin:4px 4px 0 0;cursor:pointer;font-family:inherit;font-size:11px} button:hover{border-color:var(--ok)}
    .topic{display:grid;grid-template-columns:10px 1fr auto;gap:6px;align-items:center;margin:4px 0;font-size:10px}.dot{width:7px;height:7px;border-radius:50%;background:#47524b}.dot.on{background:var(--ok);box-shadow:0 0 8px rgba(77,255,145,.7)}.dot.off{background:var(--bad)}.count{color:var(--muted);text-align:right}
    #log{font-size:10px;line-height:1.45;color:var(--muted);white-space:pre-wrap;max-height:155px;overflow:auto}.badge{color:var(--ok)}
  </style>
  <script src="https://cdn.jsdelivr.net/npm/eventemitter2@6.4.9/lib/eventemitter2.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/roslib@1/build/roslib.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.89.0/build/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.89.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
  <div id="topbar"><div id="brand">TANK-CV <span>RViz Web 3D</span></div><div id="status" class="bad">LOADING</div></div>
  <div id="viewer"></div>
  <div id="side">
    <h2>RVIZ_WEB TERRAIN MESH COORD v4</h2>
    <div class="hint">좌표: Unity raw(x,y,z) → ROS tank_map(x, z, y) → Web(x, z, -y)<br>LiDAR ray는 all_detected_points_map 기준, cloud 표시는 선택 모드 기준.</div>
    <div class="row"><div class="key">ROSBRIDGE</div><div id="rosbridgeInfo" class="value">-</div></div>
    <div class="row"><div class="key">FIXED FRAME</div><div id="frameInfo" class="value">-</div></div>
    <div class="row"><div class="key">CLOUD MODE</div><div id="cloudInfo" class="value">-</div></div>
    <div class="row"><div class="key">RAYS</div><div id="rayInfo" class="value">-</div></div>
    <div class="row"><div class="key">VECTORS</div><div id="vectorInfo" class="value">-</div></div>
    <div class="row"><div class="key">BOUNDS</div><div id="boundsInfo" class="value">waiting</div></div>
    <div class="section">
      <button onclick="setCloud('off')">Cloud OFF</button><button onclick="setCloud('detected')">Detected</button><button onclick="setCloud('all')">All</button><button onclick="setCloud('terrain')">Terrain</button>
      <button onclick="setCloud('final')">Final cloud</button><button onclick="setCloud('ground')">Ground pts</button><button onclick="setCloud('nonground')">Non-ground</button>
      <button onclick="toggleRays()">Ray ON/OFF</button><button onclick="toggleVectors()">Vector ON/OFF</button><button onclick="fitData()">Fit data</button><button onclick="resetCamera()">Reset</button><button onclick="clearDynamic()">Clear</button>
    </div>
    <div class="section"><h2>TOPICS</h2><div id="topics"></div></div>
    <div class="section"><h2>LOG</h2><div id="log"></div></div>
  </div>

<script>
let cfg=null, ros=null, scene=null, camera=null, renderer=null, controls=null, rootGroup=null;
let fixedFrame=new URLSearchParams(location.search).get('frame') || 'tank_map';
let cloudMode=new URLSearchParams(location.search).get('cloud') || 'detected';
let raysOn=(new URLSearchParams(location.search).get('rays') || '1') === '1';
let vectorsOn=(new URLSearchParams(location.search).get('vectors') || '1') === '1';
let cloudSub=null, rayCloudSub=null, currentCloudTopic=null;
let markerObjects=new Map(), pathObjects=new Map(), poseObjects=new Map(), vectorObjects=new Map();
let cloudObject=null, rayObject=null;
let latestOriginMap=null, latestRayCloudPoints=[], latestSelectedCloudPoints=[], latestPlayerPoseMap=null;
let dataBox=new THREE.Box3(), hasDataBounds=false;
let topicRows={};
let MAX_CLOUD_POINTS=14000, MAX_RAYS=90, POTENTIAL_VECTOR_SCALE=12.0, POTENTIAL_VECTOR_Z_OFFSET=4.0;

function log(msg){ const t=new Date().toLocaleTimeString(); const el=document.getElementById('log'); el.textContent=`[${t}] ${msg}\n`+el.textContent; }
function setStatus(text, cls){ const e=document.getElementById('status'); e.textContent=text; e.className=cls||'warn'; }
function updateInfo(){
  document.getElementById('frameInfo').textContent=fixedFrame;
  document.getElementById('cloudInfo').textContent=String(cloudMode).toUpperCase();
  document.getElementById('rayInfo').textContent=raysOn?'ON':'OFF';
  document.getElementById('vectorInfo').textContent=vectorsOn?'ON':'OFF';
}
function topicId(topic){ return 'topic_'+String(topic||'').replace(/[^a-zA-Z0-9_]/g,'_'); }
function ensureTopic(topic){
  if(topicRows[topic]) return;
  const row=document.createElement('div'); row.className='topic'; row.id=topicId(topic);
  row.innerHTML=`<span class="dot"></span><span>${topic}</span><span class="count">wait</span>`;
  document.getElementById('topics').appendChild(row);
  topicRows[topic]=row;
}
function markTopic(topic, count, good=true){
  ensureTopic(topic); const row=topicRows[topic]; const dot=row.querySelector('.dot'); const c=row.querySelector('.count');
  dot.className='dot '+(good?'on':'off'); c.textContent=String(count);
}
function addBounds(v){ if(Number.isFinite(v.x)&&Number.isFinite(v.y)&&Number.isFinite(v.z)){ dataBox.expandByPoint(v); hasDataBounds=true; } }
function updateBoundsLabel(){ if(!hasDataBounds){ document.getElementById('boundsInfo').textContent='waiting'; return; } const s=dataBox.getSize(new THREE.Vector3()); document.getElementById('boundsInfo').textContent=`${s.x.toFixed(1)} x ${s.y.toFixed(1)} x ${s.z.toFixed(1)}`; }

function rosToWebPoint(p){ return new THREE.Vector3(Number(p?.x||0), Number(p?.z||0), -Number(p?.y||0)); }
function rosVectorToWeb(v){ return new THREE.Vector3(Number(v?.x||0), Number(v?.z||0), -Number(v?.y||0)); }
function yawFromQuat(q){ if(!q) return 0; const x=+q.x||0, y=+q.y||0, z=+q.z||0, w=(q.w===undefined?1:+q.w); return Math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)); }
function rotateMapPoint2D(p, yaw){ const c=Math.cos(yaw), s=Math.sin(yaw); return {x:p.x*c - p.y*s, y:p.x*s + p.y*c, z:p.z}; }
function colorFromMsg(c, fallback=0x55ff99){ if(!c) return fallback; const r=Math.max(0,Math.min(255,Math.round((c.r??0.5)*255))); const g=Math.max(0,Math.min(255,Math.round((c.g??1)*255))); const b=Math.max(0,Math.min(255,Math.round((c.b??0.6)*255))); return (r<<16)|(g<<8)|b; }
function alphaFromMsg(c, fallback=0.85){ const a=Number(c?.a); return Number.isFinite(a)?Math.max(0.04,Math.min(1,a)):fallback; }

function initThree(){
  const div=document.getElementById('viewer');
  scene=new THREE.Scene(); scene.background=new THREE.Color(0x020706);
  camera=new THREE.PerspectiveCamera(55, div.clientWidth/div.clientHeight, 0.1, 5000);
  camera.position.set(150,230,130);
  renderer=new THREE.WebGLRenderer({antialias:true}); renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2)); renderer.setSize(div.clientWidth,div.clientHeight); div.appendChild(renderer.domElement);
  controls=new THREE.OrbitControls(camera, renderer.domElement); controls.target.set(150,0,-150); controls.update();
  scene.add(new THREE.AmbientLight(0xffffff,0.85)); const light=new THREE.DirectionalLight(0xffffff,0.7); light.position.set(60,160,40); scene.add(light);
  rootGroup=new THREE.Group(); scene.add(rootGroup);
  const grid=new THREE.GridHelper(320,64,0x105f38,0x105f38); grid.position.set(150,-0.02,-150); rootGroup.add(grid);
  const axes=new THREE.AxesHelper(12); rootGroup.add(axes);
  window.addEventListener('resize',()=>{ const w=div.clientWidth,h=div.clientHeight; camera.aspect=w/h; camera.updateProjectionMatrix(); renderer.setSize(w,h); });
  animate();
}
function animate(){ requestAnimationFrame(animate); controls.update(); renderer.render(scene,camera); }
function resetCamera(){ camera.position.set(150,230,130); controls.target.set(150,0,-150); controls.update(); }
function fitData(){
  if(!hasDataBounds){ resetCamera(); log('fit fallback: no data bounds'); return; }
  const c=dataBox.getCenter(new THREE.Vector3()), s=dataBox.getSize(new THREE.Vector3()); const maxDim=Math.max(s.x,s.y,s.z,25);
  controls.target.copy(c); camera.position.set(c.x+maxDim*0.75, c.y+maxDim*0.9+70, c.z+maxDim*1.2);
  camera.near=0.1; camera.far=Math.max(5000,maxDim*20); camera.updateProjectionMatrix(); controls.update(); updateBoundsLabel();
}
function removeObject(obj){ if(obj){ rootGroup.remove(obj); } }
function clearDynamic(){
  for(const [,o] of markerObjects) removeObject(o); markerObjects.clear();
  for(const [,o] of pathObjects) removeObject(o); pathObjects.clear();
  for(const [,o] of poseObjects) removeObject(o); poseObjects.clear();
  for(const [,o] of vectorObjects) removeObject(o); vectorObjects.clear();
  removeObject(cloudObject); cloudObject=null; removeObject(rayObject); rayObject=null;
  dataBox=new THREE.Box3(); hasDataBounds=false; updateBoundsLabel(); log('cleared dynamic objects');
}

function makeLine(points, color, opacity=1.0, segments=false){
  if(!points || points.length<2) return null;
  const geom=new THREE.BufferGeometry().setFromPoints(points);
  const mat=new THREE.LineBasicMaterial({color, transparent:opacity<1, opacity});
  const obj=segments?new THREE.LineSegments(geom,mat):new THREE.Line(geom,mat);
  points.forEach(addBounds); return obj;
}
function makePoints(points, size, color, opacity=1.0){
  const geom=new THREE.BufferGeometry().setFromPoints(points);
  const mat=new THREE.PointsMaterial({size, color, transparent:opacity<1, opacity, sizeAttenuation:true});
  const obj=new THREE.Points(geom,mat); points.forEach(addBounds); return obj;
}
function colorTripletFromMsg(c, fallback=[0.33,1.0,0.6]){
  if(!c) return fallback;
  return [Math.max(0,Math.min(1,Number(c.r??fallback[0]))), Math.max(0,Math.min(1,Number(c.g??fallback[1]))), Math.max(0,Math.min(1,Number(c.b??fallback[2])))];
}
function makeTriangleList(marker){
  const rawPts=marker.points||[];
  if(rawPts.length<3) return null;
  const verts=[], cols=[];
  const defaultRGB=colorTripletFromMsg(marker.color,[0.35,1.0,0.18]);
  for(let i=0;i<rawPts.length;i++){
    const v=markerPointToWeb(marker, rawPts[i]);
    verts.push(v.x,v.y,v.z);
    const rgb=colorTripletFromMsg((marker.colors||[])[i], defaultRGB);
    cols.push(rgb[0],rgb[1],rgb[2]);
    addBounds(v);
  }
  const geom=new THREE.BufferGeometry();
  geom.addAttribute('position', new THREE.Float32BufferAttribute(verts,3));
  geom.addAttribute('color', new THREE.Float32BufferAttribute(cols,3));
  geom.computeBoundingSphere();
  const opacity=alphaFromMsg(marker.color,0.82);
  const mat=new THREE.MeshBasicMaterial({side:THREE.DoubleSide, vertexColors:THREE.VertexColors, transparent:opacity<1, opacity});
  return new THREE.Mesh(geom,mat);
}
function makeArrow(points, color, opacity=1.0){
  const group=new THREE.Group(); if(!points || points.length<2) return group;
  const line=makeLine(points,color,opacity,false); if(line) group.add(line);
  const a=points[points.length-2], b=points[points.length-1]; const dir=new THREE.Vector3().subVectors(b,a); const len=dir.length();
  if(len>0.01){ const cone=new THREE.Mesh(new THREE.ConeGeometry(Math.max(0.35,len*0.045),Math.max(0.9,len*0.13),12),new THREE.MeshBasicMaterial({color,transparent:opacity<1,opacity})); cone.position.copy(b); cone.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),dir.clone().normalize()); group.add(cone); }
  points.forEach(addBounds); return group;
}

function markerKey(topic,m){ return `${topic}|${m.ns||''}|${m.id}`; }
function deleteMarker(topic,m){ const k=markerKey(topic,m); const o=markerObjects.get(k); if(o){ removeObject(o); markerObjects.delete(k); } }
function setMarker(topic,m,obj){
  const k=markerKey(topic,m);

  // 같은 topic / namespace / id Marker가 있으면 먼저 교체
  deleteMarker(topic,m);

  markerObjects.set(k,obj);
  rootGroup.add(obj);

  // ROS Marker lifetime을 웹에서도 적용한다.
  const lifetime = m.lifetime || {};
  const lifetimeMs =
    Number(lifetime.sec || 0) * 1000 +
    Number(lifetime.nanosec || 0) / 1000000;

  if(lifetimeMs > 0){
    setTimeout(() => {
      // 그 사이 같은 ID의 새 Marker가 들어왔다면 새 Marker를 지우면 안 된다.
      if(markerObjects.get(k) === obj){
        removeObject(obj);
        markerObjects.delete(k);
      }
    }, lifetimeMs);
  }
}

function markerPointToWeb(m,p){
  const pose=m.pose||{}; const pos=pose.position||{x:0,y:0,z:0}; const yaw=yawFromQuat(pose.orientation);
  let local={x:+p.x||0, y:+p.y||0, z:+p.z||0};
  if(Math.abs(yaw)>1e-6) local=rotateMapPoint2D(local,yaw);
  return rosToWebPoint({x:local.x+(+pos.x||0), y:local.y+(+pos.y||0), z:local.z+(+pos.z||0)});
}
function applyMarkerPose(obj, pose){
  if(!pose) return;
  const p=rosToWebPoint(pose.position||{}); obj.position.copy(p);
  obj.rotation.y=yawFromQuat(pose.orientation||{});
  addBounds(p);
}

function renderMarker(topic,m){
  if(m.action===3){ for(const [k,o] of Array.from(markerObjects.entries())){ if(k.startsWith(topic+'|')){ removeObject(o); markerObjects.delete(k); } } return; }
  if(m.action===2){ deleteMarker(topic,m); return; }
  const color=colorFromMsg(m.color); const opacity=alphaFromMsg(m.color);
  const sx=Math.max(Number(m.scale?.x||0.4),0.02), sy=Math.max(Number(m.scale?.y||0.4),0.02), sz=Math.max(Number(m.scale?.z||0.4),0.02);
  let obj=null, type=m.type;
  if(type===0){ // ARROW
    const pts=(m.points||[]).map(p=>markerPointToWeb(m,p));
    if(pts.length>=2) obj=makeArrow(pts,color,opacity);
    else {
      const start=rosToWebPoint(m.pose?.position||{});
      const yaw=yawFromQuat(m.pose?.orientation||{});
      const end=start.clone().add(new THREE.Vector3(Math.cos(yaw)*Math.max(sx,3),0,-Math.sin(yaw)*Math.max(sx,3)));
      obj=makeArrow([start,end],color,opacity);
    }
  } else if(type===1){ // CUBE
    obj=new THREE.Mesh(new THREE.BoxGeometry(sx,sz,sy),new THREE.MeshBasicMaterial({color,transparent:opacity<1,opacity})); applyMarkerPose(obj,m.pose);
  } else if(type===2){ // SPHERE
    obj=new THREE.Mesh(new THREE.SphereGeometry(0.5,16,10),new THREE.MeshBasicMaterial({color,transparent:opacity<1,opacity})); obj.scale.set(sx,sz,sy); applyMarkerPose(obj,m.pose);
  } else if(type===3){ // CYLINDER
    obj=new THREE.Mesh(new THREE.CylinderGeometry(sx/2,sx/2,sz,16),new THREE.MeshBasicMaterial({color,transparent:opacity<1,opacity})); applyMarkerPose(obj,m.pose);
  } else if(type===4){ // LINE_STRIP
    const pts=(m.points||[]).map(p=>markerPointToWeb(m,p)); obj=makeLine(pts,color,opacity,false);
  } else if(type===5){ // LINE_LIST
    const pts=(m.points||[]).map(p=>markerPointToWeb(m,p)); obj=makeLine(pts,color,opacity,true);
  } else if(type===6 || type===7 || type===8){ // CUBE_LIST / SPHERE_LIST / POINTS
    const pts=(m.points||[]).map(p=>markerPointToWeb(m,p)); obj=makePoints(pts,Math.max(sx,0.35),color,opacity);
  } else if(type===9){ // TEXT_VIEW_FACING, approximate as small sphere anchor
    obj=new THREE.Mesh(new THREE.SphereGeometry(Math.max(0.6,sz*0.25),8,6),new THREE.MeshBasicMaterial({color,transparent:true,opacity:0.75})); applyMarkerPose(obj,m.pose);
  } else if(type===11){ // TRIANGLE_LIST: final terrain elevation mesh. Preserve per-vertex marker.colors like RViz.
    obj=makeTriangleList(m);
  }
  if(obj) setMarker(topic,m,obj);
}
function renderMarkerArray(topic,msg){
  const arr=msg.markers||[];
  let triPts=0;
  arr.forEach(m=>{ if(m.type===11 && m.points) triPts += m.points.length; renderMarker(topic,m); });
  markTopic(topic, triPts ? `${arr.length} markers / ${triPts/3} tris` : arr.length);
  if(topic.includes('/terrain/final_elevation_markers') && triPts>0 && !window.__fitFinalTerrainOnce){ window.__fitFinalTerrainOnce=true; setTimeout(fitData,500); log(`final terrain mesh rendered: ${Math.floor(triPts/3)} triangles`); }
  updateBoundsLabel();
}

function renderPath(topic,msg){
  const pts=(msg.poses||[]).map(ps=>rosToWebPoint(ps.pose.position));
  if(pathObjects.has(topic)) removeObject(pathObjects.get(topic));
  if(pts.length>=2){ const obj=makeLine(pts,0xffff33,1.0,false); pathObjects.set(topic,obj); rootGroup.add(obj); }
  markTopic(topic,pts.length); updateBoundsLabel();
}

function bytesFromRosData(data){ if(typeof data==='string'){ const bin=atob(data); const bytes=new Uint8Array(bin.length); for(let i=0;i<bin.length;i++) bytes[i]=bin.charCodeAt(i); return bytes; } if(Array.isArray(data)) return new Uint8Array(data); if(data && data.buffer) return new Uint8Array(data); return new Uint8Array(0); }
function fieldOffset(msg,name){ const f=(msg.fields||[]).find(x=>x.name===name); return f?f.offset:-1; }
function decodePointCloud2(msg, limit){
  const ox=fieldOffset(msg,'x'), oy=fieldOffset(msg,'y'), oz=fieldOffset(msg,'z'); if(ox<0||oy<0||oz<0) return [];
  const bytes=bytesFromRosData(msg.data); const dv=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength); const n=(msg.width||0)*(msg.height||1); const step=msg.point_step||16; const little=!msg.is_bigendian;
  const stride=Math.max(1,Math.ceil(n/Math.max(1,limit))); const pts=[];
  for(let i=0;i<n;i+=stride){ const off=i*step; if(off+Math.max(ox,oy,oz)+4>bytes.length) break; const x=dv.getFloat32(off+ox,little), y=dv.getFloat32(off+oy,little), z=dv.getFloat32(off+oz,little); if(Number.isFinite(x)&&Number.isFinite(y)&&Number.isFinite(z)) pts.push(rosToWebPoint({x,y,z})); }
  return pts;
}
function renderCloud(topic,msg){
  const pts=decodePointCloud2(msg,MAX_CLOUD_POINTS); latestSelectedCloudPoints=pts;
  removeObject(cloudObject); cloudObject=null;
  if(pts.length){ cloudObject=makePoints(pts,0.55,0xff9a22,0.88); rootGroup.add(cloudObject); }
  markTopic(topic,pts.length); updateBoundsLabel();
}
function renderRayCloud(topic,msg){ latestRayCloudPoints=decodePointCloud2(msg,MAX_CLOUD_POINTS); markTopic(topic,latestRayCloudPoints.length); updateRays(); }
function currentRayOriginWeb(){
  if(latestOriginMap) return rosToWebPoint(latestOriginMap);
  if(latestPlayerPoseMap){ const p=latestPlayerPoseMap.position||{}; return rosToWebPoint({x:p.x||0,y:p.y||0,z:(+p.z||0)+1.2}); }
  return null;
}
function updateRays(){
  removeObject(rayObject); rayObject=null;
  if(!raysOn || !latestRayCloudPoints.length) return;
  const origin=currentRayOriginWeb(); if(!origin){ markTopic('synthetic_rays',0,false); return; }
  const pts=[]; const stride=Math.max(1,Math.ceil(latestRayCloudPoints.length/MAX_RAYS));
  for(let i=0;i<latestRayCloudPoints.length;i+=stride){ pts.push(origin,latestRayCloudPoints[i]); }
  rayObject=makeLine(pts,0x33ccff,0.30,true); if(rayObject) rootGroup.add(rayObject);
  markTopic('synthetic_rays',Math.floor(pts.length/2)); log(`synthetic rays rendered: ${Math.floor(pts.length/2)}`);
}
function renderOrigin(topic,msg){ if(msg.point){ latestOriginMap=msg.point; markTopic(topic,1); updateRays(); } }

function renderPose(topic,msg){
  const pose=msg.pose||msg; if(!pose || !pose.position) return;
  if(topic.includes('/player/pose') || topic.includes('/latest_pose')) latestPlayerPoseMap=pose;
  if(poseObjects.has(topic)) removeObject(poseObjects.get(topic));
  const pos=rosToWebPoint(pose.position); const group=new THREE.Group();
  const isEnemy=topic.includes('enemy'); const isTarget=topic.includes('target')||topic.includes('goal');
  const color=isEnemy?0xff3333:(isTarget?0xffff33:0x33aaff);
  const sphere=new THREE.Mesh(new THREE.SphereGeometry(isTarget?1.2:1.5,12,8),new THREE.MeshBasicMaterial({color}));
  sphere.position.copy(pos); group.add(sphere);
  const yaw=yawFromQuat(pose.orientation||{}); const dir=new THREE.Vector3(Math.cos(yaw),0,-Math.sin(yaw)); const arrow=makeArrow([pos,pos.clone().add(dir.multiplyScalar(7.0))],color,0.85); if(false && arrow) group.add(arrow);
}

function vectorStartMap(){
  if(latestPlayerPoseMap && latestPlayerPoseMap.position){ const p=latestPlayerPoseMap.position; return {x:+p.x||0, y:+p.y||0, z:(+p.z||0)+POTENTIAL_VECTOR_Z_OFFSET}; }
  if(latestOriginMap){ return {x:+latestOriginMap.x||0, y:+latestOriginMap.y||0, z:(+latestOriginMap.z||0)+POTENTIAL_VECTOR_Z_OFFSET}; }
  return null;
}
function renderVector(name,topic,msg){
  if(!vectorsOn){ markTopic(topic,0); return; }
  const startMap=vectorStartMap(); if(!startMap){ markTopic(topic,'no anchor',false); return; }
  const v=msg.vector||msg; const mag=Math.sqrt((+v.x||0)**2+(+v.y||0)**2+(+v.z||0)**2); if(mag<1e-6){ markTopic(topic,0); return; }
  const scale=POTENTIAL_VECTOR_SCALE; const endMap={x:startMap.x+(+v.x||0)/mag*scale, y:startMap.y+(+v.y||0)/mag*scale, z:startMap.z+(+v.z||0)/mag*scale};
  const pts=[rosToWebPoint(startMap), rosToWebPoint(endMap)];
  const color=name==='repulsive'?0xff5500:(name==='attractive'?0x33ff66:0xff33ff);
  const obj=makeArrow(pts,color,0.95);
  if(vectorObjects.has(name)) removeObject(vectorObjects.get(name));
  vectorObjects.set(name,obj); rootGroup.add(obj); markTopic(topic,1); updateBoundsLabel();
}

function subscribeTopic(topic,type,cb,throttle=250){
  if(!topic) return null; ensureTopic(topic);
  const sub=new ROSLIB.Topic({ros,name:topic,messageType:type,throttle_rate:throttle,queue_length:1});
  sub.subscribe(msg=>cb(topic,msg)); log('subscribe '+topic); return sub;
}
function subscribeAll(){
  (cfg.topics.paths||[]).forEach(t=>subscribeTopic(t,'nav_msgs/Path',renderPath,500));
  (cfg.topics.markerArrays||[]).forEach(t=>subscribeTopic(t,'visualization_msgs/MarkerArray',renderMarkerArray,250));
  (cfg.topics.poses||[]).forEach(t=>subscribeTopic(t,'geometry_msgs/PoseStamped',renderPose,250));
  const vectors=cfg.topics.vectors||{}; Object.keys(vectors).forEach(name=>subscribeTopic(vectors[name],'geometry_msgs/Vector3Stamped',(topic,msg)=>renderVector(name,topic,msg),250));
  subscribeTopic(cfg.topics.lidarOrigin,'geometry_msgs/PointStamped',renderOrigin,250);
  if(raysOn && cfg.topics.rayCloud){ rayCloudSub=subscribeTopic(cfg.topics.rayCloud,'sensor_msgs/PointCloud2',renderRayCloud,400); }
  setCloud(cloudMode);
}
function setCloud(mode){
  cloudMode=mode||'off'; if(cloudSub){ cloudSub.unsubscribe(); cloudSub=null; } removeObject(cloudObject); cloudObject=null; latestSelectedCloudPoints=[];
  const maps=cfg?.topics?.pointCloud2||{}; const topic=maps[cloudMode]; currentCloudTopic=topic||null;
  if(topic) cloudSub=subscribeTopic(topic,'sensor_msgs/PointCloud2',renderCloud,300);
  updateInfo();
}
function toggleRays(){
  raysOn=!raysOn; updateInfo();
  if(raysOn && !rayCloudSub && cfg?.topics?.rayCloud){ rayCloudSub=subscribeTopic(cfg.topics.rayCloud,'sensor_msgs/PointCloud2',renderRayCloud,400); }
  updateRays();
}
function toggleVectors(){ vectorsOn=!vectorsOn; updateInfo(); if(!vectorsOn){ for(const [,o] of vectorObjects) removeObject(o); vectorObjects.clear(); } }

async function main(){
  initThree(); updateInfo();
  const urlParams=new URLSearchParams(location.search);
  const resp=await fetch('/api/config?frame='+encodeURIComponent(fixedFrame)+'&v='+Date.now(),{cache:'no-store'});
  cfg=await resp.json(); fixedFrame=cfg.fixedFrame||fixedFrame;
  MAX_CLOUD_POINTS=cfg.limits?.maxCloudPoints||MAX_CLOUD_POINTS; MAX_RAYS=cfg.limits?.maxRayLines||MAX_RAYS; POTENTIAL_VECTOR_SCALE=cfg.limits?.potentialVectorScale||POTENTIAL_VECTOR_SCALE; POTENTIAL_VECTOR_Z_OFFSET=cfg.limits?.potentialVectorZOffset||POTENTIAL_VECTOR_Z_OFFSET;
  if(!urlParams.has('cloud')) cloudMode=cfg.defaultCloud||cloudMode;
  if(!urlParams.has('rays')) raysOn=!!cfg.defaultRays;
  if(!urlParams.has('vectors')) vectorsOn=!!cfg.defaultVectors;
  updateInfo(); log('viewer='+cfg.viewer); log('coord='+cfg.coordinatePolicy);
  const host=cfg.rosbridgeHost || window.location.hostname; const url='ws://'+host+':'+cfg.rosbridgePort; document.getElementById('rosbridgeInfo').textContent=url;
  ros=new ROSLIB.Ros();
  let rbRetry=null;
  const rbConnect=()=>{ try{ ros.connect(url); }catch(e){ log('connect err '+e); } };
  ros.on('connection',()=>{ if(rbRetry){clearTimeout(rbRetry);rbRetry=null;} setStatus('ROSBRIDGE CONNECTED','ok'); log('connected '+url); subscribeAll(); });
  ros.on('error',()=>{ setStatus('ROSBRIDGE RETRY…','bad'); });
  ros.on('close',()=>{ setStatus('ROSBRIDGE RETRY…','bad'); log('rosbridge closed; retry 2s'); if(!rbRetry) rbRetry=setTimeout(()=>{rbRetry=null;rbConnect();},2000); });
  rbConnect();
}
main().catch(e=>{ setStatus('INIT FAILED','bad'); log(e.stack||String(e)); });
</script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Tank RViz Web 3D server")
    parser.add_argument("--host", default=os.environ.get("TANK_RVIZ_WEB_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("TANK_RVIZ_WEB_PORT", "5055")))
    parser.add_argument("--rosbridge-port", type=int, default=int(os.environ.get("TANK_RVIZ_WEB_ROSBRIDGE_PORT", "9090")))
    args, _unknown = parser.parse_known_args(argv)

    config = deepcopy(DEFAULT_CONFIG)
    config["rosbridgePort"] = args.rosbridge_port
    app = make_app(config)
    print("=" * 70)
    print("TANK RViz Web 3D")
    print(f"Viewer: {config['viewer']}")
    print(f"Open: http://127.0.0.1:{args.port}/rviz3d?frame={config['fixedFrame']}&cloud=detected&rays=1&vectors=1")
    print(f"ROS bridge expected: ws://<browser-host>:{args.rosbridge_port}")
    print("=" * 70)
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()