import {test} from 'node:test';
import assert from 'node:assert/strict';
import {Matrix4, Vector3} from 'three';
import {cameraPoseToFieldSpaceMatrix, mountingPoseToViewMatrix, position3DToFieldSpaceVector,FIELD_CENTER_X_METERS, FIELD_CENTER_Y_METERS} from '../../src/webui/js/utils/fieldSpaceTransforms.js';
const B=new Matrix4().set(0,0,1,0, -1,0,0,0, 0,-1,0,0, 0,0,0,1);
const A=new Matrix4().makeRotationX(-Math.PI/2);
const rows=m=>Array.from({length:4},(_,r)=>Array.from({length:4},(_,c)=>m.elements[c*4+r]));
const close=(a,b)=>a.forEach((x,i)=>assert.ok(Math.abs(x-b[i])<1e-8,`${i}: ${x} != ${b[i]}`));
for(const [yaw,pitch,roll] of [[0,0,0],[90,0,0],[-90,0,0],[180,0,0],[0,30,0],[0,0,25],[37,19,-23]]) {
 test(`field rotation matches WPILib at yaw/pitch/roll ${yaw}/${pitch}/${roll}`,()=>{
  const rad=Math.PI/180;
  const R=new Matrix4().makeRotationZ(yaw*rad).multiply(new Matrix4().makeRotationY(pitch*rad)).multiply(new Matrix4().makeRotationX(roll*rad));
  const camera=R.clone().multiply(B).setPosition(4,2,.6);
  const actual=cameraPoseToFieldSpaceMatrix(rows(camera));
  const expected=A.clone().multiply(R).multiply(A.clone().invert());
  expected.setPosition((4-FIELD_CENTER_X_METERS)*1000,600,(-2+FIELD_CENTER_Y_METERS)*1000);
  close(actual.elements,expected.elements); assert.ok(Math.abs(actual.determinant()-1)<1e-8);
 });
 test(`mount preview matches backend convention ${yaw}/${pitch}/${roll}`,()=>{
  const actual=mountingPoseToViewMatrix({x_offset:.3,y_offset:.4,z_offset:.6,yaw,pitch,roll});
  const rad=Math.PI/180;
  const R=new Matrix4().makeRotationZ(yaw*rad).multiply(new Matrix4().makeRotationY(pitch*rad)).multiply(new Matrix4().makeRotationX(roll*rad));
  const expected=A.clone().multiply(R).multiply(A.clone().invert()).setPosition(.3,.6,-.4);
  close(actual.elements,expected.elements);
 });
}
test('field translation and invalid inputs',()=>{
 close(position3DToFieldSpaceVector([FIELD_CENTER_X_METERS,FIELD_CENTER_Y_METERS,1]).toArray(),[0,1000,0]);
 assert.equal(cameraPoseToFieldSpaceMatrix([[NaN]]),null);
 assert.equal(position3DToFieldSpaceVector([Infinity,0,0]),null);
});
