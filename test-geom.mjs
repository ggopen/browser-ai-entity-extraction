import * as THREE from 'three';
const g = new THREE.BufferGeometry();
const arr = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]);
g.setAttribute('position', new THREE.BufferAttribute(arr, 3));
console.log('attr:', g.attributes.position);
console.log('count:', g.attributes.position.count);
