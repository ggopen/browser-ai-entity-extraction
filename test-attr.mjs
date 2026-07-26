import * as THREE from 'three';
const arr = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]); // 3 vertices
const attr = new THREE.BufferAttribute(arr, 3);
console.log('count:', attr.count);
console.log('array.length:', attr.array.length);
console.log('itemSize:', attr.itemSize);
console.log('keys:', Object.keys(attr));
