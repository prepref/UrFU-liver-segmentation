import { SVG } from '@svgdotjs/svg.js';
import '@svgdotjs/svg.draggable.js';
import JSZip from 'jszip';

const container = document.getElementById('svgContainer');
let draw,
   shape,
   handles = [],
   shapeType = null;

const epsilon = 3; // чувствительность упрощения

Promise.all([
   fetch('img/image.jpg').then((res) => res.blob()),
   fetch('img/contour.svg').then((res) => res.text())
])
.then(([imageBlob, svgText]) => {
   const imageUrl = URL.createObjectURL(imageBlob);
   document.getElementById('baseImage').src = imageUrl;

   const parser = new DOMParser();
   const svgDoc = parser.parseFromString(svgText, 'image/svg+xml');
   const svgElement = svgDoc.documentElement;

   container.innerHTML = '';
   container.appendChild(svgElement);

   draw = SVG(svgElement);

   shape =
      draw.findOne('path') ||
      draw.findOne('polygon') ||
      draw.findOne('polyline');

   if (!shape) {
      console.error('Нет поддерживаемых фигур (path, polygon, polyline)');
      return;
   }

   shapeType = shape.node.tagName;
   createHandles();
})
.catch((err) => console.error('Ошибка при загрузке файлов:', err));


function createHandles() {
   handles.forEach((h) => h.remove());
   handles = [];

   let points = [];

   if (shapeType === 'path') {
      const d = shape.attr('d');
      points = extractPathPoints(d);
   } else {
      const raw = shape.attr('points');
      points = raw
         .trim()
         .split(' ')
         .map((p) => p.split(',').map(Number));
   }

   points = simplifyRDP(points, epsilon);

   points.forEach(([x, y]) => {
      const handle = draw
         .circle(10)
         .center(x, y)
         .fill('#ff0')
         .stroke({ width: 1, color: '#000' })
         .draggable();

      handle.on('dragmove', updateShapeFromHandles);
      handle.on('dblclick', () => {
         handle.remove();
         handles = handles.filter((h) => h !== handle);
         updateShapeFromHandles();
      });

      handles.push(handle);
   });
}

function updateShapeFromHandles() {
   const points = handles.map((h) => `${h.cx()},${h.cy()}`);
   if (shapeType === 'path') {
      const d = `M${points.join(' L')} Z`;
      shape.plot(d);
   } else {
      shape.attr('points', points.join(' '));
   }
}

function extractPathPoints(d) {
   const match = d.match(/[-\d\.]+,[-\d\.]+/g);
   return match ? match.map((p) => p.split(',').map(Number)) : [];
}

function simplifyRDP(points, epsilon) {
   if (points.length < 3) return points;

   const getPerpendicularDistance = (pt, lineStart, lineEnd) => {
      const [x, y] = pt;
      const [x1, y1] = lineStart;
      const [x2, y2] = lineEnd;
      const dx = x2 - x1;
      const dy = y2 - y1;

      if (dx === 0 && dy === 0) return Math.hypot(x - x1, y - y1);

      const t = ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy);
      const closestX = x1 + t * dx;
      const closestY = y1 + t * dy;
      return Math.hypot(x - closestX, y - closestY);
   };

   const rdp = (pts, start, end) => {
      let maxDist = 0;
      let index = start;

      for (let i = start + 1; i < end; i++) {
         const dist = getPerpendicularDistance(pts[i], pts[start], pts[end]);
         if (dist > maxDist) {
            index = i;
            maxDist = dist;
         }
      }

      if (maxDist > epsilon) {
         const left = rdp(pts, start, index);
         const right = rdp(pts, index, end);
         return left.slice(0, -1).concat(right);
      } else {
         return [pts[start], pts[end]];
      }
   };

   return rdp(points, 0, points.length - 1);
}

function addHandle() {
   if (handles.length === 0) return;
   const last = handles[handles.length - 1];
   const newX = last.cx() + 10;
   const newY = last.cy() + 10;

   const handle = draw
      .circle(10)
      .center(newX, newY)
      .fill('#0f0')
      .stroke({ width: 1, color: '#000' })
      .draggable();

   handle.on('dragmove', updateShapeFromHandles);
   handle.on('dblclick', () => {
      handle.remove();
      handles = handles.filter((h) => h !== handle);
      updateShapeFromHandles();
   });

   handles.push(handle);
   updateShapeFromHandles();
}

function saveSVG() {
   const svg = draw.svg();
   const blob = new Blob([svg], { type: 'image/svg+xml' });
   const url = URL.createObjectURL(blob);
   const a = document.createElement('a');
   a.href = url;
   a.download = 'edited_contours.svg';
   a.click();
}

const controlPanel = document.createElement('div');
controlPanel.style = 'padding:10px;text-align:center';
controlPanel.innerHTML = `
  <button id="addBtn">Добавить точку</button>
  <button id="saveBtn">Сохранить SVG</button>
  <button id="uploadBtn">Загрузить фото</button>
  <input type="file" id="uploadImageInput" style="display:none" />
  <p style="color:gray">Двойной клик по точке — удалить</p>
`;
document.body.appendChild(controlPanel);

document.getElementById('addBtn').onclick = addHandle;
document.getElementById('saveBtn').onclick = saveSVG;

document.getElementById('uploadBtn').onclick = () => {
   document.getElementById('uploadImageInput').click();
};

document.getElementById('uploadImageInput').onchange = async (e) => {
   const file = e.target.files[0];
   if (!file) return;

   const formData = new FormData();
   formData.append('file', file);

   try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
         method: 'POST',
         body: formData,
      });

      if (!response.ok)
         throw new Error('Ошибка от сервера: ' + response.status);

      const result = await response.json();
      console.log('Ответ от /predict:', result);
      alert('Файл успешно отправлен на /predict!');
   } catch (err) {
      console.error('Ошибка при отправке:', err);
      alert('Ошибка при отправке на API');
   }
};
