const fs = require('fs');
const jpeg = require('jpeg-js');

// 将JPEG图像解码为像素数组
function decodeJpeg(buffer) {
  const { width, height, data } = jpeg.decode(buffer);
  const pixels = new Uint8Array(width * height);
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    // 转换为灰度值
    const gray = (r + g + b) / 3;
    pixels[i / 4] = gray;
  }
  return pixels;
}

// 获取图像文件列表
const imageFiles = fs.readdirSync('./images');

// 初始化MNIST文件头部
const header = Buffer.alloc(16);
header.writeUInt32BE(2051, 0); // 魔数
header.writeUInt32BE(imageFiles.length, 4); // 样本数
header.writeUInt32BE(28, 8); // 图像高度
header.writeUInt32BE(28, 12); // 图像宽度

// 创建输出文件并写入头部
const outputStream = fs.createWriteStream('train-images-idx3-ubyte');
outputStream.write(header);

// 遍历图像文件列表，并将每个图像写入输出文件
for (let i = 0; i < imageFiles.length; i++) {
  const filePath = `./images/${imageFiles[i]}`;
  const buffer = fs.readFileSync(filePath);
  const pixels = decodeJpeg(buffer);
  outputStream.write(pixels);
}

// 关闭输出文件流
outputStream.close();
