const express = require('express');
const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const path = require('path');

//使用gpu
// const tf = require('@tensorflow/tfjs-node');

const app = express();

const IMG_WIDTH = 224;
const IMG_HEIGHT = 224;
const NUM_CLASSES = 10;
const BATCH_SIZE = 32;
const NUM_EPOCHS = 10;

const dataDir = path.join(__dirname, './static/archive');

// 定义图像预处理函数
function preprocessImage(imgBuffer) {
  const pixels = tf.node.decodeImage(imgBuffer, 3);
  const resized = tf.image.resizeBilinear(pixels, [IMG_HEIGHT, IMG_WIDTH]);
  const scaled = resized.div(255.0);
  return scaled;
}

// 创建数据集
const dataset = tf.data.Dataset(`file://${dataDir}/*/*/*.jpg`)
//   .shuffle(1000)
//   .map(filePath => {
//     const parts = tf.node.path.basename(filePath).split('/');
//     const label = parseInt(parts[0], 10);
//     const imgBuffer = fs.readFileSync(filePath);
//     const img = preprocessImage(imgBuffer);
//     return { x: img, y: tf.oneHot(tf.tensor1d([label]), NUM_CLASSES) };
//   })
//   .batch(BATCH_SIZE)
//   .repeat(NUM_EPOCHS);

  console.log(dataset);



// console.log(tf);
// const digitalImgArr = []
// const diaitalLabelArr = []
// const archiveDir = fs.readdirSync(path.join(__dirname, './static/archive'));
// archiveDir.forEach((dir) => {
//     const digitalDir = fs.readdirSync(path.join(__dirname, './static/archive/' + dir));
//     digitalDir.forEach((file, index) => {
       
//         const imgPath = path.join(__dirname, './static/archive/' + dir + '/' + file);
//         const imgBuffer = fs.readFileSync(imgPath);
//         const picels = tfNode.node.decodeImage(imgBuffer, 3);
//         digitalImgArr.push(picels.dataSync())
//         const labelArr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
//         labelArr[dir] = 1;
//         diaitalLabelArr.push(labelArr);
       

//     })
// })

// let trainImg, trainLab;
// function tenserData() {
//     // console.log(digitalImgArr)
//     // const trainImg_org = tf.tensor2d(digitalImgArr);
//     // trainImg_org.print();
//     trainLab = tf.tensor2d(diaitalLabelArr.slice(0,1000),[1000,10]);
//     // trainImg = tf.mul(tf.sub(trainImg_org, tf.scalar(127.5)), tf.scalar(1 / 127.5));

//     // console.log(trainImg_org)
// }

// run()
async function run() {
    const model = createModel();
    tenserData();

    const result = await model.fit(trainImg, trainLab, {
        epochs: 20,
        batchSize: 128,
        validationSplit: 0.2,
        callbacks: {
            onEpochEnd: (eopchs,log) => {console.log(eopchs,log)},
            // onBatchEnd
        }
    })
    model.summary();
}


// let model;
function createModel() {

    let model = tf.sequential();
    //输入层
    // model.add(tf.input({shape:[28,28,1]}));
    //unit1
    model.add(tf.layers.conv2d({ inputShape: [28, 28, 1], filters: 16, kernelSize: 5, strides: 1, activation: 'relu', kernetInitializer: 'varianceScaling', }));
    // model.add(tf.layers.conv2d({ filters:16, kernelSize:3, activation:'relu',}));
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2], }));

    model.add(tf.layers.conv2d({ filters: 32, kernelSize: 5, strides: 1, activation: 'relu', kernetInitializer: 'varianceScaling', }));
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2], }));


    //unit2
    model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu', padding: 'same' }));
    // model.add(tf.layers.conv2d({ filters:32, kernelSize:3, activation:'relu', padding: 'same'}));
    model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2, padding: 'same' }));
    // // model.add(tf.layers.dropout({ rate:0.5 }));
    // // //unit3
    // model.add(tf.layers.conv2d({ filters:64, kernelSize:3, activation:'relu', padding: 'same'}));
    model.add(tf.layers.conv2d({ filters: 128, kernelSize: 3, activation: 'relu', padding: 'same' }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2, padding: 'same' }));
    // // //unit4
    // model.add(tf.layers.conv2d({ filters:128, kernelSize:3, activation:'relu', padding: 'same'}));
    // model.add(tf.layers.conv2d({ filters:128, kernelSize:3, activation:'relu', padding: 'same'}));
    // model.add(tf.layers.maxPooling2d({ poolSize:2, strides: 2, padding:'same'}));
    // // model.add(tf.layers.dropout({ rate:0.5 }));
    // // //unit5
    // model.add(tf.layers.conv2d({ filters:256, kernelSize:3, activation:'relu', padding: 'same'}));
    // model.add(tf.layers.conv2d({ filters:256, kernelSize:3, activation:'relu', padding: 'same'}));
    // model.add(tf.layers.maxPooling2d({ poolSize:2, strides: 2, padding:'same'}));

    model.add(tf.layers.dropout({ rate: 0.5 }));
    //全连接层
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 128, activation: "relu" }));
    // model.add(tf.layers.dropout({ rate:0.5 }));
    // model.add(tf.layers.dense({ units: 128,activation: "relu"}));
    // model.add(tf.layers.dropout({ rate:0.5 }));

    //输出层
    model.add(tf.layers.dense({ units: 10 }));

    const optimizer = tf.train.adam(0.001);
    model.compile({
        loss: tf.losses.softmaxCrossEntropy,
        // loss: "categoricalCrossentropy",
        optimizer,
        metrics: ['accuracy']
    })

    return model;
}

// JSON.stringify({
//     digitalImgArr,
//     diaitalLabelArr
// })
// fs.writeFileSync(path.join("./static/digitalData.json", digitalImgArr.toString()))


// app.get("/digitalData",(req,res) => {
//     console.log(digitalImgArr,diaitalLabelArr)
//     res.send({
//         digitalImgArr,
//         diaitalLabelArr
//     })
// })




app.listen(8085, () => console.log('server is running on port 8085'));