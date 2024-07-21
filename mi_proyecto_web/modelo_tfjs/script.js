let model;

async function loadModel() {
    model = await tf.loadLayersModel('modelo_tfjs/model.json');
    console.log("Modelo cargado");
}

loadModel();

function preprocessImage(image) {
    // Preprocesa la imagen para que tenga la forma correcta
    let tensor = tf.browser.fromPixels(image)
        .resizeNearestNeighbor([28, 28]) // cambiar el tamaño de la imagen
        .mean(2) // convertir a escala de grises
        .expandDims(2) // añadir dimensión extra
        .expandDims() // añadir otra dimensión para el batch
        .toFloat();
    return tensor.div(255.0); // normalizar
}

async function predict() {
    let canvas = document.getElementById('canvas');
    let tensor = preprocessImage(canvas);

    let predictions = await model.predict(tensor).data();
    let resultado = Array.from(predictions)
        .map((p, i) => ({ probabilidad: p, clase: i }))
        .sort((a, b) => b.probabilidad - a.probabilidad)
        .slice(0, 1); // Obtener la clase con mayor probabilidad

    document.getElementById('resultado').innerText = `Predicción: ${resultado[0].clase} con probabilidad de ${resultado[0].probabilidad.toFixed(2)}`;
}