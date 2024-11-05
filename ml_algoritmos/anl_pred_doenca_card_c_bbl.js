const { RandomForestClassifier } = require('ml-random-forest');
const fs = require('fs');
const { parse } = require('csv-parse/sync');

// Função para carregar e normalizar os dados
function loadData(filePath) {
  const fileContent = fs.readFileSync(filePath, 'utf8');
  const records = parse(fileContent, {
    columns: true,
    skip_empty_lines: true
  });

  const features = [];
  const labels = [];

  for (let row of records) {
    // Converter os valores dos atributos em números e normalizar entre 0 e 1
    const normalizedRow = Object.values(row).slice(0, -1).map(Number);
    features.push(normalizedRow);
    labels.push(Number(row['hd'])); // Supondo que 'hd' é a última coluna
  }

  return { features, labels };
}

// Carregar bases de dados de treino e teste
const trainData = loadData('./treino.csv');
const testData = loadData('./teste.csv');

// Configurar o modelo Random Forest
const options = {
  nEstimators: 100,      // Número de árvores na floresta
  maxFeatures: 0.8,      // Proporção de features a serem usadas em cada árvore
  replacement: true,     // Usar reposição no bootstrap
  seed: 42               // Semente para reprodutibilidade
};

const classifier = new RandomForestClassifier(options);

// Treinar o modelo com a base de treino
classifier.train(trainData.features, trainData.labels);

// Fazer previsões com a base de teste
const predictions = classifier.predict(testData.features);

// Avaliar o modelo calculando a acurácia
let correctPredictions = 0;
for (let i = 0; i < predictions.length; i++) {
  if (predictions[i] === testData.labels[i]) {
    correctPredictions++;
  }
}

const accuracy = (correctPredictions / predictions.length) * 100;
console.log(`Acurácia do modelo: ${accuracy.toFixed(2)}%`);
