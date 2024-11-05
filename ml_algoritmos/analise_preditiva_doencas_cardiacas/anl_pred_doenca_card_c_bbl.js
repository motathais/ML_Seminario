// Algoritmo que utiliza biblioteca ml-random-forest para análise preditiva de doenças cardíacas com Random Forest

const { RandomForestClassifier } = require('ml-random-forest'); // Importa a classe RandomForestClassifier da biblioteca ml-random-forest para construir o modelo de Random Forest.
const fs = require('fs'); // Importa o módulo fs para manipulação de arquivos.
const { parse } = require('csv-parse/sync'); // Importa a função parse da biblioteca csv-parse para ler arquivos CSV de forma síncrona.

// Função para carregar e normalizar os dados
function loadData(filePath) {
  const fileContent = fs.readFileSync(filePath, 'utf8'); // Lê o conteúdo do arquivo CSV no caminho especificado.
  const records = parse(fileContent, {
    columns: true, // Define que cada linha deve ser interpretada como um objeto com chaves baseadas nos cabeçalhos.
    skip_empty_lines: true // Ignora linhas vazias no CSV.
  });

  const features = []; // Array para armazenar os atributos (características) de cada amostra.
  const labels = []; // Array para armazenar as etiquetas (classes) de cada amostra.

  for (let row of records) {
    // Converter os valores dos atributos em números e normalizar entre 0 e 1
    const normalizedRow = Object.values(row).slice(0, -1).map(Number); // Extrai os valores dos atributos, converte-os para números e exclui a última coluna.
    features.push(normalizedRow); // Adiciona a linha normalizada no array de features.
    labels.push(Number(row['hd'])); // Supondo que 'hd' é a última coluna, extrai seu valor como etiqueta e converte para número.
  }

  return { features, labels }; // Retorna um objeto contendo os atributos (features) e etiquetas (labels).
}

// Carregar bases de dados de treino e teste
const trainData = loadData('../bases/treino.csv'); // Carrega a base de dados de treino a partir do arquivo CSV.
const testData = loadData('../bases/teste.csv'); // Carrega a base de dados de teste a partir do arquivo CSV.

// Configurar o modelo Random Forest
const options = {
  nEstimators: 100,      // Número de árvores na floresta, define a quantidade de árvores a serem usadas.
  maxFeatures: 0.8,      // Proporção das características (features) a serem usadas em cada árvore (80%).
  replacement: true,     // Define que amostras podem ser reutilizadas no bootstrap (reposição).
  seed: 42               // Semente para garantir a reprodutibilidade dos resultados.
};

const classifier = new RandomForestClassifier(options); // Cria o modelo de Random Forest com as opções configuradas.

// Treinar o modelo com a base de treino
classifier.train(trainData.features, trainData.labels); // Treina o modelo utilizando os atributos e etiquetas da base de treino.

// Fazer previsões com a base de teste
const predictions = classifier.predict(testData.features); // Gera previsões com o modelo usando os atributos da base de teste.

// Avaliar o modelo calculando a acurácia
let correctPredictions = 0; // Inicializa uma variável para contar previsões corretas.
for (let i = 0; i < predictions.length; i++) {
  if (predictions[i] === testData.labels[i]) { // Compara cada previsão com a etiqueta real correspondente.
    correctPredictions++; // Incrementa o contador de previsões corretas.
  }
}

const accuracy = (correctPredictions / predictions.length) * 100; // Calcula a acurácia dividindo o número de previsões corretas pelo total.
console.log(`Acurácia do modelo: ${accuracy.toFixed(2)}%`); // Exibe a acurácia do modelo formatada com duas casas decimais.
