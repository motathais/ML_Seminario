/*algoritmo naive bayes*/

// Função para calcular a probabilidade de cada classe
function calculateClassProbabilities(summaries, input) {
    let probabilities = {};

    for (let classValue in summaries) {
        probabilities[classValue] = summaries[classValue].probability; // Probabilidade anterior da classe
        let classSummaries = summaries[classValue].summaries;

        classSummaries.forEach((summary, i) => {
            let mean = summary.mean;
            let stddev = summary.stddev;
            let x = input[i];
            probabilities[classValue] *= gaussianProbability(x, mean, stddev);
        });
    }

    return probabilities;
}

// Função para calcular a probabilidade gaussiana (distribuição normal)
function gaussianProbability(x, mean, stddev) {
    const exponent = Math.exp(-((x - mean) ** 2) / (2 * stddev ** 2));
    return (1 / (Math.sqrt(2 * Math.PI) * stddev)) * exponent;
}

// Função para resumir o conjunto de dados (cálculo de médias e desvios padrão por atributo)
function summarizeByClass(dataset) {
    let separated = separateByClass(dataset);
    let summaries = {};

    for (let classValue in separated) {
        let instances = separated[classValue];
        summaries[classValue] = {
            probability: instances.length / dataset.length,
            summaries: summarizeAttributes(instances)
        };
    }

    return summaries;
}

// Função para separar os dados por classe
function separateByClass(dataset) {
    let separated = {};
    dataset.forEach(row => {
        let classValue = row[row.length - 1];
        if (!separated[classValue]) {
            separated[classValue] = [];
        }
        separated[classValue].push(row);
    });
    return separated;
}

// Função para calcular a média e o desvio padrão de cada atributo
function summarizeAttributes(instances) {
    let attributes = instances[0].length - 1; // Exclui o rótulo (última coluna)
    let summaries = [];

    for (let i = 0; i < attributes; i++) {
        let attributeValues = instances.map(row => row[i]);
        let mean = attributeValues.reduce((acc, val) => acc + val, 0) / attributeValues.length;
        let variance = attributeValues.reduce((acc, val) => acc + (val - mean) ** 2, 0) / attributeValues.length;
        let stddev = Math.sqrt(variance);
        summaries.push({ mean, stddev });
    }

    return summaries;
}

// Função para fazer a predição com base nas probabilidades calculadas
function predict(summaries, input) {
    let probabilities = calculateClassProbabilities(summaries, input);
    let bestLabel = null;
    let bestProb = -1;

    for (let classValue in probabilities) {
        if (bestLabel === null || probabilities[classValue] > bestProb) {
            bestProb = probabilities[classValue];
            bestLabel = classValue;
        }
    }

    return bestLabel;
}

// Função principal que treina o modelo e faz previsões
function naiveBayes(train, test) {
    let summaries = summarizeByClass(train);
    let predictions = test.map(input => predict(summaries, input));
    return predictions;
}

// Exemplo de uso
const dataset = [
    [3.393533211, 2.331273381, 'A'],
    [3.110073483, 1.781539638, 'A'],
    [1.343808831, 3.368360954, 'B'],
    [3.582294042, 4.67917911, 'B'],
    [2.280362439, 2.866990263, 'A'],
    [7.423436942, 4.696522875, 'B'],
    [5.745051997, 3.533989803, 'B'],
    [9.172168622, 2.511101045, 'A'],
    [7.792783481, 3.424088941, 'A']
];

const testSet = [
    [3.0, 2.0],
    [7.5, 3.0]
];

// Treina o modelo com o dataset e faz predições no conjunto de teste
const predictions = naiveBayes(dataset, testSet);
console.log(predictions);

/*
Explicação do Algoritmo
Treinamento:

A função summarizeByClass separa o conjunto de dados por classe e calcula a média e o desvio padrão de cada atributo para cada classe.
Essas estatísticas são armazenadas como "resumos" que serão usados para calcular as probabilidades na fase de predição.
Probabilidades Gaussianas:

A função gaussianProbability usa a fórmula da distribuição normal para calcular a probabilidade de um atributo ter determinado valor, dado sua média e desvio padrão.
Predição:

A função predict usa as probabilidades calculadas para cada classe e faz a predição final, selecionando a classe com a maior probabilidade.
Predições Finais:

A função naiveBayes retorna as predições feitas para um conjunto de teste com base no modelo treinado.*/


