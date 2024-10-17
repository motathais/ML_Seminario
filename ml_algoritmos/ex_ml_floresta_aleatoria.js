/*algoritmo floresta aleatoria*/

// Função auxiliar para obter um subconjunto aleatório dos dados
function bootstrapSample(data, n) {
    const sample = [];
    for (let i = 0; i < n; i++) {
        const index = Math.floor(Math.random() * data.length);
        sample.push(data[index]);
    }
    return sample;
}

// Função para criar uma árvore de decisão (usando o mesmo código de árvore de decisão anterior)
function decisionTree(data, attributes) {
    const classes = data.map(item => item.label);
    if (new Set(classes).size === 1) {
        return { label: classes[0] };
    }

    if (attributes.length === 0) {
        const majorityClass = classes.sort((a, b) =>
            classes.filter(v => v === a).length - classes.filter(v => v === b).length
        ).pop();
        return { label: majorityClass };
    }

    let bestAttribute = null;
    let bestValue = null;
    let bestGain = -Infinity;
    let bestSplit = null;

    for (const attribute of attributes) {
        const values = new Set(data.map(item => item[attribute]));
        for (const value of values) {
            const { left, right } = splitData(data, attribute, value);
            if (left.length === 0 || right.length === 0) continue;

            const entropyLeft = calculateEntropy(left.map(item => item.label));
            const entropyRight = calculateEntropy(right.map(item => item.label));
            const totalEntropy = (left.length / data.length) * entropyLeft + (right.length / data.length) * entropyRight;

            const informationGain = calculateEntropy(data.map(item => item.label)) - totalEntropy;

            if (informationGain > bestGain) {
                bestGain = informationGain;
                bestAttribute = attribute;
                bestValue = value;
                bestSplit = { left, right };
            }
        }
    }

    const leftBranch = decisionTree(bestSplit.left, attributes.filter(attr => attr !== bestAttribute));
    const rightBranch = decisionTree(bestSplit.right, attributes.filter(attr => attr !== bestAttribute));

    return {
        attribute: bestAttribute,
        value: bestValue,
        left: leftBranch,
        right: rightBranch,
    };
}

// Função para fazer previsões usando uma árvore de decisão
function predict(tree, input) {
    if (tree.label) {
        return tree.label;
    }

    if (input[tree.attribute] <= tree.value) {
        return predict(tree.left, input);
    } else {
        return predict(tree.right, input);
    }
}

// Função para calcular a entropia (mesmo código da árvore de decisão)
function calculateEntropy(data) {
    const total = data.length;
    const counts = data.reduce((acc, item) => {
        acc[item] = (acc[item] || 0) + 1;
        return acc;
    }, {});

    return Object.keys(counts).reduce((entropy, key) => {
        const p = counts[key] / total;
        return entropy - p * Math.log2(p);
    }, 0);
}

// Função para dividir os dados
function splitData(data, attribute, value) {
    const left = data.filter(item => item[attribute] <= value);
    const right = data.filter(item => item[attribute] > value);
    return { left, right };
}

// Função para criar uma floresta de árvores de decisão
function randomForest(data, attributes, nTrees) {
    const forest = [];

    for (let i = 0; i < nTrees; i++) {
        const sample = bootstrapSample(data, data.length);
        const tree = decisionTree(sample, attributes);
        forest.push(tree);
    }

    return forest;
}

// Função para fazer a predição final com base nas árvores (voto majoritário)
function predictForest(forest, input) {
    const predictions = forest.map(tree => predict(tree, input));
    const counts = predictions.reduce((acc, label) => {
        acc[label] = (acc[label] || 0) + 1;
        return acc;
    }, {});

    return Object.keys(counts).reduce((a, b) => (counts[a] > counts[b] ? a : b));
}

// Exemplo de uso
const dataset = [
    { age: 25, income: 50000, label: 'No' },
    { age: 30, income: 60000, label: 'No' },
    { age: 45, income: 80000, label: 'Yes' },
    { age: 35, income: 70000, label: 'Yes' },
    { age: 50, income: 90000, label: 'Yes' },
    { age: 22, income: 40000, label: 'No' },
    { age: 37, income: 75000, label: 'Yes' }
];

const attributes = ['age', 'income'];
const forest = randomForest(dataset, attributes, 5);

// Conjunto de teste
const testSet = [
    { age: 26, income: 51000 },
    { age: 40, income: 85000 }
];

testSet.forEach(input => {
    const prediction = predictForest(forest, input);
    console.log(`Predição: ${prediction}`);
});

/*Explicação do Algoritmo
Amostragem com Reposição (Bootstrap):

A função bootstrapSample cria um subconjunto aleatório dos dados de treinamento para cada árvore. Isso significa que cada árvore é treinada em dados ligeiramente diferentes.
Árvore de Decisão:

A função decisionTree constrói uma árvore de decisão para classificar os dados, da mesma forma que a árvore de decisão tradicional.
Comitê de Árvores (Floresta):

A função randomForest cria várias árvores (definido por nTrees), cada uma usando um subconjunto dos dados.
Voto Majoritário:

A função predictForest faz a predição final. Ela coleta as predições de todas as árvores e retorna a classe mais votada. */