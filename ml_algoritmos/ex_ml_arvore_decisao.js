// Função para calcular a entropia
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

// Função para dividir os dados baseado num atributo e valor
function splitData(data, attribute, value) {
    const left = data.filter(item => item[attribute] <= value);
    const right = data.filter(item => item[attribute] > value);
    return { left, right };
}

// Função principal que cria a árvore de decisão
function decisionTree(data, attributes) {
    // Se os dados possuem apenas uma classe, retornamos a classe
    const classes = data.map(item => item.label);
    if (new Set(classes).size === 1) {
        return { label: classes[0] };
    }

    // Se não houver mais atributos para dividir, retornamos a classe majoritária
    if (attributes.length === 0) {
        const majorityClass = classes.sort((a, b) =>
            classes.filter(v => v === a).length - classes.filter(v => v === b).length
        ).pop();
        return { label: majorityClass };
    }

    // Procuramos o melhor atributo para dividir
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

    // Recursivamente construímos a árvore
    const leftBranch = decisionTree(bestSplit.left, attributes.filter(attr => attr !== bestAttribute));
    const rightBranch = decisionTree(bestSplit.right, attributes.filter(attr => attr !== bestAttribute));

    return {
        attribute: bestAttribute,
        value: bestValue,
        left: leftBranch,
        right: rightBranch,
    };
}

// Exemplo de uso
const dataset = [
    { age: 25, income: 50000, label: 'No' },
    { age: 30, income: 60000, label: 'No' },
    { age: 45, income: 80000, label: 'Yes' },
    { age: 35, income: 70000, label: 'Yes' },
    { age: 50, income: 90000, label: 'Yes' }
];

const attributes = ['age', 'income'];
const tree = decisionTree(dataset, attributes);
console.log(JSON.stringify(tree, null, 2));

/* Explicação do Algoritmo:
Entropia: A função calculateEntropy calcula a entropia de um conjunto de dados com base em suas classes. Menor entropia significa que os dados são mais homogêneos.
Divisão de dados: A função splitData divide os dados em dois subconjuntos baseados em um valor de um atributo.
Construção da árvore: A função decisionTree procura o melhor atributo e valor para dividir os dados, com base no ganho de informação (diferença entre a entropia atual e a entropia após a divisão).
Recursão: A árvore é construída de maneira recursiva, até que as classes nos nós sejam puras ou não haja mais atributos para dividir.*/
