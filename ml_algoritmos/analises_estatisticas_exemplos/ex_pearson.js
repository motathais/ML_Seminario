function pearsonCorrelation(x, y) {
    if (x.length !== y.length) {
        throw new Error("As listas de dados devem ter o mesmo comprimento.");
    }
 
    const n = x.length;
    const sumX = x.reduce((acc, val) => acc + val, 0);
    const sumY = y.reduce((acc, val) => acc + val, 0);
    const sumXY = x.reduce((acc, val, i) => acc + val * y[i], 0);
    const sumX2 = x.reduce((acc, val) => acc + val ** 2, 0);
    const sumY2 = y.reduce((acc, val) => acc + val ** 2, 0);
 
    const numerator = (n * sumXY) - (sumX * sumY);
    const denominator = Math.sqrt((n * sumX2 - sumX ** 2) * (n * sumY2 - sumY ** 2));
 
    if (denominator === 0) return 0;
 
    return numerator / denominator;
}
 
// Exemplo de uso
const x = [1, 2, 3, 4, 5];
const y = [2, 4, 6, 8, 10];
 
const correlation = pearsonCorrelation(x, y);
console.log("Coeficiente de correlação de Pearson:", correlation);