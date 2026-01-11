import numpy as np
import Function as f


class PCA:
    def __init__(self, X):
        self.X = X
        # Pasul 1: Standardizarea datelor folosind modulul Functii
        self.X_std = f.standardizare(X)

        # Pasul 2: Calcularea matricei de covarianță (pe date standardizate = matrice de corelație)
        self.cov = np.cov(m=self.X_std, rowvar=False)

        # Pasul 3: Calcularea valorilor și vectorilor proprii
        valori, vectori = np.linalg.eigh(a=self.cov)

        # Pasul 4: Sortarea descrescătoare a valorilor proprii (logica componentelor principale)
        k_desc = [k for k in reversed(np.argsort(valori))]

        self.alpha = valori[k_desc]  # Valorile proprii (Varianța explicată)
        self.vector_valori_proprii = vectori[:, k_desc]  # Vectorii proprii (Direcțiile)

        # Pasul 5: Calcularea componentelor principale (Scorurile brute)
        self.com = self.X_std @ self.vector_valori_proprii

    def getAlpha(self):
        """Returnează valorile proprii (Alpha)."""
        return self.alpha

    def getComponente(self):
        """Returnează matricea componentelor principale."""
        return self.com

    def getScores(self):
        """Calculează scorurile standardizate (Componente / radical din Alpha)."""
        return self.com / np.sqrt(self.alpha)

    def getRxc(self):
        """Calculează matricea de corelație între variabilele originale și componente (Factor Loadings)."""
        return self.vector_valori_proprii * np.sqrt(self.alpha)

    def getComunalitati(self):
        """Calculează comunalitățile (pătratul coeficienților de corelație cumulați)."""
        Rxc = self.getRxc()
        return np.cumsum(Rxc ** 2, axis=1)