import numpy as np


class CTC(object):
    """CTC class."""

    def __init__(self, BLANK=0):
        """Initialize instance variables.

        Argument
        --------
        blank: (int, optional)
                blank label index. Default 0.

        """
        self.BLANK = BLANK

    def targetWithBlank(self, target):
        """Extend target sequence with blank.

        Input
        -----
        target: (np.array, dim = 1)
                target output

        Return
        ------
        extSymbols: (np.array, dim = 1)
                    extended label sequence with blanks
        skipConnect: (np.array, dim = 1)
                    skip connections

        """
        extSymbols = np.ones((len(target) * 2 + 1)) * self.BLANK
        skipConnect = np.zeros((len(target) * 2 + 1))

        # -------------------------------------------->

        # Your Code goes here
        for i in range(len(target)):
            extSymbols[2*i+1] = target[i]
            if i > 0 and target[i] != target[i-1]:
                skipConnect[2*i+1] = 1
        # <---------------------------------------------

        return extSymbols.astype(int), skipConnect.astype(int)

    def forwardProb(self, logits, extSymbols, skipConnect):
        """Compute forward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, channel))
                predict (log) probabilities

        extSymbols: (np.array, dim = 1)
                    extended label sequence with blanks

        skipConnect: (np.array, dim = 1)
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (output len, out channel))
                forward probabilities

        """
        S, T = len(extSymbols), len(logits)
        alpha = np.zeros(shape=(T, S))

        # -------------------------------------------->

        # Your Code goes here
        alpha[0][0] = logits[0][extSymbols[0]]
        alpha[0][1] = logits[0][extSymbols[1]]
        for t in range(1, T):
            alpha[t][0] = alpha[t-1][0] * logits[t][extSymbols[0]]
            for i in range(1, S):
                alpha[t][i] = alpha[t-1][i-1] + alpha[t-1][i]
                if skipConnect[i]:
                    alpha[t][i] += alpha[t-1][i-2]
                alpha[t][i] *= logits[t][extSymbols[i]]
        # <---------------------------------------------

        return alpha

    def backwardProb(self, logits, extSymbols, skipConnect):
        """Compute backward probabilities.

        Input
        -----

        logits: (np.array, dim = (input_len, channel))
                predict (log) probabilities

        extSymbols: (np.array, dim = 1)
                    extended label sequence with blanks

        skipConnect: (np.array, dim = 1)
                    skip connections

        Return
        ------
        beta: (np.array, dim = (output len, out channel))
                backward probabilities

        """
        S, T = len(extSymbols), len(logits)
        beta = np.zeros(shape=(T, S))

        # -------------------------------------------->

        # Your Code goes here
        beta[T-1][S-1] = 1
        beta[T-1][S-2] = 1
        for t in range(T-2, -1, -1):
            beta[t][S-1] = beta[t+1][S-1] * logits[t+1][extSymbols[S-1]]
            for i in range(S-2, -1, -1):
                beta[t][i] = beta[t+1][i] * logits[t+1][extSymbols[i]] + beta[t+1][i+1] * logits[t+1][extSymbols[i+1]]
                if i <= S-3 and skipConnect[i+2]:
                    beta[t][i] += beta[t+1][i+2] * logits[t+1][extSymbols[i+2]]
        # <---------------------------------------------

        return beta

    def postProb(self, alpha, beta):
        """Compute posterior probabilities.

        Input
        -----
        alpha: (np.array)
                forward probability

        beta: (np.array)
                backward probability

        Return
        ------
        gamma: (np.array)
                posterior probability

        """
        gamma = None

        # -------------------------------------------->

        # Your Code goes here
        temp = alpha * beta
        gamma = temp / temp.sum(axis=1, keepdims=True)
        # <---------------------------------------------

        return gamma
