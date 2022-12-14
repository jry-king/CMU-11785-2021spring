import numpy as np
from ctc import *


class CTCLoss(object):
    """CTC Loss class."""

    def __init__(self, BLANK=0):
        """Initialize instance variables.

        Argument:
                blank (int, optional) – blank label index. Default 0.
        """
        # -------------------------------------------->
        # Don't Need Modify
        super(CTCLoss, self).__init__()
        self.BLANK = BLANK
        self.gammas = []
        # <---------------------------------------------

    def __call__(self, logits, target, input_lengths, target_lengths):
        # -------------------------------------------->
        # Don't Need Modify
        return self.forward(logits, target, input_lengths, target_lengths)
        # <---------------------------------------------

    def forward(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward.

        Computes the CTC Loss.

        Input
        -----
        logits: (seqlength, batch_size, len(Symbols))
                log probabilities (output sequence) from the RNN/GRU

        target: (batch_size, paddedtargetlen)
                target sequences.

        input_lengths: (batch_size,)
                        lengths of the inputs.

        target_lengths: (batch_size,)
                        lengths of the target.

        Returns
        -------
        loss: scalar
            (avg) divergence between the posterior probability γ(t,r) and the input symbols (y_t^r)

        """
        # -------------------------------------------->
        # Don't Need Modify
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths
        # <---------------------------------------------

        #####  Attention:
        #####  Output losses will be divided by the target lengths
        #####  and then the mean over the batch is taken

        # -------------------------------------------->
        # Don't Need Modify
        B, _ = target.shape
        totalLoss = np.zeros(B)
        # <---------------------------------------------

        for b in range(B):
            # -------------------------------------------->
            # Computing CTC Loss for single batch
            # Process:
            #     Extend Sequence with blank ->
            #     Compute forward probabilities ->
            #     Compute backward probabilities ->
            #     Compute posteriors using total probability function
            #     Compute Expected Divergence and take average on batches
            # <---------------------------------------------

            # -------------------------------------------->

            # Your Code goes here
            current_logit = logits[:input_lengths[b], b]
            current_target = target[b, :target_lengths[b]]
            calculator = CTC()
            ext_symbols, skip_connect = calculator.targetWithBlank(current_target)
            alpha = calculator.forwardProb(current_logit, ext_symbols, skip_connect)
            beta = calculator.backwardProb(current_logit, ext_symbols, skip_connect)
            gamma = calculator.postProb(alpha, beta)
            self.gammas.append(gamma)
            losses = gamma.copy()
            for r in range(losses.shape[1]):
                losses[:,r] *= np.log(current_logit[:,ext_symbols[r]])
            totalLoss[b] = -1 * np.sum(losses)
            # <---------------------------------------------

        return np.mean(totalLoss)

    def backward(self):
        """CTC loss backard.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        logits: (seqlength, batch_size, len(Symbols))
                log probabilities (output sequence) from the RNN/GRU

        target: (batch_size, paddedtargetlen)
                target sequences.

        input_lengths: (batch_size,)
                        lengths of the inputs.

        target_lengths: (batch_size,)
                        lengths of the target.

        Returns
        -------
        dY: scalar
            derivative of divergence wrt the input symbols at each time.

        """
        # -------------------------------------------->
        # Don't Need Modify
        T, B, C = self.logits.shape
        dY = np.full_like(self.logits, 0)
        # <---------------------------------------------


        for b in range(B):
            # -------------------------------------------->
            # Computing CTC Derivative for single batch
            # <---------------------------------------------

            # -------------------------------------------->

            # Your Code goes here
            current_logit = self.logits[:self.input_lengths[b], b]
            current_target = self.target[b, :self.target_lengths[b]]
            calculator = CTC()
            ext_symbols, _ = calculator.targetWithBlank(current_target)
            for t in range(len(current_logit)):
                for i in range(len(ext_symbols)):
                    dY[t][b][ext_symbols[i]] -= self.gammas[b][t][i] / current_logit[t][ext_symbols[i]]
            # <---------------------------------------------

        return dY
