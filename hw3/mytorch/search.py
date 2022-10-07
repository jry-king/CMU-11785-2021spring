import numpy as np


def GreedySearch(SymbolSets, y_probs):
    """Greedy Search.

    Input
    -----
    SymbolSets: list
                all the symbols (the vocabulary without blank)

    y_probs: (# of symbols + 1, Seq_length, batch_size)
            Your batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size.

    Returns
    ------
    forward_path: str
                the corresponding compressed symbol sequence i.e. without blanks
                or repeated symbols.

    forward_prob: scalar (float)
                the forward probability of the greedy path

    """
    # Follow the pseudocode from lecture to complete greedy search :-)

    _, T, B = y_probs.shape
    logits = y_probs[:, :, 0]           # batch_size = 1, just get the sample out
    best_path = []
    forward_path = ""
    forward_prob = 1
    for t in range(T):
        index = np.argmax(logits[:, t])
        if best_path == [] or best_path[-1] != index:
            best_path.append(index)
        forward_prob *= logits[index, t]
    while 0 in best_path:
        best_path.remove(0)
    for index in best_path:
        forward_path += SymbolSets[index-1]
    return (forward_path, forward_prob)

##############################################################################


def BeamSearch(SymbolSets, y_probs, BeamWidth):
    """Beam Search.

    Input
    -----
    SymbolSets: list
                all the symbols (the vocabulary without blank)

    y_probs: (# of symbols + 1, Seq_length, batch_size)
            Your batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size.

    BeamWidth: int
                Width of the beam.

    Return
    ------
    bestPath: str
            the symbol sequence with the best path score (forward probability)

    mergedPathScores: dictionary
                        all the final merged paths with their scores.

    """
    # Follow the pseudocode from lecture to complete beam search :-)

    # setup
    # use strings to represent paths, and None for path composed of blanks
    # all paths are already compressed, and blanks are not explicitly represented, so separate paths ending with symbols or blanks
    # sets used to keep pruned paths ending with valid symbols or blanks at each iteration (PathsWithTerminalSymbol/Blank in slides)
    paths = set()
    paths_blank = set()
    # dicts used to keep scores of pruned paths ((Blank)PathScore in slides)
    path_score = {}
    path_score_blank = {}
    # sets used to keep new paths extended from last iteration, ending with valid symbols or blanks (NewPathsWithTerminalSymbol/Blank in slides)
    # separate from paths(_blank) since extension includes two steps, both of which takes both kinds of paths as input
    new_paths = set()
    new_paths_blank = set()
    # dicts used to keep scores of the new paths (New(Blank)PathScore in slides)
    new_path_score = {}
    new_path_score_blank = {}
    # batch_size = 1, just get the logits of the sample out (number of symbols + 1, seq_len)
    logits = y_probs[:, :, 0]
    _, T = logits.shape

    # initialize paths with each symbols including blank, and their scores with scores at t=0
    # can treat as first extension
    new_paths_blank.add(None)
    new_path_score_blank[None] = logits[0, 0]
    for i, c in enumerate(SymbolSets):
        new_paths.add(c)
        new_path_score[c] = logits[i+1, 0]

    # start iteration
    # perform T-1 prunings and extensions to get final (unpruned) paths
    for t in range(1, T):
        # prune, get the top {beam_width} scores and paths
        # new_paths(_blank), new_path_score(_blank) -> paths(_blank), path_score(_blank)
        path_score.clear()
        path_score_blank.clear()
        scorelist = []
        for p in new_paths:
            scorelist.append(new_path_score[p])
        for p in new_paths_blank:
            scorelist.append(new_path_score_blank[p])
        scorelist.sort(reverse=True)
        cutoff = scorelist[BeamWidth-1] if BeamWidth-1 < len(scorelist) else scorelist[-1] # lowest acceptable score
        paths.clear()
        for p in new_paths:
            if new_path_score[p] >= cutoff:
                paths.add(p)
                path_score[p] = new_path_score[p]
        paths_blank.clear()
        for p in new_paths_blank:
            if new_path_score_blank[p] >= cutoff:
                paths_blank.add(p)
                path_score_blank[p] = new_path_score_blank[p]
        
        # extend with symbol
        # paths(_blank), path_score(_blank) -> new_paths, new_path_score
        new_paths.clear()
        new_path_score.clear()
        # extend path ending with blanks, which always change the sequence
        for p in paths_blank:
            for i, c in enumerate(SymbolSets):
                p_new = p + c if p != None else c       # need to remove None if a symbol is added
                new_paths.add(p_new)
                new_path_score[p_new] = path_score_blank[p] * logits[i+1, t]
        # extend path ending with symbols
        for p in paths:
            for i, c in enumerate(SymbolSets):
                p_new = p if c == p[-1] else p + c
                if p_new in new_paths:
                    new_path_score[p_new] += path_score[p] * logits[i+1, t]
                else:
                    new_paths.add(p_new)
                    new_path_score[p_new] = path_score[p] * logits[i+1, t]
        
        # extend with blank, which doesn't change the compressed sequence
        # paths(_blank), path_score(_blank) -> new_paths_blank, new_path_score_blank
        new_paths_blank.clear()
        new_path_score_blank.clear()
        # extend path ending with blanks
        for p in paths_blank:
            new_paths_blank.add(p)
            new_path_score_blank[p] = path_score_blank[p] * logits[0, t]
        # extend path ending with symbols
        for p in paths:
            if p in new_paths_blank:
                new_path_score_blank[p] += path_score[p] * logits[0, t]
            else:
                new_paths_blank.add(p)
                new_path_score_blank[p] = path_score[p] * logits[0, t]

        print(f"t={t}")
        print(f"PathsWithTerminalSymbol: {paths}")
        print(f"PathsWithTerminalBlank: {paths_blank}")
        print(f"PathScore: {path_score}")
        print(f"BlankPathScore: {path_score_blank}")
        print(f"NewPathsWithTerminalSymbol: {new_paths}")
        print(f"NewPathsWithTerminalBlank: {new_paths_blank}")
        print(f"NewPathScore: {new_path_score}")
        print(f"NewBlankPathScore: {new_path_score_blank}")
    
    # merge identical paths differing only by the final blank
    # new_paths(_blank), new_path_score(blank) -> merged_paths, final_path_score
    merged_paths = new_paths
    final_path_score = new_path_score
    for p in new_paths_blank:
        if p in merged_paths:
            final_path_score[p] += new_path_score_blank[p]
        else:
            merged_paths.add(p)
            final_path_score[p] = new_path_score_blank[p]

    # get the best path
    best_path = ""
    best_score = 0
    for k in final_path_score:
        if best_score < final_path_score[k]:
            best_score = final_path_score[k]
            best_path = k
    return (best_path, final_path_score)
