# When Does an Individual Accept Misinformation?
This repository provides a selection of CCOBRA models for reasoning with misinformation. The models use a cache file and can thus be evaluated quickly, once a parameter setting has been pre-trained. For pre-training (optimizing parameters), an additional script is provided, as well as for transforming the originaly experimental data (source: https://osf.io/tuw89/) into CCOBRA-readable format. Further, optimization hyperparameters for bounded basinhopping are consistent and managed in class "optPars".
## Models
 - Classical Reasoning -- People who think analytically, classify news items more accurately.
 - Motivated Reasoning -- People who think analytically, classify information as correct that is favorable with respect to their own political stance.
 - Fast-And-Frugal Tree: Max -- Decision Tree strategy that implements the Take-The-Best heuristic.
 - Fast-And-Frugal Tree: ZigZag (Z+) -- Decision Tree strategy that implements the Take-The-Best heuristic and alternates exit directions on every cue.
 - Recognition Heuristic -- News items with perceived familiarity over a certain threshold are accepted.
 - Recognition Heuristic (linear) -- News items with high perceived familiarity are accepted more often.
 - Classical Reasoning & Reaction Time -- People who give slow responses, classify news items as incorrect more often.
 - Linear Combination: Sentiment Analysis -- Acceptance probability can be determined by sentiment analysis of a news item headline.

Further:
 - Hybrid model over all above models: --- Selects best predicting model per participant. 

## Dependencies:
ccobra, pandas, numpy, random, math, scipy, empath, os, csv
