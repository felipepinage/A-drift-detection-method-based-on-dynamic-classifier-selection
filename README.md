# A-drift-detection-method-based-on-dynamic-classifier-selection

Authors: Felipe Pinagé, Eulandos dos Santos, João Gama

Abstract
Machine learning algorithms can be applied to several practical problems, such as
spam, fraud and intrusion detection, and customer preferences, among others. In most
of these problems, data come in streams, whichmean that data distributionmay change
over time, leading to concept drift. The literature is abundant on providing supervised
methods based on error monitoring for explicit drift detection. However, these methods
may become infeasible in some real-world applications—where there is no fully
labeled data available, and may depend on a significant decrease in accuracy to be able
to detect drifts. There are also methods based on blind approaches, where the decision
model is updated constantly. However, this may lead to unnecessary system updates.
In order to overcome these drawbacks, we propose in this paper a semi-supervised
drift detector that uses an ensemble of classifiers based on self-training online learning
and dynamic classifier selection. For each unknown sample, a dynamic selection
strategy is used to choose among the ensemble’s component members, the classifier
most likely to be the correct one for classifying it. The prediction assigned by the
chosen classifier is used to compute an estimate of the error produced by the ensemble
members. The proposed method monitors such a pseudo-error in order to detect
drifts and to update the decision model only after drift detection. The achievement of
this method is relevant in that it allows drift detection and reaction and is applicable
in several practical problems. The experiments conducted indicate that the proposed
method attains high performance and detection rates, while reducing the amount of
labeled data used to detect drift.

Data Mining and Knowledge Discovery
https://doi.org/10.1007/s10618-019-00656-w

Read and share here: https://rdcu.be/bT45D
