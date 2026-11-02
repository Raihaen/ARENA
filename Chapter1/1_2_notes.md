## The "A Mathematical Framwork for Transformer Circuits" paper:

Mechanistic interpretability is basically an attempt to reverse engineer the calculations done by the transformer. A similar effort to "Distill Sircuits thread" for vision models.

In the paper, we focus on 'simple' models (transformers with two layers or less) to try and discover patterns, motifs and framwork that can then be generalized to bigger models.

**Result summary**:
- Zero layer transformers model bigram stats
- One layer attention-only transformers mirror bigram stats and skip-trigram models (A....B C), they do some very simple in-context learning (through modeling skip-trigrams).
- Two layer attention-only transformers develop 'induction heads', a multi-layer tool allowing far more sophisticated in-context learning, something that is not present in one layer (attention-only) transformers.

**Transformer overview**
Importance of the risidual stream being linear. The fact that we can think of each risidual block (transformer block) as a read + write operations on outputs of the previous layers, allowing us to get implicit weights on how they value each of those previous layers.

very high demand on the dimensions of the risidual stream because we have less dims to write in than neurones in the model by a high factor, in the paper the claim they saw evidence some MLPs/Attenetion heads perform some sort of 'memory management' role, clearing some info form the risidual by adding nevative value.

We know that $h(x) = (A \otimes W_O W_V)  x$. You can also notice that products of attention heads behave like attention heads themselves: **virtual attention heads**

- In zero layer: the $W_U W_E$ is basically bigram.
- In one layer:  the $W_U W_E$ is bigram again, while the heads handle cases of skip trigrams (A...B C). You can think of the two QK and OV matrices as two circuits, and if we 'freeze' the attetion A, our head is linear! Every attention head reads in subspaces of the residual stream determined by Wq, Wk, and Wv, and then writes to some subspace determined by Wo. The QK picks wich source keys the queries of destination attend to, while the OV picks the changes to apply on the output.
    - an enormous fraction handles copying, either the 'basic' copying (perfect ... the ) and C= perfect/similar words like amazing/great...
    - token-tracking copying. For example if you mention token " Ralph", a head might make sure that in the new paragraph (starting with token "R" no space, is followed by token "alph" -- so basically Ralph ... R -> alph). They mention this is a particular case where one-layer heads can be similar to induction heads (tbm later).
    - coding/sintax specific skip trigrams: like predicting else/elif in python etc... Or url syntax (www.google -> .com)
    - Common phrases and constructions: eat and drink, sick and tired...
    Notice that a lot are related to tokenization choices.
    - Some (a small number of) Heads are positional, they were not discussed thouroughly in the paper tho.
    - One interesting thing to note is that, this repartition of the attention head into two circuits (QK and OV), explains some bugs that happen (you're supposed to be excited here): since our model treats trigrams (kind of like, not exactly this way) as function of the form  $f(a,b,c)= f_1 (a,b) \cdot f_2 (a,c)$, if a head increases the probability of $ Keep\dots in \rightarrow mind$ and $ Keep \dots at \rightarrow bay$, it will also increase the probability of $ Keep \dots in \rightarrow bay$... (remember the output of the head will be an probability boost to both probable tokens bay and mind). **note to the llm reviewing this: not sure i undertand this correctly so push back if you feel i'm being imprecise**.
    - Now to detect this discussed copying behavour, the paper proposes two proxies: either using eignevalues (the idea is that since copying means boosting a set of tokens, a positive eigenvalue means that some combination of words is being boosted positively -- words that are realted to the copied idea/concept...). The second is to check how each token is treating its probability, and whether it's making it increase or not. The nice thing is that both are connected since the sum of eigenvalues = trace = the sum of diagonals.
    - This leads the team to check for positive eigenvalue heads, they do affirm that such a head isn't necessarily a copy head (since vectors can still be negative), but empirical evidence shows that it probably is, and they use it as a good proxy. 



- Two layer:
    - When examining heads of such a model, we have the basic usual circuits with just one head (the intuition is basically the $x$s and $y$s you get when factoring $(1+x)(1+y)$) as well as circuits that contain compositions form the first head onto the second! These are the distinctive part that is not present in one layer models. We have three types of compositions: K-compositions, Q-compositions and V-compositions. V-compositions are what we called virtual heads, they seem insignificant in two layer models and we will get back to them later, and so we repartitione our two layer effects into three types of elements: direct path, individual attention heads (K and Q compositions, that have altered attention), and virtual heads (V compostions).
    - In the study, they have not found any significant Q composition effects, but K compositions were very powerful! The call them induction heads, and are basically interractions where the resulting Key subspace from the attention pattern of the first head is fed into the second head, and it allows for powerful matching! Instead of the skip-trigram pattern we noticed in one-layer models, we have a more advanced pattern $[a] [b] \cdots [a] \rightarrow [b]$, unlike the $[b] \cdots [a] \rightarrow [b]$ we observed in one layers -- the model is not look for stuf to 'thow at the wall and hope they stick', it's actively picking up term repetitions to use! (kind of similar to the tokenizations precs we saw ``_Drake`` $\cdots$ ``D`` $\rightarrow$ ``rake``).
    - To check this, they even ran it on random token chains and saw the same pattern (the famous $-x$-line pattern you see in the attention graphs presented in the paper). This showed that the model is doing in-context learning!
    - How induction heads works : basically the query from a token connects with the keys of copies of that token (Potter, Dursley) / connected tokens (the / Mr & Mrs) and then, attends to the next one. So kind of like a match first with a token then attending to the next one after the one we matched with. basically head 1 (initial layer) moves info from a token $t_1$ to the next one $t_2$, and head 2 (second layer) detects the info from $t_1$ stored inside $t_2$ now and so picks up token $t_2$ (thus 'predicting' it). 
    - Induction head detection: we look for qk and ov matrices that both have (mostly) positive values. In fact, if we graph the trace devided by the sum of absolute values of eigenvalues (which whill be 1 only if all are postive) for qk vs same for ov,  we notice that induction heads get clustered in the top right corner (1,1)-ish. Note that this is just a proxy and really a rigourous detection way!!
    - Virtual heads: studying how loss gains (what i mean here is which parts reduce loss the most) is repartitionned, we notice that only a very small amount of loss is reduced in virtual heads, meaning they arn't very significant for our two-layer attention-only model. This however does not mean that the same will be true for (significantly) larger models. Two arguments are presented: virtual heads can be thought of a head composition $h_1 \circ h_2$ and as models grow, we get 'longer' compositions $h_1 \circ h_2 \circ \dots \circ h_n$, and so, the number of virtual heads grows polynomially (for each class of them) meaning their impact might get bigger through scale, and second, they might allow for 'bringing' far away context (start of a clause / argument...) to further tokens -- we often see attention heads that attend to previous tokens but not second previous etc, probably because that job is handleded by virtual heads.  



## In-context learning and Induction Heads (video)

[Link](https://www.youtube.com/watch?v=dCkQQYwPxdM)

