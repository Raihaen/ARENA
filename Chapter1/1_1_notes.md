## Transformer implementation

The residual stream is *really* fundamental. It's the central object of the transformer. It's how the model remembers things, moves information between layers for composition, and it's the medium used to store the information that attention moves between positions.

Attention layers are the only bit of a transformer that moves information between positions (i.e. between vectors at different sequence positions in the residual stream).

Attention layers are made up of `n_heads` heads - each with their own parameters, own attention pattern, and own information about how to copy things from source to destination. The heads act independently and additively, we just add their outputs together, and back to the stream.

Q = X @ Wq
K = X @ Wk
this 'shapes' each line (each token) with row vectors of Wq and Wk. Turning each to a set of Token x The query's dimensions, and a Token x the key's dimension.
when we do Q K^T we are measuring how 'significant' and 'close' the queries are to the answers ($q_i \cdot k_i = |k_i| |q_i| \cos{\theta}$). for a row i, we are basically checking how each of the T keys is affecting the query of token i.
Then when we add the mask $M$, we are letting each query compare 'responses' from keys at that position or before it.

This turns then into probability of taking which vi's for each next token. $p_1 v_1 , p_2 v_2 \dots , p_k v_k$ 

Think of the key as like 'type' or 'properties' of the word: say for example dog is a noun, an object of a sentence ... The q asks is there a noun here? the k gives it some nouns it found. and then the vis contain more info like domestic, animal ...


attention as an evolution of convolution NNs: Attention layers are effectively our way of saying to the transformer, "don't impose a prior of locality, but instead develop your own algorithm to figure out which tokens are important to which other tokens in any given sequence."

After the attention comes the MLP layer. This is like 2/3 of the total number of parameters of the model! It's where the thinking happens! You take the properties you got in the previous part (plus the risiduals) and then pass them to the MLP to do the thinking and pick an adequate word (say you think of color sky adjective and it picks blue).

Two ways to interpret the MLP layer (both are connected) (i should revisit this part to understand it deeper)
- See $W_{in}$ as keys and $W_{out}$ as values just like in the transformer.
- Or memory managment: think of the operation $x_2 = x_1 + MLP(x_1)$ as 'removing' from $x_1$ the positive projections on a vector $$v \approx - W^{out \, T}_{[i, :]} = W^{in}_{[:, i]} $$ 

Tied embedings: it might seem motivated at first to have $W_U = W_E^T $, to minimize the number of parameters, but it's not really that good of an idea, the reason is that, because of the residuals, using $W_U = W_E^T $ creates some similarities: $x^T W_U W_U ^ T$ is a symetric map, thus making $P(obama | barack) \Rightarrow P(barack | obama)$ equally as likely, we don't want that... It however is still used for smaller models.

When it comes to interpretability, we can think of layernorm as almost linear, treating $\sqrt{ \sigma (x) + \epsilon}$ as a constant (small perturbations on the prompt won't affect the root) because (not always true tho, more on this later) it won't change the norm a lot.

Positional embedings: We do $x_0 = E(tokens) + P$, where P is also learned just like E. Its goal is to provide positional information to the model (and allow it to learn how to use it).

Parameters vs Activations:
- Parameters are the weights and biases learned during training. They don't change when the model input changes.
- Activations are numbers calculated during the forward pass, in function of the input. We can use hooks to access these values (more on it later).

Training loop:
The typical training loop is :
- For x epoches, you take n samples from the training set, and rerun the training step  K times, then evaluate the model (through calculating accuracy in our case- or other metrics).
- In the training step:
	- you run the model on your batch token sample
	- this creates the logits using the model
	- you turn the logits into probabilities, get the probability of each actual next token (in our current implementation)
	- calculate the cross entropy loss (average)
	- run loss.backward()
	- update your params: for example simple gradient descent param -= learning rate * param.grad, or something like weighted adam...
	- zero out the grads (so they dont cumulate), usually this is done before the loss.backward() step, but in this implementation it's after.
	- step += 1
	and repeat.

